import math
import os
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, open_dict
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
import models
import utils
from torch.distributions.categorical import Categorical

from env import MultiTrajectoryTSP, TSPDataset


class TSPModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        if cfg.node_dim > 2:
            assert (
                "noAug" in cfg.val_type
            ), "High-dimension TSP doesn't support augmentation"

        ## Encoder model
        if cfg.encoder_type == "mha":
            self.encoder = models.MHAEncoder(
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                embedding_dim=cfg.embedding_dim,
                input_dim=24 if cfg.data_augment else 2,
                add_init_projection=cfg.add_init_projection,
            )
        elif cfg.encoder_type == "revmha":
            self.encoder = models.RevMHAEncoder(
                n_layers=cfg.n_layers,
                n_heads=cfg.n_heads,
                embedding_dim=cfg.embedding_dim,
                input_dim=24 if cfg.data_augment else 2,
                intermediate_dim=cfg.embedding_dim * 4,
                add_init_projection=cfg.add_init_projection,
            )

        ## Decoder model
        if cfg.decoder_type == "DecoderForLarge":
            self.decoder = models.DecoderForLarge(
                embedding_dim=cfg.embedding_dim,
                n_heads=cfg.n_heads,
                tanh_clipping=cfg.tanh_clipping,
                multi_pointer=cfg.multi_pointer,
                multi_pointer_level=cfg.multi_pointer_level,
                add_more_query=cfg.add_more_query,
            )
        else:
            self.decoder = models.Decoder(
                embedding_dim=cfg.embedding_dim,
                n_heads=cfg.n_heads,
                tanh_clipping=cfg.tanh_clipping,
            )

        self.cfg = cfg
        self.save_hyperparameters(cfg)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.learning_rate * len(self.cfg.gpus),
            weight_decay=self.cfg.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=1.0
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"}]

    def train_dataloader(self):
        self.group_size = self.cfg.group_size
        dataset = TSPDataset(
            size=self.cfg.graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.epoch_size,
            data_distribution=self.cfg.data_distribution,
        )
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=self.cfg.train_batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        dataset = TSPDataset(
            size=self.cfg.graph_size,
            node_dim=self.cfg.node_dim,
            num_samples=self.cfg.val_size,
            data_distribution=self.cfg.data_distribution,
            data_path=self.cfg.val_data_path,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.val_batch_size,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def training_step(self, batch, _):
        B, N, _ = batch.shape
        G = self.group_size

        batch_idx_range = torch.arange(B)[:, None].expand(B, G)
        group_idx_range = torch.arange(G)[None, :].expand(B, G)

        env = MultiTrajectoryTSP(batch.cpu())  # to cpu for env
        s, r, d = env.reset(group_size=G)  # reset env

        # Data argument
        if self.cfg.data_augment:
            batch = utils.data_augment(batch)

        # Encode
        embeddings = self.encoder(batch)
        self.decoder.reset(batch, embeddings, G)  # decoder reset

        entropy_list = []
        log_prob = torch.zeros(B, G, device=self.device)
        while not d:
            if s.current_node is None:
                first_action = torch.randperm(N)[None, :G].expand(B, G)
                s, r, d = env.step(first_action)
                continue
            else:
                last_node = s.current_node

            action_probs = self.decoder(
                last_node.to(self.device), s.ninf_mask.to(self.device), s.selected_count
            )
            m = Categorical(action_probs.reshape(B * G, -1))
            entropy_list.append(m.entropy().mean().item())
            action = m.sample().view(B, G)
            chosen_action_prob = (
                action_probs[batch_idx_range, group_idx_range, action].reshape(B, G)
                + 1e-8
            )
            log_prob += chosen_action_prob.log()
            s, r, d = env.step(action.cpu())

        # Note that when G == 1, we can only use the PG without baseline so far
        r_trans = r.to(self.device)  # -1/torch.exp(r)
        if self.cfg.divide_std:
            advantage = (
                (r_trans - r_trans.mean(dim=1, keepdim=True))
                / (r_trans.std(dim=1, unbiased=False, keepdim=True) + 1e-8)
                if G != 1
                else r_trans
            )
        else:
            advantage = (
                (r_trans - r_trans.mean(dim=1, keepdim=True)) if G != 1 else r_trans
            )
        loss = (-advantage * log_prob).mean()

        length_max = -r.max(dim=1)[0].mean().clone().detach().item()
        lenght_mean = -r.mean(1).mean().clone().detach().item()
        entropy_mean = sum(entropy_list) / len(entropy_list)
        adv = advantage.abs().mean().clone().detach().item()

        self.log(
            name="length_max", value=length_max, prog_bar=True,
        )  # sync_dist=True
        self.log(
            name="length_mean", value=lenght_mean, prog_bar=True,
        )
        self.log(
            name="entropy", value=entropy_mean, prog_bar=True,
        )
        self.log(name="adv", value=adv, prog_bar=True)

        embed_max = embeddings.abs().max()
        self.log(
            name="embed_max", value=embed_max, prog_bar=True,
        )

        assert torch.isnan(loss).sum() == 0, print("loss is nan!")

        return {"loss": loss, "length": length_max, "entropy": entropy_mean}

    def training_epoch_end(self, outputs):
        outputs = torch.as_tensor([item["length"] for item in outputs])
        self.train_length_mean = outputs.mean().item()
        self.train_length_std = outputs.std().item()

    def validate_all(self, batch, val_type=None, return_pi=False):
        batch = utils.augment_xy_data_by_8_fold(batch)
        B, N, _ = batch.shape
        G = N

        env = MultiTrajectoryTSP(batch)
        s, r, d = env.reset(group_size=G)

        if self.cfg.data_augment:
            batch = utils.data_augment(batch)

        embeddings = self.encoder(batch)
        self.decoder.reset(batch, embeddings, G, trainging=False)

        first_action = torch.randperm(N)[None, :G].expand(B, G).to(self.device)
        pi = first_action[..., None]
        s, r, d = env.step(first_action)

        while not d:
            action_probs = self.decoder(
                s.current_node.to(self.device),
                s.ninf_mask.to(self.device),
                s.selected_count,
            )
            action = action_probs.argmax(dim=2)
            pi = torch.cat([pi, action[..., None]], dim=-1)
            s, r, d = env.step(action)

        B = round(B / 8)
        reward = r.reshape(8, B, G)
        pi = pi.reshape(8, B, G, N)

        reward_greedy = reward[0, :, 0]
        reward_ntraj, idx_dim_ntraj = reward[0, :, :].max(dim=-1)
        max_reward_aug_ntraj, idx_dim_2 = reward.max(dim=2)
        max_reward_aug_ntraj, idx_dim_0 = max_reward_aug_ntraj.max(dim=0)

        if return_pi:
            best_pi_greedy = pi[0, :, 0]
            idx_dim_ntraj = idx_dim_ntraj.reshape(B, 1, 1)
            best_pi_n_traj = pi[0, :, :].gather(
                1, idx_dim_ntraj.repeat(1, 1, N)
            )  # (B,G,N)
            idx_dim_0 = idx_dim_0.reshape(1, B, 1, 1)
            idx_dim_2 = idx_dim_2.reshape(8, B, 1, 1).gather(0, idx_dim_0)
            best_pi_aug_ntraj = pi.gather(0, idx_dim_0.repeat(1, 1, G, N))
            best_pi_aug_ntraj = best_pi_aug_ntraj.gather(
                2, idx_dim_2.repeat(1, 1, 1, N)
            )
            return -max_reward_aug_ntraj, best_pi_aug_ntraj.squeeze()
        return {
            "max_reward_aug_ntraj": -max_reward_aug_ntraj,
            "reward_greedy": -reward_greedy,
            "reward_ntraj": -reward_ntraj,
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outputs = self.validate_all(batch)
        return outputs

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_validation_epoch_start(self) -> None:
        self.validation_start_time = time.time()

    def validation_epoch_end(self, outputs):
        max_reward_aug_ntraj = [item["max_reward_aug_ntraj"] for item in outputs]
        reward_greedy = [item["reward_greedy"] for item in outputs]
        reward_ntraj = [item["reward_ntraj"] for item in outputs]

        self.max_reward_aug_ntraj = torch.cat(max_reward_aug_ntraj).mean().item()
        self.max_reward_aug_ntraj_std = torch.cat(max_reward_aug_ntraj).std().item()
        self.reward_greedy = torch.cat(reward_greedy).mean().item()
        self.reward_greedy_std = torch.cat(reward_greedy).std().item()
        self.reward_ntraj = torch.cat(reward_ntraj).mean().item()
        self.reward_ntraj_std = torch.cat(reward_ntraj).std().item()

    def on_validation_epoch_end(self):
        validation_time = time.time() - self.validation_start_time

        self.log_dict(
            {
                # "train_graph_size": self.train_graph_size,
                # "train_length": self.train_length_mean,
                "max_reward_aug_ntraj": self.max_reward_aug_ntraj,
                "reward_greedy": self.reward_greedy,
                "reward_ntraj": self.reward_ntraj,
            },
            on_epoch=True,
            on_step=False,
        )

        self.print(
            f"\nEpoch {self.current_epoch}: ",
            # f'train_graph_size={self.train_graph_size}, ',
            # 'train_performance={:.03f}±{:.03f}, '.format(self.train_length_mean, self.train_length_std),
            "max_reward_aug_ntraj={:.03f}±{:.03f}, ".format(
                self.max_reward_aug_ntraj, self.max_reward_aug_ntraj_std
            ),
            "reward_greedy={:.03f}±{:.3f}, ".format(
                self.reward_greedy, self.reward_greedy_std
            ),
            "reward_ntraj={:.03f}±{:.03f}, ".format(
                self.reward_ntraj, self.reward_ntraj_std
            ),
            "validation time={:.03f}".format(validation_time),
        )


@hydra.main(config_name="config")
def run(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    cfg.run_name = cfg.run_name or cfg.default_run_name
    if cfg.save_dir is None:
        root_dir = (os.getcwd(),)
    elif os.path.isabs(cfg.save_dir):
        root_dir = cfg.save_dir
    else:
        root_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.save_dir)
    root_dir = os.path.join(root_dir, f"{cfg.run_name}")
    with open_dict(cfg):
        cfg.root_dir = root_dir

    cfg.val_data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.val_data_path)

    # build  TSPModel
    tsp_model = TSPModel(cfg)
    checkpoint_callback = ModelCheckpoint(
        monitor="reward_ntraj",
        dirpath=os.path.join(root_dir, "checkpoints"),
        filename=cfg.encoder_type + "{epoch:02d}-{val_length:.2f}",
        save_top_k=3,
        save_last=True,
        mode="min",
        every_n_val_epochs=2,
    )

    # set up loggers
    loggers = []
    if cfg.tensorboard:
        tb_logger = TensorBoardLogger("logs")
        loggers.append(tb_logger)
    # wandb logger
    if cfg.wandb:
        os.makedirs(os.path.join(os.path.abspath(root_dir), "wandb"))
        wandb_logger = WandbLogger(
            name=cfg.run_name,
            save_dir=root_dir,
            project=cfg.wandb_project,
            log_model=True,
            save_code=True,
            group=time.strftime("%Y%m%d", time.localtime()),
            tags=cfg.default_run_name.split("-")[:-1],
        )
        wandb_logger.log_hyperparams(cfg)
        wandb_logger.watch(tsp_model)
        loggers.append(wandb_logger)

    # auto resumed training
    last_ckpt_path = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        resume = last_ckpt_path
    elif cfg.load_path:
        resume = cfg.load_path
    else:
        resume = None

    # build trainer
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        gpus=cfg.gpus,
        accelerator="dp",
        precision=cfg.precision,
        max_epochs=cfg.total_epoch,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0,
        resume_from_checkpoint=resume,
        logger=loggers,
        callbacks=[checkpoint_callback],
    )

    # training and save ckpt
    trainer.fit(tsp_model)
    trainer.save_checkpoint(os.path.join(root_dir, "checkpoint.ckpt"))


if __name__ == "__main__":
    run()
