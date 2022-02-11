import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

import os


class GroupState:
    def __init__(self, group_size, x):
        # x.shape = [B, N, 2]
        self.batch_size = x.size(0)
        self.group_size = group_size
        self.device = x.device

        self.selected_count = 0
        # current_node.shape = [B, G]
        self.current_node = None
        # selected_node_list.shape = [B, G, selected_count]
        self.selected_node_list = torch.zeros(
            x.size(0), group_size, 0, device=x.device
        ).long()
        # ninf_mask.shape = [B, G, N]
        self.ninf_mask = torch.zeros(x.size(0), group_size, x.size(1), device=x.device)

    def move_to(self, selected_idx_mat):
        # selected_idx_mat.shape = [B, G]
        self.selected_count += 1
        self.current_node = selected_idx_mat
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected_idx_mat[:, :, None]), dim=2
        )
        self.ninf_mask.scatter_(
            dim=-1, index=selected_idx_mat[:, :, None], value=-torch.inf
        )


class MultiTrajectoryTSP:
    def __init__(self, x, x_raw=None, integer=False):
        self.integer = integer
        if x_raw is None:
            self.x_raw = x.clone()
        else:
            self.x_raw = x_raw
        self.x = x
        self.batch_size = self.B = x.size(0)
        self.graph_size = self.N = x.size(1)
        self.node_dim = self.C = x.size(2)
        self.group_size = self.G = None
        self.group_state = None

    def reset(self, group_size):
        self.group_size = group_size
        self.group_state = GroupState(group_size=group_size, x=self.x)
        reward = None
        done = False
        return self.group_state, reward, done

    def step(self, selected_idx_mat):
        # move state
        self.group_state.move_to(selected_idx_mat)

        # returning values
        done = self.group_state.selected_count == self.graph_size
        if done:
            reward = -self._get_group_travel_distance()  # note the minus sign!
        else:
            reward = None
        return self.group_state, reward, done

    def _get_group_travel_distance(self):
        # ordered_seq.shape = [B, G, N, C]
        shp = (self.B, self.group_size, self.N, self.C)
        gathering_index = self.group_state.selected_node_list.unsqueeze(3).expand(*shp)
        seq_expanded = self.x_raw[:, None, :, :].expand(*shp)
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        # segment_lengths.size = [B, G, N]
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        if self.integer:
            group_travel_distances = segment_lengths.round().sum(2)
        else:
            group_travel_distances = segment_lengths.sum(2)
        return group_travel_distances


def readDataFile(filePath):
    """
        read validation dataset from "https://github.com/Spider-scnu/TSP"
    """
    res = []
    with open(filePath, "r") as fp:
        datas = fp.readlines()
        for data in datas:
            data = [float(i) for i in data.split("o")[0].split()]
            loc_x = torch.FloatTensor(data[::2])
            loc_y = torch.FloatTensor(data[1::2])
            data = torch.stack([loc_x, loc_y], dim=1)
            res.append(data)
    res = torch.stack(res, dim=0)
    return res


def readTSPLib(filePath):
    """
        read TSPLib
    """
    data_trans, data_raw = [], []
    with open(filePath, "r") as fp:
        loc_x = []
        loc_y = []
        datas = fp.readlines()
        for data in datas:
            if ":" in data or "EOF" in data or "NODE_COORD_SECTION" in data:
                continue
            data = [float(i) for i in data.split()]
            if len(data) == 3:
                loc_x.append(data[1])
                loc_y.append(data[2])
        loc_x = torch.FloatTensor(loc_x)
        loc_y = torch.FloatTensor(loc_y)

        data = torch.stack([loc_x, loc_y], dim=1)
        data_raw.append(data)

        mx = loc_x.max() - loc_x.min()
        my = loc_y.max() - loc_y.min()
        data = torch.stack([loc_x - loc_x.min(), loc_y - loc_y.min()], dim=1)
        data = data / max(mx, my)
        data_trans.append(data)

    data_trans = torch.stack(data_trans, dim=0)
    data_raw = torch.stack(data_raw, dim=0)
    return data_trans, data_raw


def readTSPLibOpt(opt_path):
    with open(opt_path, "r") as fp:
        datas = fp.readlines()
        tours = []
        for data in datas:
            if ":" in data or "-1" in data or "TOUR_SECTION" in data or "EOF" in data:
                continue
            tours.extend([int(i) - 1 for i in data.split()])
        tours = np.array(tours, dtype=np.int)
    return tours


class TSPDataset(Dataset):
    def __init__(
        self,
        size=50,
        node_dim=2,
        num_samples=100000,
        data_distribution="uniform",
        data_path=None,
    ):
        super(TSPDataset, self).__init__()
        if data_distribution == "uniform":
            self.data = torch.rand(num_samples, size, node_dim)
        elif data_distribution == "normal":
            self.data = torch.randn(num_samples, size, node_dim)
        self.size = num_samples
        if not data_path is None:
            if data_path.split(".")[-1] == "tsp":
                self.data, data_raw = readTSPLib(data_path)
                opt_path = data_path.replace(".tsp", ".opt.tour")
                print(opt_path)
                if os.path.exists(opt_path):
                    self.opt_route = readTSPLibOpt(opt_path)
                    tmp = np.roll(self.opt_route, -1)
                    d = data_raw[0, self.opt_route] - data_raw[0, tmp]
                    self.opt = np.linalg.norm(d, axis=-1).sum()
                else:
                    self.opt = -1
                self.data = data_raw

            else:
                self.data = readDataFile(data_path)
            self.size = self.data.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    TSPDataset(data_path="./data/ALL_tsp/att48.tsp")
