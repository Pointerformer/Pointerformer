import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.utils.checkpoint
import utils
import revtorch as rv


class MHAEncoderLayer(torch.nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.Wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm1 = torch.nn.BatchNorm1d(embedding_dim)
        self.norm2 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, x, mask=None):
        q = utils.make_heads(self.Wq(x), self.n_heads)
        k = utils.make_heads(self.Wk(x), self.n_heads)
        v = utils.make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(utils.multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class MHAEncoder(torch.nn.Module):
    def __init__(
        self, n_layers, n_heads, embedding_dim, input_dim, add_init_projection=True
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.attn_layers = torch.nn.ModuleList(
            [
                MHAEncoderLayer(embedding_dim=embedding_dim, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        for idx, layer in enumerate(self.attn_layers):
            x = layer(x, mask)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self, embedding_dim, n_heads=8, tanh_clipping=10.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)
        # self.query_fc = torch.nn.Linear(embedding_dim*2, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state

    def reset(self, coordinates, embeddings, G, trainging=False):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]
        self.coordinates = coordinates
        self.embeddings = embeddings
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.glimpse_k = utils.make_heads(self.Wk(embeddings), self.n_heads)
        self.glimpse_v = utils.make_heads(self.Wv(embeddings), self.n_heads)
        self.logit_k = embeddings.transpose(1, 2)
        # self.group_ninf_mask = group_ninf_mask

    def forward(self, last_node, group_ninf_mask, step):
        self.group_ninf_mask = group_ninf_mask
        B, N, H = self.embeddings.shape
        G = self.group_ninf_mask.size(1)

        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)

        # q_graph.shape = [B, n_heads, 1, key_dim]
        # q_first.shape = q_last.shape = [B, n_heads, G, key_dim]
        if self.q_first is None:
            # self.q_first = utils.make_heads(self.Wq_first(last_node_embedding),
            #                                 self.n_heads)
            self.q_first = self.Wq_first(last_node_embedding)

        q_last = self.Wq_last(last_node_embedding)
        # glimpse_q.shape = [B, n_heads, G, key_dim]
        glimpse_q = self.q_first + q_last + self.q_graph

        glimpse_q = utils.make_heads(glimpse_q, self.n_heads)
        if self.n_decoding_neighbors is not None:
            D = self.coordinates.size(-1)
            K = torch.count_nonzero(self.group_ninf_mask[0, 0] == 0.0).item()
            K = min(self.n_decoding_neighbors, K)
            last_node_coordinate = self.coordinates.gather(
                dim=1, index=last_node.unsqueeze(-1).expand(B, G, D)
            )
            distances = torch.cdist(last_node_coordinate, self.coordinates)
            distances[self.group_ninf_mask == -np.inf] = np.inf
            indices = distances.topk(k=K, dim=-1, largest=False).indices
            glimpse_mask = torch.ones_like(self.group_ninf_mask) * (-np.inf)
            glimpse_mask.scatter_(
                dim=-1, index=indices, src=torch.zeros_like(glimpse_mask)
            )
        else:
            glimpse_mask = self.group_ninf_mask

        # q_last_nhead=utils.make_heads(q_last,self.n_heads)
        attn_out = utils.multi_head_attention(
            q=glimpse_q, k=self.glimpse_k, v=self.glimpse_v, mask=glimpse_mask,
        )

        # mha_out.shape = [B, G, H]
        # score.shape = [B, G, N]
        final_q = self.multi_head_combine(attn_out)
        score = torch.matmul(final_q, self.logit_k)

        score_clipped = torch.tanh(score) * self.tanh_clipping
        score_masked = score_clipped + self.group_ninf_mask

        probs = F.softmax(score_masked, dim=2)

        assert (probs == probs).all(), "Probs should not contain any nans!"
        return probs


"""
RevMHAEncoder
"""


class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states: Tensor):

        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        hidden_states_t = hidden_states.transpose(0, 1)
        mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)[
            0
        ].transpose(0, 1)

        return mha_output


class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: Tensor):
        hidden_states = (
            self.output_layer_norm(hidden_states.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        intermediate_output = self.feed_forward(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output


class RevMHAEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        input_dim: int,
        intermediate_dim: int,
        add_init_projection=True,
    ):
        super().__init__()
        if add_init_projection or input_dim != embedding_dim:
            self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        self.num_hidden_layers = n_layers
        blocks = []
        for _ in range(n_layers):
            f_func = MHABlock(embedding_dim, n_heads)
            g_func = FFBlock(embedding_dim, intermediate_dim)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    def forward(self, x: Tensor, mask=None):
        if hasattr(self, "init_projection_layer"):
            x = self.init_projection_layer(x)
        x = torch.cat([x, x], dim=-1)
        out = self.sequence(x)
        return torch.stack(out.chunk(2, dim=-1))[-1]


class DecoderForLarge(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        n_heads=8,
        tanh_clipping=10.0,
        multi_pointer=1,
        multi_pointer_level=1,
        add_more_query=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.W_visited = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.Wv = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.multi_head_combine = torch.nn.Linear(embedding_dim, embedding_dim)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state
        self.multi_pointer = multi_pointer  #
        self.multi_pointer_level = multi_pointer_level
        self.add_more_query = add_more_query

    def reset(self, coordinates, embeddings, G, trainging=True):
        # embeddings.shape = [B, N, H]
        # graph_embedding.shape = [B, 1, H]
        # q_graph.hape = [B, n_heads, 1, key_dim]
        # glimpse_k.shape = glimpse_v.shape =[B, n_heads, N, key_dim]
        # logit_k.shape = [B, H, N]
        # group_ninf_mask.shape = [B, G, N]

        B, N, H = embeddings.shape
        # G = group_ninf_mask.size(1)

        self.coordinates = coordinates  # [:,:2]
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, G, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.logit_k = embeddings.transpose(1, 2)
        if self.multi_pointer > 1:
            self.logit_k = utils.make_heads(
                self.Wk(embeddings), self.multi_pointer
            ).transpose(
                2, 3
            )  # [B, n_heads, key_dim, N]

    def forward(self, last_node, group_ninf_mask, S):
        B, N, H = self.embeddings.shape
        G = group_ninf_mask.size(1)

        # Get last node embedding
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = self.Wq_last(last_node_embedding)

        # Get frist node embedding
        if self.q_first is None:
            self.q_first = self.Wq_first(last_node_embedding)
        group_ninf_mask = group_ninf_mask.detach()

        mask_visited = group_ninf_mask.clone()
        mask_visited[mask_visited == -np.inf] = 1.0
        q_visited = self.W_visited(torch.bmm(mask_visited, self.embeddings) / N)
        D = self.coordinates.size(-1)
        last_node_coordinate = self.coordinates.gather(
            dim=1, index=last_node.unsqueeze(-1).expand(B, G, D)
        )
        distances = torch.cdist(last_node_coordinate, self.coordinates)

        if self.add_more_query:
            final_q = q_last + self.q_first + self.q_graph + q_visited
        else:
            final_q = q_last + self.q_first + self.q_graph

        if self.multi_pointer > 1:
            final_q = utils.make_heads(
                self.wq(final_q), self.n_heads
            )  # (B,n_head,G,H)  (B,n_head,H,N)
            score = (torch.matmul(final_q, self.logit_k) / math.sqrt(H)) - (
                distances / math.sqrt(2)
            ).unsqueeze(
                1
            )  # (B,n_head,G,N)
            if self.multi_pointer_level == 1:
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8

                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs
        else:
            score = torch.matmul(final_q, self.logit_k) / math.sqrt(
                H
            ) - distances / math.sqrt(2)
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        score_masked = score_clipped + mask_prob
        probs = F.softmax(score_masked, dim=2)

        return probs
