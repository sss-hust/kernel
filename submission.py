import torch
import torch.distributed as dist
from task import input_t, output_t


# submission.py
import torch
import torch.distributed as dist
from task import input_t, output_t


class OptimizedAllToAll:
    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        assert cfg.num_experts % world_size == 0, "num_experts must be divisible by world_size"

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        device = x.device
        cfg = self.cfg
        num_tokens = x.size(0)

        # Flatten indices and compute destination rank for each expert assignment
        flat_indices = indices.flatten()  # [num_tokens * experts_per_token]
        dst_ranks = flat_indices // self.num_local_experts  # [N]

        # Source info: (src_rank, token_idx, expert_local_idx)
        src_token_idx = torch.arange(
            num_tokens, device=device).repeat_interleave(cfg.experts_per_token)
        expert_local_idx = torch.arange(
            cfg.experts_per_token, device=device).repeat(num_tokens)

        # Pack metadata: [src_rank, token_idx, expert_local_idx, global_expert_id]
        # All as int16 or int32 to save bandwidth
        meta = torch.stack([
            torch.full_like(flat_indices, self.rank, dtype=torch.int16),
            src_token_idx.to(torch.int16),
            expert_local_idx.to(torch.int16),
            flat_indices.to(torch.int16)
        ], dim=1)  # [N, 4]

        # Data to send: repeat x for each expert assignment
        send_data = x[src_token_idx]  # [N, hidden_dim]

        # Count how many to send to each rank
        send_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(
            0, dst_ranks, torch.ones_like(dst_ranks, dtype=torch.long))
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Sort by dst_rank for contiguous communication
        perm = torch.argsort(dst_ranks)
        send_data_sorted = send_data[perm]
        meta_sorted = meta[perm]

        # Pack data and meta into one tensor: [meta (4 * int16), data (float16)]
        # Convert meta to float16 to match data dtype (avoid dtype switching in NCCL)
        meta_fp16 = meta_sorted.to(torch.float16)
        # [N, 4 + hidden_dim]
        packed_send = torch.cat([meta_fp16, send_data_sorted], dim=1)

        total_recv = recv_counts.sum().item()
        packed_recv = torch.empty(
            total_recv, 4 + cfg.hidden_dim, dtype=torch.float16, device=device
        )

        dist.all_to_all_single(
            packed_recv,
            packed_send,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )

        # Unpack
        recv_meta_fp16 = packed_recv[:, :4]
        recv_data = packed_recv[:, 4:]

        # Convert meta back to int16
        recv_meta = recv_meta_fp16.to(torch.int16)

        return recv_data, recv_meta, recv_counts

    def combine(self, expert_y: torch.Tensor, recv_meta: torch.Tensor, weights: torch.Tensor, original_num_tokens: int):
        device = expert_y.device
        cfg = self.cfg

        # recv_meta: [M, 4] -> [src_rank, token_idx, expert_local_idx, global_expert_id]
        src_ranks = recv_meta[:, 0].to(torch.long)

        # Count how many to send back to each rank
        send_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(
            0, src_ranks, torch.ones_like(src_ranks, dtype=torch.long))
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Sort by src_rank
        perm = torch.argsort(src_ranks)
        send_y = expert_y[perm]
        send_meta = recv_meta[perm]

        # Pack output and meta
        packed_send = torch.cat([send_y, send_meta.to(torch.float16)], dim=1)
        total_recv = recv_counts.sum().item()
        packed_recv = torch.empty(
            total_recv, cfg.hidden_dim + 4, dtype=torch.float16, device=device
        )

        dist.all_to_all_single(
            packed_recv,
            packed_send,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )

        # Unpack
        recv_y = packed_recv[:, :cfg.hidden_dim].to(torch.float32)
        recv_meta_back = packed_recv[:, cfg.hidden_dim:].to(torch.int16)

        # Accumulate with weights
        token_indices = recv_meta_back[:, 1].to(torch.long)
        expert_local_indices = recv_meta_back[:, 2].to(torch.long)
        w = weights[token_indices, expert_local_indices].unsqueeze(1)
        weighted_y = recv_y * w

        output = torch.zeros(original_num_tokens, cfg.hidden_dim,
                             dtype=torch.float32, device=device)
        output.index_add_(0, token_indices, weighted_y)

        return output.to(cfg.out_dtype)


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    ata = OptimizedAllToAll(cfg, rank, world_size)

    # Dispatch: 1 all-to-all
    expert_x, meta, _ = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulate expert computation
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)

    # Combine: 1 all-to-all
    y = ata.combine(expert_y, meta, rank_data.weights, rank_data.num_tokens)

    return y
