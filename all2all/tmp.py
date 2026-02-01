import torch
import torch.distributed as dist
from task import input_t, output_t


class OptimizedAllToAll:
    META_DIM = 4  # src_rank, src_token_idx, src_expert_idx, global_expert_id

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        device = x.device
        cfg = self.cfg

        # Flatten indices and x for vectorized processing
        # Shape: (num_tokens * experts_per_token)
        flat_indices = indices.flatten()

        # Shape: (num_tokens * experts_per_token, hidden_dim)
        flat_x = x.repeat_interleave(cfg.experts_per_token, dim=0)

        # Determine destination ranks for each token
        # Shape: (num_tokens * experts_per_token)
        dst_ranks = flat_indices // self.num_local_experts

        # Calculate send_counts using scatter_add for efficiency
        # This avoids Python loops and is much faster on GPU
        ones = torch.ones_like(dst_ranks, dtype=torch.long)
        send_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks, ones)

        # Exchange send/recv counts with all other ranks
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Sort tokens by destination rank to prepare for all_to_all
        # This groups data destined for the same rank together
        perm = torch.argsort(dst_ranks)
        send_buf = flat_x[perm]

        # Prepare metadata for the combine step
        # Shape: (num_tokens, 1)
        token_ids = torch.arange(
            cfg.max_num_tokens, dtype=torch.int32, device=device).unsqueeze(1)
        # Shape: (num_tokens, experts_per_token)
        src_token_indices = token_ids[:x.shape[0]
                                      ].expand(-1, cfg.experts_per_token)

        # Shape: (num_tokens, experts_per_token)
        src_expert_indices = torch.arange(
            cfg.experts_per_token, dtype=torch.int32, device=device).expand(x.shape[0], -1)

        # Create and send metadata
        # src_rank, src_token_idx, src_expert_idx, global_expert_id
        meta_to_send = torch.stack([
            torch.full_like(indices, self.rank),
            src_token_indices,
            src_expert_indices,
            indices
        ], dim=-1).view(-1, self.META_DIM)

        send_meta_buf = meta_to_send[perm]

        # OPTIMIZATION: Merge data and metadata communication
        # Pack data and metadata into a single tensor to reduce communication rounds
        total_tokens = send_buf.shape[0]
        packed_dim = cfg.hidden_dim + self.META_DIM

        # Create packed send buffer: [data, meta_as_float]
        packed_send_buf = torch.empty(total_tokens, packed_dim,
                                      dtype=cfg.in_dtype, device=device)
        packed_send_buf[:, :cfg.hidden_dim] = send_buf
        packed_send_buf[:, cfg.hidden_dim:] = send_meta_buf.to(cfg.in_dtype)

        # Allocate packed receive buffer
        total_recv = int(recv_counts.sum().item())
        packed_recv_buf = torch.empty(total_recv, packed_dim,
                                      dtype=cfg.in_dtype, device=device)

        # Single all-to-all communication for both data and metadata
        dist.all_to_all_single(
            packed_recv_buf,
            packed_send_buf,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # Unpack received data and metadata
        recv_buf = packed_recv_buf[:, :cfg.hidden_dim]
        recv_meta_buf = packed_recv_buf[:, cfg.hidden_dim:].to(torch.int32)

        return recv_buf, recv_meta_buf

    def combine(self, expert_y: torch.Tensor, meta: torch.Tensor, weights: torch.Tensor):
        device = expert_y.device
        cfg = self.cfg

        # Determine destination ranks from metadata
        dst_ranks = meta[:, 0]  # src_rank is the destination for combine

        # Calculate send_counts using scatter_add
        ones = torch.ones_like(dst_ranks, dtype=torch.long)
        send_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks, ones)

        # Exchange send/recv counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Sort by destination rank for efficient communication
        perm = torch.argsort(dst_ranks)
        send_buf = expert_y[perm]
        send_meta_buf = meta[perm]

        # OPTIMIZATION: Merge data and metadata communication
        # Pack expert output and metadata into a single tensor
        total_tokens = send_buf.shape[0]
        packed_dim = cfg.hidden_dim + self.META_DIM

        # Create packed send buffer: [expert_output, meta_as_float]
        packed_send_buf = torch.empty(total_tokens, packed_dim,
                                      dtype=cfg.out_dtype, device=device)
        packed_send_buf[:, :cfg.hidden_dim] = send_buf
        packed_send_buf[:, cfg.hidden_dim:] = send_meta_buf.to(cfg.out_dtype)

        # Allocate packed receive buffer
        total_recv = int(recv_counts.sum().item())
        packed_recv_buf = torch.empty(total_recv, packed_dim,
                                      dtype=cfg.out_dtype, device=device)

        # Single all-to-all communication for both data and metadata
        dist.all_to_all_single(
            packed_recv_buf,
            packed_send_buf,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # Unpack received data and metadata
        recv_buf = packed_recv_buf[:, :cfg.hidden_dim]
        recv_meta_buf = packed_recv_buf[:, cfg.hidden_dim:].to(torch.int32)

        # Vectorized weighted sum using scatter_add_
        # This is the key optimization for the combine step
        output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim,
                             dtype=torch.float32, device=device)

        # Extract necessary info from metadata
        src_token_indices = recv_meta_buf[:, 1]
        src_expert_indices = recv_meta_buf[:, 2]

        # Gather weights corresponding to the received tokens
        w = weights[src_token_indices, src_expert_indices].unsqueeze(1)

        # Apply weights
        weighted_recv_buf = recv_buf.to(torch.float32) * w

        # Use index_add_ for potentially better memory access patterns
        # output.index_add_(0, src_token_indices, weighted_recv_buf)
        sort_idx = src_token_indices.argsort()

        # 2. 重排数据 (Gather)：把数据搬到连续的内存里
        # sorted_data 的形状是 (Total_Recv, Hidden)
        sorted_data = weighted_recv_buf[sort_idx]

        # 3. 变形 (View)：利用显存的连续性，把数据看作 (Num_Tokens, TopK, Hidden)
        # 前提：我们知道每个 Token 都有 TopK 个结果回来
        # 这一步是 0 开销的
        sorted_view = sorted_data.view(
            cfg.max_num_tokens, cfg.experts_per_token, cfg.hidden_dim)

        # 4. 规约 (Sum)：在连续内存上做加法
        # GPU 对这种操作有极强的优化 (SIMD)，且没有任何原子锁冲突
        # 结果直接写入 Output，只写一次！
        output = sorted_view.sum(dim=1).to(cfg.out_dtype)

        return output.to(cfg.out_dtype)


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    ata = OptimizedAllToAll(cfg, rank, world_size)

    # Dispatch tokens to experts
    dispatched_x, meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulate expert computation
    expert_y = dispatched_x.to(cfg.out_dtype) * (1 + rank)

    # Combine results from experts
    y = ata.combine(expert_y, meta, rank_data.weights)

    return y[: rank_data.num_tokens]
