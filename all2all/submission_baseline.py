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
        num_tokens = x.shape[0]

        # 1. Determine destinations and unique tokens to send
        dst_ranks_flat = (indices // self.num_local_experts).flatten()
        src_token_indices_flat = torch.arange(
            num_tokens, device=device).repeat_interleave(cfg.experts_per_token)

        # Create a globally unique key for each token: src_rank * max_tokens + src_token_idx
        # Note: We use self.rank as src_rank here.
        global_token_keys_flat = self.rank * \
            cfg.max_num_tokens + src_token_indices_flat

        # Find unique (global_token_key, dst_rank) pairs
        key_dst_rank_pairs = global_token_keys_flat * self.world_size + dst_ranks_flat
        unique_pairs, unique_inverse_indices = torch.unique(
            key_dst_rank_pairs, return_inverse=True)

        # Extract unique keys and their destinations
        unique_global_keys = unique_pairs // self.world_size
        unique_dst_ranks = unique_pairs % self.world_size

        # The token data to send corresponds to the unique keys
        unique_src_token_indices = unique_global_keys % cfg.max_num_tokens
        send_buf_data = x[unique_src_token_indices]

        # 2. First All-to-All: Send unique tokens AND their globally unique keys
        send_counts_data = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts_data.scatter_add_(
            0, unique_dst_ranks, torch.ones_like(unique_dst_ranks, dtype=torch.long))
        recv_counts_data = torch.empty_like(send_counts_data)
        dist.all_to_all_single(recv_counts_data, send_counts_data)

        # Sort data and keys together for sending
        perm_data = torch.argsort(unique_dst_ranks)
        send_buf_data_sorted = send_buf_data[perm_data]
        send_buf_keys_sorted = unique_global_keys[perm_data]

        # Pack keys and data. Key (int64) is viewed as 2 * int32, then cast to data dtype.
        packed_send_buf = torch.cat([
            send_buf_keys_sorted.view(
                torch.int32).reshape(-1, 2).to(cfg.in_dtype),
            send_buf_data_sorted
        ], dim=1)

        total_recv_data = int(recv_counts_data.sum().item())
        packed_recv_buf = torch.empty(
            total_recv_data, packed_send_buf.shape[1], dtype=cfg.in_dtype, device=device)

        dist.all_to_all_single(
            packed_recv_buf, packed_send_buf,
            output_split_sizes=recv_counts_data.tolist(),
            input_split_sizes=send_counts_data.tolist())

        # Unpack received keys and data
        recv_keys_packed = packed_recv_buf[:, :2].to(torch.int32)
        # The reshape was incorrect. We need to view the (N, 2) int32 tensor as an (N,) int64 tensor.
        recv_keys = recv_keys_packed.contiguous().view(torch.int64).flatten()
        recv_data = packed_recv_buf[:, 2:]

        # 3. Second All-to-All for metadata (same as before)
        src_expert_indices_flat = torch.arange(
            cfg.experts_per_token, device=device).repeat(num_tokens)
        global_expert_ids_flat = indices.flatten()
        meta_to_send = torch.stack([
            torch.full_like(global_expert_ids_flat,
                            self.rank, dtype=torch.int32),
            src_token_indices_flat,
            src_expert_indices_flat,
            global_expert_ids_flat
        ], dim=-1)
        send_counts_meta = torch.zeros(
            self.world_size, dtype=torch.long, device=device)
        send_counts_meta.scatter_add_(
            0, dst_ranks_flat, torch.ones_like(dst_ranks_flat, dtype=torch.long))
        recv_counts_meta = torch.empty_like(send_counts_meta)
        dist.all_to_all_single(recv_counts_meta, send_counts_meta)
        perm_meta = torch.argsort(dst_ranks_flat)
        send_buf_meta_sorted = meta_to_send[perm_meta]
        total_recv_meta = int(recv_counts_meta.sum().item())
        recv_meta_buf = torch.empty(
            total_recv_meta, self.META_DIM, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_meta_buf, send_buf_meta_sorted.to(torch.int32),
            output_split_sizes=recv_counts_meta.tolist(),
            input_split_sizes=send_counts_meta.tolist())

        # 4. On receiver: Reconstruct the expert input tensor using the explicit keys
        # Create a lookup map from the received keys to their index in the buffer
        # This is a robust way to map, independent of sorting order.
        map_size = int(recv_keys.max().item()) + 1 if len(recv_keys) > 0 else 0
        # Handle case where a rank receives no tokens
        if map_size > 0:
            # Use a large tensor as a hash map
            # Need to handle potential size issues if keys are very large
            # Let's assume keys are manageable.
            # A better approach for sparse keys would be a real hash map, but this is complex in torch.
            # Let's use sorting, which is robust.
            sorted_recv_keys, sorted_indices = torch.sort(recv_keys)

            # For each metadata entry, calculate its global key
            meta_global_keys = recv_meta_buf[:, 0] * \
                cfg.max_num_tokens + recv_meta_buf[:, 1]

            # Find where each meta_key is in our sorted_recv_keys
            locations = torch.searchsorted(sorted_recv_keys, meta_global_keys)

            # Map locations back to original indices in recv_data
            gather_indices = sorted_indices[locations]

            dispatched_x = recv_data[gather_indices]
        else:
            # This rank received no tokens, create an empty tensor with the correct shape
            dispatched_x = torch.empty(
                0, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)

        return dispatched_x, recv_meta_buf

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

        # Pack expert output and metadata into a single tensor
        packed_send_buf = torch.cat(
            [send_buf, send_meta_buf.to(cfg.out_dtype)], dim=1)

        # Allocate packed receive buffer
        total_recv = int(recv_counts.sum().item())
        packed_recv_buf = torch.empty(total_recv, packed_send_buf.shape[1],
                                      dtype=cfg.out_dtype, device=device)

        # Single all-to-all communication for both data and metadata
        dist.all_to_all_single(
            packed_recv_buf,
            packed_send_buf,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # Unpack received data and metadata
        recv_buf = packed_recv_buf[:, :cfg.hidden_dim].contiguous()
        recv_meta_buf = packed_recv_buf[:, cfg.hidden_dim:].to(
            torch.int32).contiguous()

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
        output.index_add_(0, src_token_indices, weighted_recv_buf)

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
