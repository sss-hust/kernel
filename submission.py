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

        # Create separate streams for compute and communication
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()

        # Pre-allocate buffers for better memory management
        device = torch.cuda.current_device()
        max_buffer_size = cfg.max_num_tokens * cfg.experts_per_token * world_size

        # Pre-allocate commonly used tensors to reduce allocation overhead
        self._send_counts = torch.zeros(
            world_size, dtype=torch.long, device=device)
        self._recv_counts = torch.zeros(
            world_size, dtype=torch.long, device=device)

        # Pre-allocate workspace tensors for metadata processing
        self._workspace_meta = torch.empty(
            max_buffer_size, self.META_DIM, dtype=torch.int32, device=device)

        # Cache for frequently used tensors
        self._token_indices_cache = {}
        self._expert_indices_cache = {}

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        device = x.device
        cfg = self.cfg

        num_tokens, experts_per_token = x.shape[0], cfg.experts_per_token
        total_experts_tokens = num_tokens * experts_per_token

        # Pre-allocate all tensors to avoid multiple allocations
        flat_indices = indices.view(-1)
        dst_ranks = flat_indices // self.num_local_experts

        # Vectorized creation of expanded data without repeat_interleave
        # This is more memory efficient
        token_repeat_indices = torch.arange(
            num_tokens, device=device, dtype=torch.long).repeat_interleave(experts_per_token)
        flat_x = x[token_repeat_indices]

        # Calculate send_counts using bincount for better performance
        send_counts = torch.bincount(
            dst_ranks, minlength=self.world_size).to(torch.long)

        # Exchange send/recv counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Use stable sort for better cache locality
        perm = torch.argsort(dst_ranks, stable=True)

        # Prepare data and metadata in parallel streams
        with torch.cuda.stream(self.compute_stream):
            # Create metadata more efficiently
            expert_indices = torch.arange(
                experts_per_token, device=device, dtype=torch.int32).repeat(num_tokens)

            # Pack metadata into a single tensor to reduce memory footprint
            # Format: [src_rank, src_token_idx, src_expert_idx, global_expert_id]
            meta_to_send = torch.stack([
                torch.full((total_experts_tokens,), self.rank,
                           device=device, dtype=torch.int32),
                token_repeat_indices.to(torch.int32),
                expert_indices,
                flat_indices.to(torch.int32)
            ], dim=1)

            # Apply permutation to both data and metadata
            send_buf = flat_x[perm]
            send_meta_buf = meta_to_send[perm]

        # Allocate receive buffers
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim,
                               dtype=cfg.in_dtype, device=device)
        recv_meta_buf = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # Perform all-to-all communication
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(self.compute_stream)

            # Use async communication if available
            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts.tolist(),
            )

            # Communication for metadata
            dist.all_to_all_single(
                recv_meta_buf.view(-1),
                send_meta_buf.view(-1),
                output_split_sizes=[
                    c * self.META_DIM for c in recv_counts.tolist()],
                input_split_sizes=[
                    c * self.META_DIM for c in send_counts.tolist()],
            )

        return recv_buf, recv_meta_buf, send_buf, send_meta_buf

    def combine(self, expert_y: torch.Tensor, meta: torch.Tensor, weights: torch.Tensor):
        device = expert_y.device
        cfg = self.cfg

        # Determine destination ranks from metadata
        dst_ranks = meta[:, 0]  # src_rank is the destination for combine

        # Calculate send_counts using bincount for better performance
        send_counts = torch.bincount(
            dst_ranks, minlength=self.world_size).to(torch.long)

        # Exchange send/recv counts
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        # Use stable sort for better cache locality
        perm = torch.argsort(dst_ranks, stable=True)

        # Prepare data for communication in compute stream
        with torch.cuda.stream(self.compute_stream):
            send_buf = expert_y[perm]
            send_meta_buf = meta[perm]

        # Allocate receive buffers
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim,
                               dtype=cfg.out_dtype, device=device)
        recv_meta_buf = torch.empty(
            total_recv, self.META_DIM, dtype=torch.int32, device=device)

        # Perform all-to-all communication
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_stream(self.compute_stream)

            dist.all_to_all_single(
                recv_buf,
                send_buf,
                output_split_sizes=recv_counts.tolist(),
                input_split_sizes=send_counts.tolist(),
            )
            dist.all_to_all_single(
                recv_meta_buf.view(-1),
                send_meta_buf.view(-1),
                output_split_sizes=[
                    c * self.META_DIM for c in recv_counts.tolist()],
                input_split_sizes=[
                    c * self.META_DIM for c in send_counts.tolist()],
            )

        # Optimized weighted aggregation using scatter_add_
        with torch.cuda.stream(self.compute_stream):
            self.compute_stream.wait_stream(self.comm_stream)

            # Pre-allocate output tensor
            output = torch.zeros(
                cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)

            # Extract token and expert indices from metadata
            src_token_indices = recv_meta_buf[:, 1]
            src_expert_indices = recv_meta_buf[:, 2]

            # Efficiently gather weights and apply them
            # Use advanced indexing to get weights in one operation
            w = weights[src_token_indices, src_expert_indices].unsqueeze(-1)

            # Apply weights to received data (convert to float32 for better precision)
            weighted_data = recv_buf.to(torch.float32) * w

            # Use scatter_add for efficient aggregation
            # This is much faster than index_add_ for sparse operations
            output.scatter_add_(
                0, src_token_indices.unsqueeze(-1).expand(-1, cfg.hidden_dim), weighted_data)

        return output.to(cfg.out_dtype), send_buf, send_meta_buf


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    ata = OptimizedAllToAll(cfg, rank, world_size)

    # Start dispatch in compute stream for better overlap
    with torch.cuda.stream(ata.compute_stream):
        dispatched_x, meta, _send_buf, _send_meta_buf = ata.dispatch(
            rank_data.x, rank_data.indices)

    # Wait for communication to finish before expert computation
    torch.cuda.current_stream().wait_stream(ata.comm_stream)

    # Expert computation with overlap preparation
    with torch.cuda.stream(ata.compute_stream):
        # Simulate expert computation
        expert_y = dispatched_x.to(cfg.out_dtype) * (1 + rank)

        # Start combine operation immediately after expert computation
        y, _send_buf_combine, _send_meta_combine = ata.combine(
            expert_y, meta, rank_data.weights)

    # Final synchronization - wait for all operations to complete
    torch.cuda.current_stream().wait_stream(ata.compute_stream)
    torch.cuda.current_stream().wait_stream(ata.comm_stream)

    return y[: rank_data.num_tokens]
