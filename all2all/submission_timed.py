import torch
import torch.distributed as dist
from task import input_t, output_t
import time
from collections import defaultdict
from typing import Dict, List


class TimingContext:
    """High-precision timing context manager using CUDA events"""

    def __init__(self, name: str, timing_stats: Dict):
        self.name = name
        self.timing_stats = timing_stats
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        # Use CUDA events for precise GPU timing without CPU-GPU synchronization overhead
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize()  # Only synchronize at the end to get timing
        elapsed_time = self.start_event.elapsed_time(
            self.end_event)  # milliseconds

        if self.name not in self.timing_stats:
            self.timing_stats[self.name] = []
        self.timing_stats[self.name].append(elapsed_time)


class OptimizedAllToAllTimed:
    META_DIM = 4  # src_rank, src_token_idx, src_expert_idx, global_expert_id

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.timing_stats = defaultdict(list)

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        device = x.device
        cfg = self.cfg

        with TimingContext("dispatch_total", self.timing_stats):
            with TimingContext("dispatch_preparation", self.timing_stats):
                # Flatten indices and x for vectorized processing
                # Shape: (num_tokens * experts_per_token)
                flat_indices = indices.flatten()

                # Shape: (num_tokens * experts_per_token, hidden_dim)
                flat_x = x.repeat_interleave(cfg.experts_per_token, dim=0)

                # Determine destination ranks for each token
                # Shape: (num_tokens * experts_per_token)
                dst_ranks = flat_indices // self.num_local_experts

            with TimingContext("dispatch_count_calculation", self.timing_stats):
                # Calculate send_counts using scatter_add for efficiency
                # This avoids Python loops and is much faster on GPU
                ones = torch.ones_like(dst_ranks, dtype=torch.long)
                send_counts = torch.zeros(
                    self.world_size, dtype=torch.long, device=device)
                send_counts.scatter_add_(0, dst_ranks, ones)

            with TimingContext("dispatch_count_exchange", self.timing_stats):
                # Exchange send/recv counts with all other ranks
                recv_counts = torch.empty_like(send_counts)
                dist.all_to_all_single(recv_counts, send_counts)

            with TimingContext("dispatch_data_preparation", self.timing_stats):
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

                # Allocate receive buffers
                total_recv = int(recv_counts.sum().item())
                recv_buf = torch.empty(total_recv, cfg.hidden_dim,
                                       dtype=cfg.in_dtype, device=device)
                recv_meta_buf = torch.empty(
                    total_recv, self.META_DIM, dtype=torch.int32, device=device)

            with TimingContext("dispatch_data_communication", self.timing_stats):
                # Perform all-to-all communication for data
                dist.all_to_all_single(
                    recv_buf,
                    send_buf,
                    output_split_sizes=recv_counts.tolist(),
                    input_split_sizes=send_counts.tolist(),
                )

            with TimingContext("dispatch_meta_communication", self.timing_stats):
                # Perform all-to-all communication for metadata
                dist.all_to_all_single(
                    recv_meta_buf.view(-1),
                    send_meta_buf.view(-1),
                    output_split_sizes=[
                        c * self.META_DIM for c in recv_counts.tolist()],
                    input_split_sizes=[
                        c * self.META_DIM for c in send_counts.tolist()],
                )

        return recv_buf, recv_meta_buf.view(-1, self.META_DIM)

    def combine(self, expert_y: torch.Tensor, meta: torch.Tensor, weights: torch.Tensor):
        device = expert_y.device
        cfg = self.cfg

        with TimingContext("combine_total", self.timing_stats):
            with TimingContext("combine_preparation", self.timing_stats):
                # Determine destination ranks from metadata
                # src_rank is the destination for combine
                dst_ranks = meta[:, 0]

            with TimingContext("combine_count_calculation", self.timing_stats):
                # Calculate send_counts using scatter_add
                ones = torch.ones_like(dst_ranks, dtype=torch.long)
                send_counts = torch.zeros(
                    self.world_size, dtype=torch.long, device=device)
                send_counts.scatter_add_(0, dst_ranks, ones)

            with TimingContext("combine_count_exchange", self.timing_stats):
                # Exchange send/recv counts
                recv_counts = torch.empty_like(send_counts)
                dist.all_to_all_single(recv_counts, send_counts)

            with TimingContext("combine_data_preparation", self.timing_stats):
                # Sort by destination rank for efficient communication
                perm = torch.argsort(dst_ranks)
                send_buf = expert_y[perm]
                send_meta_buf = meta[perm]

                # Allocate receive buffers
                total_recv = int(recv_counts.sum().item())
                recv_buf = torch.empty(total_recv, cfg.hidden_dim,
                                       dtype=cfg.out_dtype, device=device)
                recv_meta_buf = torch.empty(
                    total_recv, self.META_DIM, dtype=torch.int32, device=device)

            with TimingContext("combine_data_communication", self.timing_stats):
                # Perform all-to-all communication
                dist.all_to_all_single(
                    recv_buf,
                    send_buf,
                    output_split_sizes=recv_counts.tolist(),
                    input_split_sizes=send_counts.tolist(),
                )

            with TimingContext("combine_meta_communication", self.timing_stats):
                dist.all_to_all_single(
                    recv_meta_buf.view(-1),
                    send_meta_buf.view(-1),
                    output_split_sizes=[
                        c * self.META_DIM for c in recv_counts.tolist()],
                    input_split_sizes=[
                        c * self.META_DIM for c in send_counts.tolist()],
                )

            with TimingContext("combine_aggregation", self.timing_stats):
                recv_meta_buf = recv_meta_buf.view(-1, self.META_DIM)

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

    def get_timing_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive timing statistics"""
        summary = {}
        for operation, times in self.timing_stats.items():
            if times:
                summary[operation] = {
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
        return summary

    def print_timing_summary(self):
        """Print detailed timing analysis"""
        print(f"\n=== Timing Summary for Rank {self.rank} ===")
        summary = self.get_timing_summary()

        # Group related operations
        dispatch_ops = [k for k in summary.keys() if k.startswith('dispatch')]
        combine_ops = [k for k in summary.keys() if k.startswith('combine')]

        print("\nDISPATCH Operations:")
        for op in sorted(dispatch_ops):
            stats = summary[op]
            print(
                f"  {op:30s}: {stats['mean']:8.3f}ms (min: {stats['min']:6.3f}, max: {stats['max']:6.3f})")

        print("\nCOMBINE Operations:")
        for op in sorted(combine_ops):
            stats = summary[op]
            print(
                f"  {op:30s}: {stats['mean']:8.3f}ms (min: {stats['min']:6.3f}, max: {stats['max']:6.3f})")

        # Calculate communication vs computation breakdown
        dispatch_comm = sum(summary.get(k, {}).get('total', 0)
                            for k in ['dispatch_count_exchange', 'dispatch_data_communication', 'dispatch_meta_communication'])
        dispatch_comp = sum(summary.get(k, {}).get('total', 0)
                            for k in ['dispatch_preparation', 'dispatch_count_calculation', 'dispatch_data_preparation'])

        combine_comm = sum(summary.get(k, {}).get('total', 0)
                           for k in ['combine_count_exchange', 'combine_data_communication', 'combine_meta_communication'])
        combine_comp = sum(summary.get(k, {}).get('total', 0)
                           for k in ['combine_preparation', 'combine_count_calculation', 'combine_data_preparation', 'combine_aggregation'])

        print(f"\nBreakdown Analysis:")
        print(
            f"  Dispatch - Communication: {dispatch_comm:8.3f}ms, Computation: {dispatch_comp:8.3f}ms")
        print(
            f"  Combine  - Communication: {combine_comm:8.3f}ms, Computation: {combine_comp:8.3f}ms")

        total_comm = dispatch_comm + combine_comm
        total_comp = dispatch_comp + combine_comp
        total_time = total_comm + total_comp

        if total_time > 0:
            print(f"  Overall  - Communication: {total_comm:8.3f}ms ({100*total_comm/total_time:.1f}%), "
                  f"Computation: {total_comp:8.3f}ms ({100*total_comp/total_time:.1f}%)")


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    ata = OptimizedAllToAllTimed(cfg, rank, world_size)

    # Dispatch tokens to experts
    dispatched_x, meta = ata.dispatch(rank_data.x, rank_data.indices)

    # Simulate expert computation
    with TimingContext("expert_computation", ata.timing_stats):
        expert_y = dispatched_x.to(cfg.out_dtype) * (1 + rank)

    # Combine results from experts
    y = ata.combine(expert_y, meta, rank_data.weights)

    # Print timing summary (only for rank 0 to avoid cluttering output)
    if rank == 0:
        ata.print_timing_summary()

    return y[: rank_data.num_tokens]
