# submission.py
"""
Vectorized All-to-All implementation with optimized metadata.
向量化 All-to-All 实现与优化的元数据结构。

Optimizations / 优化点:
1. Dispatch: Uses tensor operations (scatter_add_, argsort) instead of Python loops
   使用张量操作替代 Python 循环，显著降低 CPU 开销。
2. Combine: Uses index_add_ for vectorized weighted sum
   使用 index_add_ 进行向量化的加权求和。
3. Metadata: Combine phase only transmits (src_token, weight) - weight pre-computed on sender
   Combine 阶段仅回传 (src_token, weight)，其中 weight 在发送方预先计算好，减少通信量。
"""
import torch
import torch.distributed as dist
from task import input_t, output_t


class VectorizedAllToAll:
    # Dispatch 阶段元数据: 全局专家ID, 源Rank, 源Token索引, 源Top-k索引
    DISPATCH_META_DIM = 4
    
    # Combine 阶段元数据: 源Token索引, 权重 (float32 视为 int32 传输)
    COMBINE_META_DIM = 2 

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor):
        """
        Vectorized dispatch: send tokens to destination ranks based on expert routing.
        向量化分发：根据专家路由将 Token 发送到目标 Rank。
        
        Args:
            x: Input tokens / 输入 Token (N, hidden_dim)
            indices: Expert indices per token / 每个 Token 的专家索引 (N, K)
            weights: Routing weights per token / 每个 Token 的路由权重 (N, K)
        
        Returns:
            recv_buf: Received token data / 接收到的 Token 数据
            recv_meta: Metadata for combine / Combine 阶段需要的元数据
            recv_weights: Pre-fetched weights / 预取的权重 (用于回传)
        """
        device = x.device
        cfg = self.cfg
        N, K = indices.shape
        
        # 1. Flatten routing info / 展平路由信息
        flat_indices = indices.flatten()  # (N*K,)
        dst_ranks = flat_indices // self.num_local_experts  # (N*K,) 计算每个任务要去的目标 Rank
        
        # Token index repeated K times / 重复 K 次的 Token 索引，用于追踪来源
        src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
        # Expert slot index (0 to K-1) / 专家槽位索引 (第几个选中的专家)
        src_k_flat = torch.arange(K, device=device, dtype=torch.int32).repeat(N)
        # Pre-fetch weights for each (token, k) pair / 预取权重，避免在接收端查表
        weights_flat = weights.flatten()  # (N*K,)
        
        # 2. Count tokens per destination rank using scatter_add / 使用 scatter_add 统计每个目标 Rank 的数据量
        send_counts = torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), torch.ones(N * K, dtype=torch.long, device=device))
        
        # Exchange counts / 交换计数，告知对方要发多少数据
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        
        # 3. Sort by destination rank for contiguous send buffers / 按目标 Rank 排序，以构建连续的发送缓冲区
        # argsort 返回排序后的索引，能够把去往同一个 Rank 的数据聚在一起
        perm = torch.argsort(dst_ranks)
        
        # Reorder token data: gather x by src_token_flat[perm] / 重排 Token 数据
        # 这里实际上实现了: [Token A->Rank 0, Token B->Rank 0, Token C->Rank 1 ...] 的物理布局
        send_buf = x[src_token_flat[perm].long()]  # (N*K, hidden_dim)
        
        # Build metadata: (global_exp, src_rank, src_token, src_k) / 构建元数据
        send_meta = torch.stack([
            flat_indices[perm].int(),
            torch.full((N * K,), self.rank, dtype=torch.int32, device=device), # 源 Rank ID
            src_token_flat[perm],
            src_k_flat[perm]
        ], dim=1)  # (N*K, 4)
        
        # Pre-fetched weights (to be returned to sender during combine) / 预取的权重
        send_weights = weights_flat[perm]  # (N*K,)
        
        # 4. All-to-all for token data / 发送 Token 数据
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        
        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        # 5. All-to-all for metadata / 发送元数据
        recv_meta = torch.empty(total_recv, self.DISPATCH_META_DIM, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_meta.view(-1), send_meta.view(-1),
            output_split_sizes=[c * self.DISPATCH_META_DIM for c in recv_counts.tolist()],
            input_split_sizes=[c * self.DISPATCH_META_DIM for c in send_counts.tolist()]
        )
        
        # 6. All-to-all for pre-fetched weights / 发送预取权重
        recv_weights = torch.empty(total_recv, dtype=torch.float32, device=device)
        dist.all_to_all_single(
            recv_weights, send_weights.float(),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        return recv_buf, recv_meta, recv_weights

    def combine(self, expert_y: torch.Tensor, meta: torch.Tensor, weights: torch.Tensor):
        """
        Vectorized combine: send computed results back to source ranks.
        向量化合并：将计算结果发回源 Rank。
        
        Args:
            expert_y: Expert outputs / 专家计算输出 (Total, hidden_dim)
            meta: Dispatch metadata / 接收到的元数据 (Total, 4)
            weights: Pre-fetched weights / 接收到的预取权重 (Total,)
        
        Returns:
            output: Combined output tensor / 合并后的输出 (max_num_tokens, hidden_dim)
        """
        device = expert_y.device
        cfg = self.cfg
        
        # 1. Destination is the original source rank / 目标是原始的源 Rank
        dst_ranks = meta[:, 1]  # src_rank column
        
        # 2. Count tokens per destination using scatter_add / 统计每个目标 Rank 的数据量
        send_counts = torch.zeros(self.world_size, dtype=torch.long, device=device)
        ones = torch.ones(meta.shape[0], dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), ones)
        
        # Exchange counts / 交换计数
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        
        # 3. Sort by destination rank / 按目标 Rank 排序
        perm = torch.argsort(dst_ranks)
        
        # Reorder expert output / 重排专家输出
        send_buf = expert_y[perm]  # (Total, hidden_dim)
        
        # Build compact combine metadata: (src_token, weight) / 构建紧凑的 Combine 元数据
        # Optimization: Only send `src_token` for positioning and `weight` for calculation.
        # 优化：只发送用于定位的 `src_token` 和用于计算的 `weight`。
        
        src_token_sorted = meta[perm, 2]  # (Total,)
        weights_sorted = weights[perm]    # (Total,)
        
        # Pack into combined buffer / 打包
        # float32 weight is viewed as int32 to stack with src_token
        # float32 的权重被视为 int32 以便与 src_token 堆叠传输
        send_meta = torch.stack([
            src_token_sorted.int(),
            weights_sorted.view(torch.int32)
        ], dim=1)  # (Total, 2)
        
        # 4. All-to-all for expert outputs / 发送计算结果
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        
        dist.all_to_all_single(
            recv_buf, send_buf.to(cfg.out_dtype),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        # 5. All-to-all for compact metadata / 发送元数据
        recv_meta = torch.empty(total_recv, self.COMBINE_META_DIM, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_meta.view(-1), send_meta.view(-1),
            output_split_sizes=[c * self.COMBINE_META_DIM for c in recv_counts.tolist()],
            input_split_sizes=[c * self.COMBINE_META_DIM for c in send_counts.tolist()]
        )
        
        # 6. Vectorized weighted sum using index_add_ / 使用 index_add_ 进行向量化加权求和
        src_token_indices = recv_meta[:, 0].long()  # (Total_Recv,)
        recv_weights = recv_meta[:, 1].view(torch.float32)  # reinterpret int32 as float32 / 重新转回 float32
        
        # Initialize output / 初始化输出 Tensor
        output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)
        
        # Weighted buffer / 计算加权后的值
        weighted_buf = recv_buf.float() * recv_weights.unsqueeze(1)
        
        # Vectorized accumulation / 向量化累加
        # out[src_token_indices[i]] += weighted_buf[i]
        output.index_add_(0, src_token_indices, weighted_buf)
        
        return output.to(cfg.out_dtype)


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    ata = VectorizedAllToAll(cfg, rank, world_size)

    # Create CUDA events for timing / 创建 CUDA 事件用于计时
    start_event = torch.cuda.Event(enable_timing=True)
    dispatch_end_event = torch.cuda.Event(enable_timing=True)
    compute_end_event = torch.cuda.Event(enable_timing=True)
    combine_end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # Dispatch tokens to experts (pass weights for pre-fetching) / 分发 Token (传入权重以预取)
    recv_buf, recv_meta, recv_weights = ata.dispatch(
        rank_data.x, rank_data.indices, rank_data.weights
    )
    dispatch_end_event.record()
    
    # Simulate expert computation / 模拟专家计算
    expert_y = recv_buf.to(cfg.out_dtype) * (1 + rank)
    compute_end_event.record()
    
    # Combine results (use pre-fetched weights) / 合并结果 (使用预取的权重)
    y = ata.combine(expert_y, recv_meta, recv_weights)
    combine_end_event.record()

    # Wait for completion / 等待完成
    combine_end_event.synchronize()

    if rank == 0:
        dispatch_time = start_event.elapsed_time(dispatch_end_event)
        compute_time = dispatch_end_event.elapsed_time(compute_end_event)
        combine_time = compute_end_event.elapsed_time(combine_end_event)
        total_time = start_event.elapsed_time(combine_end_event)
        print(f"\n[Rank 0 Timing] Total: {total_time:.3f}ms | "
              f"Dispatch: {dispatch_time:.3f}ms | "
              f"Compute: {compute_time:.3f}ms | "
              f"Combine: {combine_time:.3f}ms")

    return y[: rank_data.num_tokens]
