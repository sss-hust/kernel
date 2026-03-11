# submission.py
"""
Pre-Weighted Vectorized All-to-All implementation with minimal metadata.
预加权向量化 All-to-All 实现，最小化元数据传输。

Key Optimization / 核心优化:
  - Dispatch sends (x * weight) instead of raw x, so combine only needs to SUM
    Dispatch 发送的是 (x * weight) 而非原始 x，因此 Combine 只需直接求和

Metadata Reduction / 元数据精简:
  - Dispatch: (global_exp, src_rank, src_token) - 3 fields (no src_k needed)
    分发阶段：3 个字段，不再需要 src_k（因为权重已乘入数据）
  - Combine: (src_token) only - 1 field (no weight needed)
    合并阶段：仅 1 个字段，无需传输权重
"""
import torch
import torch.distributed as dist
from task import input_t, output_t


class VectorizedAllToAll:
    # Dispatch 阶段元数据: 全局专家ID, 源Rank, 源Token索引 (无需 src_k)
    DISPATCH_META_DIM = 3
    
    # Combine 阶段元数据: 仅源Token索引 (无需 weight，因为已预乘)
    COMBINE_META_DIM = 1

    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor):
        """
        Pre-weighted vectorized dispatch: send (token * weight) to destination ranks.
        预加权向量化分发：发送 (token * weight) 到目标 Rank。
        
        Args:
            x: Input tokens / 输入 Token (N, hidden_dim)
            indices: Expert indices per token / 每个 Token 的专家索引 (N, K)
            weights: Routing weights per token / 每个 Token 的路由权重 (N, K)
        
        Returns:
            recv_buf: Received pre-weighted token data / 接收到的预加权 Token 数据
            recv_meta: Metadata for combine / Combine 阶段需要的元数据 (无需权重)
        """
        device = x.device
        cfg = self.cfg
        N, K = indices.shape
        
        # 1. Flatten routing info / 展平路由信息
        flat_indices = indices.flatten()  # (N*K,)
        dst_ranks = flat_indices // self.num_local_experts  # (N*K,) 计算每个任务要去的目标 Rank
        
        # Token index repeated K times / 重复 K 次的 Token 索引，用于追踪来源
        src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
        # Flatten weights for pre-multiplication / 展平权重用于预乘
        weights_flat = weights.flatten()  # (N*K,)
        
        # 2. Count tokens per destination rank using scatter_add / 使用 scatter_add 统计每个目标 Rank 的数据量
        send_counts = torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), torch.ones(N * K, dtype=torch.long, device=device))
        
        # Exchange counts / 交换计数，告知对方要发多少数据
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        
        # 3. Sort by destination rank for contiguous send buffers / 按目标 Rank 排序，以构建连续的发送缓冲区
        perm = torch.argsort(dst_ranks)
        
        # Reorder and PRE-WEIGHT token data: (x[token] * weight) / 重排并预加权 Token 数据
        # Key optimization: multiply weight BEFORE sending, so combine only needs to sum
        # 核心优化：发送前乘以权重，合并时只需直接求和
        gathered_x = x[src_token_flat[perm].long()]  # (N*K, hidden_dim)
        # Convert weights to same dtype as x to avoid dtype mismatch / 转换权重类型以避免类型不匹配
        gathered_weights = weights_flat[perm].to(gathered_x.dtype).unsqueeze(1)  # (N*K, 1)
        send_buf = (gathered_x * gathered_weights).to(cfg.in_dtype)  # (N*K, hidden_dim) 预加权，保持输入类型
        
        # Build metadata: (global_exp, src_rank, src_token) - no src_k needed / 构建元数据，无需 src_k
        send_meta = torch.stack([
            flat_indices[perm].int(),
            torch.full((N * K,), self.rank, dtype=torch.int32, device=device),  # 源 Rank ID
            src_token_flat[perm]
        ], dim=1)  # (N*K, 3)
        
        # 4. All-to-all for pre-weighted token data / 发送预加权 Token 数据
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
        
        # No need to send weights separately - they're already multiplied into the data
        # 无需单独发送权重 - 已经乘入数据中
        
        return recv_buf, recv_meta

    def combine(self, expert_y: torch.Tensor, meta: torch.Tensor):
        """
        Simplified combine: send pre-weighted results back and directly sum.
        简化的合并：发回预加权结果，直接求和。
        
        Args:
            expert_y: Expert outputs (already pre-weighted) / 专家计算输出（已预加权）(Total, hidden_dim)
            meta: Dispatch metadata / 接收到的元数据 (Total, 3)
        
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
        
        # Build minimal combine metadata: only src_token / 构建最小元数据：仅 src_token
        # No weight needed - it was pre-multiplied during dispatch
        # 无需权重 - 已在 dispatch 阶段预乘
        send_meta = meta[perm, 2].int()  # (Total,) - just src_token
        
        # 4. All-to-all for expert outputs / 发送计算结果
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        
        dist.all_to_all_single(
            recv_buf, send_buf.to(cfg.out_dtype),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        # 5. All-to-all for minimal metadata (just src_token) / 发送最小元数据
        recv_meta = torch.empty(total_recv, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_meta, send_meta,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        # 6. Direct sum using index_add_ (no weighting needed) / 使用 index_add_ 直接求和
        src_token_indices = recv_meta.long()  # (Total_Recv,)
        
        # Initialize output / 初始化输出 Tensor
        output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)
        
        # Direct accumulation - weights were pre-multiplied / 直接累加 - 权重已预乘
        output.index_add_(0, src_token_indices, recv_buf.float())
        
        return output.to(cfg.out_dtype)

def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)

    device = rank_data.x.device
    
    # ========== 创建所有计时事件 / Create all timing events ==========
    events = {}
    event_names = [
        # Dispatch 阶段 / Dispatch phase
        'start',
        'd_flatten',        # 展平路由信息
        'd_count',          # scatter_add 统计
        'd_exchange_count', # all_to_all 交换计数
        'd_sort',           # argsort 排序
        'd_preweight',      # 预加权 (gather + multiply)
        'd_build_meta',     # 构建元数据
        'd_a2a_data',       # all_to_all 发送数据
        'd_a2a_meta',       # all_to_all 发送元数据
        # Compute 阶段 / Compute phase
        'compute',
        # Combine 阶段 / Combine phase
        'c_count',          # scatter_add 统计
        'c_exchange_count', # all_to_all 交换计数
        'c_sort',           # argsort 排序
        'c_reorder',        # 重排数据
        'c_a2a_data',       # all_to_all 发送数据
        'c_a2a_meta',       # all_to_all 发送元数据
        'c_index_add',      # index_add_ 累加
        'end'
    ]
    for name in event_names:
        events[name] = torch.cuda.Event(enable_timing=True)
    
    num_local_experts = cfg.num_experts // world_size
    N, K = rank_data.indices.shape
    x = rank_data.x
    indices = rank_data.indices
    weights = rank_data.weights
    
    # ==================== DISPATCH 阶段 ====================
    events['start'].record()
    
    # 1. Flatten routing info / 展平路由信息
    flat_indices = indices.flatten()
    dst_ranks = flat_indices // num_local_experts
    src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
    weights_flat = weights.flatten()
    events['d_flatten'].record()
    
    # 2. Count tokens per destination / 统计每个目标的数据量
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    send_counts.scatter_add_(0, dst_ranks.long(), torch.ones(N * K, dtype=torch.long, device=device))
    events['d_count'].record()
    
    # 3. Exchange counts / 交换计数
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    events['d_exchange_count'].record()
    
    # 4. Sort by destination / 按目标排序
    perm = torch.argsort(dst_ranks)
    events['d_sort'].record()
    
    # 5. Pre-weight token data / 预加权 Token 数据
    gathered_x = x[src_token_flat[perm].long()]
    gathered_weights = weights_flat[perm].to(gathered_x.dtype).unsqueeze(1)
    send_buf = (gathered_x * gathered_weights).to(cfg.in_dtype)
    events['d_preweight'].record()
    
    # 6. Build metadata / 构建元数据
    send_meta = torch.stack([
        flat_indices[perm].int(),
        torch.full((N * K,), rank, dtype=torch.int32, device=device),
        src_token_flat[perm]
    ], dim=1)
    events['d_build_meta'].record()
    
    # 7. All-to-all for data / 发送数据
    total_recv = int(recv_counts.sum().item())
    recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
    dist.all_to_all_single(
        recv_buf, send_buf,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist()
    )
    events['d_a2a_data'].record()
    
    # 8. All-to-all for metadata / 发送元数据
    DISPATCH_META_DIM = 3
    recv_meta = torch.empty(total_recv, DISPATCH_META_DIM, dtype=torch.int32, device=device)
    dist.all_to_all_single(
        recv_meta.view(-1), send_meta.view(-1),
        output_split_sizes=[c * DISPATCH_META_DIM for c in recv_counts.tolist()],
        input_split_sizes=[c * DISPATCH_META_DIM for c in send_counts.tolist()]
    )
    events['d_a2a_meta'].record()
    
    # ==================== COMPUTE 阶段 ====================
    expert_y = recv_buf.to(cfg.out_dtype) * (1 + rank)
    events['compute'].record()
    
    # ==================== COMBINE 阶段 ====================
    meta = recv_meta
    
    # 1. Count tokens per destination / 统计每个目标的数据量
    c_dst_ranks = meta[:, 1]
    c_send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    c_send_counts.scatter_add_(0, c_dst_ranks.long(), torch.ones(meta.shape[0], dtype=torch.long, device=device))
    events['c_count'].record()
    
    # 2. Exchange counts / 交换计数
    c_recv_counts = torch.empty_like(c_send_counts)
    dist.all_to_all_single(c_recv_counts, c_send_counts)
    events['c_exchange_count'].record()
    
    # 3. Sort by destination / 按目标排序
    c_perm = torch.argsort(c_dst_ranks)
    events['c_sort'].record()
    
    # 4. Reorder data and build metadata / 重排数据和构建元数据
    c_send_buf = expert_y[c_perm]
    c_send_meta = meta[c_perm, 2].int()
    events['c_reorder'].record()
    
    # 5. All-to-all for data / 发送数据
    c_total_recv = int(c_recv_counts.sum().item())
    c_recv_buf = torch.empty(c_total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    dist.all_to_all_single(
        c_recv_buf, c_send_buf.to(cfg.out_dtype),
        output_split_sizes=c_recv_counts.tolist(),
        input_split_sizes=c_send_counts.tolist()
    )
    events['c_a2a_data'].record()
    
    # 6. All-to-all for metadata / 发送元数据
    c_recv_meta = torch.empty(c_total_recv, dtype=torch.int32, device=device)
    dist.all_to_all_single(
        c_recv_meta, c_send_meta,
        output_split_sizes=c_recv_counts.tolist(),
        input_split_sizes=c_send_counts.tolist()
    )
    events['c_a2a_meta'].record()
    
    # 7. index_add_ accumulation / 累加
    output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)
    output.index_add_(0, c_recv_meta.long(), c_recv_buf.float())
    y = output.to(cfg.out_dtype)
    events['c_index_add'].record()
    
    events['end'].record()
    
    # ==================== 输出计时结果 ====================
    events['end'].synchronize()
    
    if rank == 0:
        def t(start, end):
            return events[start].elapsed_time(events[end])
        
        # Dispatch 详细计时
        d_flatten = t('start', 'd_flatten')
        d_count = t('d_flatten', 'd_count')
        d_exchange = t('d_count', 'd_exchange_count')
        d_sort = t('d_exchange_count', 'd_sort')
        d_preweight = t('d_sort', 'd_preweight')
        d_build_meta = t('d_preweight', 'd_build_meta')
        d_a2a_data = t('d_build_meta', 'd_a2a_data')
        d_a2a_meta = t('d_a2a_data', 'd_a2a_meta')
        dispatch_total = t('start', 'd_a2a_meta')
        
        # Compute 计时
        compute_time = t('d_a2a_meta', 'compute')
        
        # Combine 详细计时
        c_count = t('compute', 'c_count')
        c_exchange = t('c_count', 'c_exchange_count')
        c_sort = t('c_exchange_count', 'c_sort')
        c_reorder = t('c_sort', 'c_reorder')
        c_a2a_data = t('c_reorder', 'c_a2a_data')
        c_a2a_meta = t('c_a2a_data', 'c_a2a_meta')
        c_index_add = t('c_a2a_meta', 'c_index_add')
        combine_total = t('compute', 'c_index_add')
        
        total_time = t('start', 'end')
        
        print(f"\n{'='*70}")
        print(f"[Rank 0] Total: {total_time:.3f}ms")
        print(f"{'='*70}")
        print(f"  DISPATCH ({dispatch_total:.3f}ms):")
        print(f"    flatten:      {d_flatten:.3f}ms")
        print(f"    count:        {d_count:.3f}ms")
        print(f"    exchange_cnt: {d_exchange:.3f}ms  [all2all]")
        print(f"    sort:         {d_sort:.3f}ms")
        print(f"    preweight:    {d_preweight:.3f}ms")
        print(f"    build_meta:   {d_build_meta:.3f}ms")
        print(f"    a2a_data:     {d_a2a_data:.3f}ms  [all2all]")
        print(f"    a2a_meta:     {d_a2a_meta:.3f}ms  [all2all]")
        print(f"  COMPUTE: {compute_time:.3f}ms")
        print(f"  COMBINE ({combine_total:.3f}ms):")
        print(f"    count:        {c_count:.3f}ms")
        print(f"    exchange_cnt: {c_exchange:.3f}ms  [all2all]")
        print(f"    sort:         {c_sort:.3f}ms")
        print(f"    reorder:      {c_reorder:.3f}ms")
        print(f"    a2a_data:     {c_a2a_data:.3f}ms  [all2all]")
        print(f"    a2a_meta:     {c_a2a_meta:.3f}ms  [all2all]")
        print(f"    index_add:    {c_index_add:.3f}ms")
        print(f"{'='*70}")

    return y[: rank_data.num_tokens]

