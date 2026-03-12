# submission.py
"""
Pre-Weighted Vectorized All-to-All implementation with minimal metadata.
预加权向量化 All-to-All 实现，极简元数据传输与隐式路由还原。

Key Optimization / 核心优化:
  1. Dispatch Sends Pre-weighted Data: (x * weight) is sent, so combine only needs to SUM.
     Dispatch 发送的是 (x * weight)，因此 Combine 只需直接求和，无需再次乘权重。
  2. Implicit Routing Restoration (隐式路由还原):
     Because all_to_all_single processes chunks in rank order, the returned results naturally 
     align with the originally sent chunks. We DO NOT need to transmit `src_rank` or 
     `src_token` over the network. The original rank uses its local `perm` array to 
     directly `index_add_` the results back. Combine metadata is 0 bytes!
     由于 all_to_all_single 按 Rank 顺序处理数据块，返回的结果天然匹配原发送块的布局。
     我们完全不需要在网络中传输 src_rank 和 src_token。原节点可以直接靠本地的 perm
     记录进行 index_add_ 累加！Combine 的元数据通信为 0。
"""
import torch
import torch.distributed as dist
from task import input_t, output_t


class VectorizedAllToAll:
    # Dispatch 阶段只需传递: 全局专家ID
    DISPATCH_META_DIM = 1
    
    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor):
        device = x.device
        cfg = self.cfg
        N, K = indices.shape
        
        flat_indices = indices.flatten()
        dst_ranks = flat_indices // self.num_local_experts
        src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
        weights_flat = weights.flatten()
        
        send_counts = torch.zeros(self.world_size, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), torch.ones(N * K, dtype=torch.long, device=device))
        
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        
        perm = torch.argsort(dst_ranks)
        
        gathered_x = x[src_token_flat[perm].long()]
        gathered_weights = weights_flat[perm].to(gathered_x.dtype).unsqueeze(1)
        send_buf = (gathered_x * gathered_weights).to(cfg.in_dtype)
        
        send_meta = flat_indices[perm].int()
        
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        
        dist.all_to_all_single(
            recv_buf, send_buf,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        recv_meta = torch.empty(total_recv, dtype=torch.int32, device=device)
        dist.all_to_all_single(
            recv_meta, send_meta,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
        
        # 暂存供 Combine 阶段使用的本地状态（省去了网络 round-trip）
        self.send_counts = send_counts
        self.recv_counts = recv_counts
        self.perm = perm
        self.src_token_flat = src_token_flat
        self.total_send = N * K
        
        return recv_buf, recv_meta

    def combine(self, expert_y: torch.Tensor):
        """
        No metadata needed! Results arrive in the exact order they were sent.
        无需任何元数据！结果到达的顺序与当初发送出去的块顺序严丝合缝。
        """
        device = expert_y.device
        cfg = self.cfg
        
        # 反向 all-to-all：输出大小等于发送量，输入大小等于接收量
        recv_buf = torch.empty(self.total_send, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        
        dist.all_to_all_single(
            recv_buf, expert_y.to(cfg.out_dtype),
            output_split_sizes=self.send_counts.tolist(),
            input_split_sizes=self.recv_counts.tolist()
        )
        
        output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)
        
        # 靠着之前的排列序号，原封不动地 index_add_ 回去
        output.index_add_(0, self.src_token_flat[self.perm].long(), recv_buf.float())
        
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
        'c_count',          # (已废弃，保留计时凑数)
        'c_exchange_count', # (已废弃，保留计时凑数)
        'c_sort',           # (已废弃，保留计时凑数)
        'c_reorder',        # (已废弃，保留计时凑数)
        'c_a2a_data',       # all_to_all 发送数据 (反向)
        'c_a2a_meta',       # (已废弃，保留计时凑数)
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
    
    # 1. Flatten routing info
    flat_indices = indices.flatten()
    dst_ranks = flat_indices // num_local_experts
    src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
    weights_flat = weights.flatten()
    events['d_flatten'].record()
    
    # 2. Count tokens per destination
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    send_counts.scatter_add_(0, dst_ranks.long(), torch.ones(N * K, dtype=torch.long, device=device))
    events['d_count'].record()
    
    # 3. Exchange counts
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts)
    events['d_exchange_count'].record()
    
    # 4. Sort by destination
    perm = torch.argsort(dst_ranks)
    events['d_sort'].record()
    
    # 5. Pre-weight token data
    gathered_x = x[src_token_flat[perm].long()]
    gathered_weights = weights_flat[perm].to(gathered_x.dtype).unsqueeze(1)
    send_buf = (gathered_x * gathered_weights).to(cfg.in_dtype)
    events['d_preweight'].record()
    
    # 6. Build metadata: ONLY expert idx!
    send_meta = flat_indices[perm].int()
    events['d_build_meta'].record()
    
    # 7. All-to-all for data
    total_recv = int(recv_counts.sum().item())
    recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
    dist.all_to_all_single(
        recv_buf, send_buf,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist()
    )
    events['d_a2a_data'].record()
    
    # 8. All-to-all for metadata
    recv_meta = torch.empty(total_recv, dtype=torch.int32, device=device)
    dist.all_to_all_single(
        recv_meta, send_meta,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist()
    )
    events['d_a2a_meta'].record()
    
    # ==================== COMPUTE 阶段 ====================
    expert_y = recv_buf.to(cfg.out_dtype) * (1 + rank)
    events['compute'].record()
    
    # ==================== COMBINE 阶段 ====================
    # 物理超度：全部省去，不耗时！
    events['c_count'].record()
    events['c_exchange_count'].record()
    events['c_sort'].record()
    events['c_reorder'].record()
    
    # 5. 反向 all-to-all，原路送回数据
    c_recv_buf = torch.empty(N * K, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
    dist.all_to_all_single(
        c_recv_buf, expert_y,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist()
    )
    events['c_a2a_data'].record()
    
    # 没有元数据的发送
    events['c_a2a_meta'].record()
    
    # 7. 本地索引累加
    output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, dtype=torch.float32, device=device)
    output.index_add_(0, src_token_flat[perm].long(), c_recv_buf.float())
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
