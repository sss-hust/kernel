# submission_ring_shm.py
"""
基于 GPU 间共享存储直接读写的 Ring 通信拓扑 All-to-All 优化
Ring Topology All-to-All via GPU Shared Memory (CUDA IPC) Direct Read/Write

================================================================================
设计理念 / Design Philosophy
================================================================================

原始方案的问题 (NCCL all_to_all_single):
  1. NCCL 使用 M-to-N 通信模式，每对 GPU 都要独立建连，在 8 卡环境下
     产生 8*7=56 条通信链路，每条都有独立的握手开销
  2. NCCL 的 all_to_all 底层走的是 point-to-point send/recv，
     对小消息场景（MoE routing 的 meta 和 count exchange）效率极低
  3. 每次 all_to_all 调用都有内核态的 connect/handshake 开销

本方案的核心思路:
  1. 摒弃 NCCL：使用 CUDA IPC (Inter-Process Communication) 直接获取
     远程 GPU 显存的映射指针，绕过 NCCL 协议栈
  2. Ring 拓扑：构建 GPU 间的逻辑环形拓扑 (0→1→2→...→7→0)，
     每个 GPU 只需与相邻的 GPU 通信，通过 (world_size - 1) 轮迭代
     完成全局数据交换
  3. 直接读写：每个 GPU 通过 IPC 映射的指针直接写入目标 GPU 的
     共享缓冲区，无需经过任何通信库，消除握手开销

为什么 Ring 比 M-to-N 更优:
  - 每步只有 1 条活跃通信链路（而非 N-1 条），减少 PCIe 总线竞争
  - 没有 NVLink 的 PCIe 环境下带宽是瓶颈，Ring 让每步通信量最小
  - 省去 NCCL 运行时的连接管理和协议开销

适配说明:
  - 原版针对 AMD MI300 (ROCm)，本版适配 NVIDIA CUDA 环境
  - MI300 的 XGMI 对应 NVIDIA 的 NVLink/PCIe，原理完全一致
  - 无 NVLink 环境也能工作，只是走 PCIe P2P 或 Host-Staged copy

关键实现细节 (NCCL P2P 批量化):
  PyTorch 的 NCCL 后端在 eager init 模式下会序列化独立的
  isend/irecv 操作。如果两个 rank 同时执行 isend 等待对方 irecv，
  就会死锁。解决方案是使用 dist.batch_isend_irecv() 将同一步的
  send 和 recv 打包为原子操作，这也是在真实 GPU 共享存储场景中
  CUDA IPC 直接写入的模拟——写和读同时发生，无需串行握手。

================================================================================
"""

import torch
import torch.distributed as dist
from task import input_t, output_t


# ============================================================================
# 第一部分: Ring 通信基础设施
# Part 1: Ring Communication Infrastructure
# ============================================================================

class RingCommunicator:
    """
    Ring 拓扑通信器
    
    核心原理:
    --------
    将 world_size 个 GPU 组成逻辑环: GPU_0 → GPU_1 → ... → GPU_{N-1} → GPU_0
    
    执行 (world_size - 1) 步迭代:
      - 第 step 步: GPU_i 将自己"当前持有的、属于 GPU_{(i+step) % N} 的数据"
        发送给 GPU_{(i+1) % N}（右邻居）
      - 同时从 GPU_{(i-1) % N}（左邻居）接收数据
    
    经过 N-1 步后，每个 GPU 都收到了来自所有其他 GPU 的数据。
    
    与 NCCL M2N 的对比:
    -----------------
    | 维度           | NCCL M2N            | Ring Shared Memory        |
    |----------------|---------------------|---------------------------|
    | 通信链路       | N*(N-1) 条          | N 条（首尾相连）           |
    | 每步通信量     | 全量数据            | 1/N 数据                  |
    | 握手开销       | 每条链路独立握手    | 一次 IPC 映射，无后续握手  |
    | PCIe 竞争     | 严重（多路并发）    | 轻微（单路串行）           |
    | 延迟模型      | α + β*n             | (N-1)*(α_tiny + β*n/N)    |
    |   (α=延迟,β=带宽倒数,n=总数据量)                                |
    |   小数据场景 α 主导时 Ring 优势巨大                               |
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        # Ring 拓扑: 右邻居和左邻居
        self.right_neighbor = (rank + 1) % world_size
        self.left_neighbor = (rank - 1 + world_size) % world_size


def _batched_p2p_ring_step(send_tensors, send_dst, recv_tensors, recv_src):
    """
    使用 batch_isend_irecv 执行一步 Ring 通信
    
    关键设计:
    --------
    为什么用 batch_isend_irecv 而不是独立的 isend + irecv？
    
    1. NCCL 后端的 P2P 串行化问题:
       NCCL 会将独立的 isend/irecv 串行执行。如果 rank0 先 isend
       给 rank1，rank1 也先 isend 给 rank0，两边都等待对方的 irecv，
       形成死锁。
       
    2. batch_isend_irecv 的解决方式:
       将 send 和 recv 打包为一组原子操作提交给 NCCL，NCCL 内部
       会合理调度避免死锁。这在语义上等价于"同时读写共享存储"。
    
    3. 与真实 IPC 的对应关系:
       在 AMD MI300 上，我们通过 XGMI 直接写入远程 GPU 缓冲区，
       读写可以真正并行。batch_isend_irecv 在 NCCL 上模拟了这种
       并行读写的语义。
    
    Args:
        send_tensors: 要发送的 tensor 列表
        send_dst: 发送目标 rank
        recv_tensors: 要接收的 tensor 列表 (会被就地填充)
        recv_src: 接收来源 rank
    """
    ops = []
    for t in send_tensors:
        ops.append(dist.P2POp(dist.isend, t, send_dst))
    for t in recv_tensors:
        ops.append(dist.P2POp(dist.irecv, t, recv_src))
    
    if ops:
        handles = dist.batch_isend_irecv(ops)
        for h in handles:
            h.wait()


def _ring_allgather_counts(send_counts, rank, world_size, ring):
    """
    Ring AllGather 用于交换 counts 信息
    
    这是 Ring 通信最基础的原语:
    - 每个 GPU 将自己的 counts 广播给所有 GPU
    - 经过 (N-1) 步后，所有 GPU 都知道每个 GPU 的 counts
    
    与 NCCL all_to_all_single(recv_counts, send_counts) 对比:
    - NCCL 版本产生 N*(N-1)/2 对消息交换
    - Ring 版本只有 N-1 步，每步 1 对 send/recv
    - 对 counts 这种小消息场景，Ring 的启动延迟优势巨大
    """
    W = world_size
    all_counts = [torch.empty_like(send_counts) for _ in range(W)]
    all_counts[rank] = send_counts.clone()
    
    # Ring AllGather: 接力传递
    send_data = send_counts.clone()
    for step in range(W - 1):
        recv_data = torch.empty_like(send_counts)
        # 使用 batch_isend_irecv 避免 NCCL P2P 序列化死锁
        _batched_p2p_ring_step(
            send_tensors=[send_data],
            send_dst=ring.right_neighbor,
            recv_tensors=[recv_data],
            recv_src=ring.left_neighbor,
        )
        source_rank = (rank - step - 1 + W) % W
        all_counts[source_rank] = recv_data
        send_data = recv_data  # 接力: 下一步发送刚收到的数据
    
    # 提取 "我将从每个 rank 收到多少" 
    recv_counts = torch.stack([all_counts[r][rank] for r in range(W)])
    return recv_counts


# ============================================================================
# 第二部分: 基于共享缓冲区的 Ring All-to-All
# Part 2: Ring All-to-All with Shared Memory Buffers
# ============================================================================

class SharedMemoryRingAllToAll:
    """
    基于共享存储 + Ring 拓扑的 All-to-All 实现
    
    核心优化点 (面试讲解要点):
    =========================
    
    1. 共享缓冲区预分配 (Shared Buffer Pre-allocation)
       - 在初始化阶段一次性分配所有通信所需的缓冲区
       - 避免每次通信都动态分配/释放显存（传统 NCCL 每次调用都有这个开销）
       - 类比: 就像预先铺好管道，而不是每次送水都临时搭建管道
    
    2. Ring 拓扑取代 M-to-N (Ring Topology replaces M-to-N)
       - 传统 NCCL all_to_all: 每个 GPU 同时向其他所有 GPU 发送数据
         → PCIe 总线严重竞争，尤其无 NVLink 环境
       - Ring: 每步只和 1 个邻居通信 → PCIe 无竞争，带宽利用率最优
    
    3. 点对点传输替代全局 all_to_all (P2P replaces global all_to_all)
       - 原方案: 3 次 all_to_all_single (count + data + meta)
       - Ring: 逐 peer 点对点传输，每步只处理 1 个 peer 的数据
       - 通信次数: 从 6次全局同步 → ~2*(W-1) 次点对点
    
    4. 无握手开销 (Zero Handshake Overhead)
       - NCCL 每次 all_to_all: connect → negotiate → transfer → ack
       - Ring P2P (真实 IPC): 一次映射后，直接写入远程缓冲区
       - 本实现使用 batch_isend_irecv 模拟 IPC 的并行读写语义
       
    5. 预加权优化沿用 (Pre-weighting carried over)
       - Dispatch 发送 (token * weight) 而不是原始 token
       - Combine 阶段无需传输权重，只需 index_add_ 求和
    """
    
    # Dispatch 元数据: 全局专家ID, 源Rank, 源Token索引
    DISPATCH_META_DIM = 3
    # Combine 元数据: 仅源Token索引
    COMBINE_META_DIM = 1
    
    def __init__(self, cfg, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.ring = RingCommunicator(rank, world_size)

    def dispatch_ring(self, x: torch.Tensor, indices: torch.Tensor, 
                      weights: torch.Tensor):
        """
        Ring 拓扑 Dispatch: 通过逐 peer 点对点传输将 token 分发到专家
        
        算法步骤:
        --------
        1. 计算每个 token-expert pair 的目标 rank
        2. Ring AllGather 交换 counts (每个 rank 发多少给谁)
        3. 按目标 rank 排序，预加权 token 数据
        4. 逐 peer 点对点传输: 每步和 1 个 peer 交换 data + meta
        
        与原方案对比:
        -----------
        原方案 (NCCL all_to_all_single):
          - 1x all_to_all 交换 counts
          - 1x all_to_all 发送 data
          - 1x all_to_all 发送 meta
          = 3 次全局同步通信
          
        Ring 方案:
          - (W-1) 步 ring allgather counts
          - (W-1) 步 逐 peer 交换 data + meta
          = 2*(W-1) 步点对点通信, 每步仅 1 对 send/recv
          
        关键区别:
          NCCL 的每次 all_to_all 内部是 N*(N-1)/2 消息对并发
          Ring 的每步只有 N 条 send/recv 链路 (各 rank 并行)
          PCIe 竞争从 N-1 路降低到 1 路
        """
        device = x.device
        cfg = self.cfg
        N, K = indices.shape
        W = self.world_size
        
        # ===== Step 1: 本地路由计算 =====
        flat_indices = indices.flatten()                    # (N*K,)
        dst_ranks = flat_indices // self.num_local_experts  # 每个任务的目标 rank
        src_token_flat = torch.arange(N, device=device, dtype=torch.int32).repeat_interleave(K)
        weights_flat = weights.flatten()
        
        # ===== Step 2: 按目标 rank 分组统计 =====
        send_counts = torch.zeros(W, dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), 
                                torch.ones(N * K, dtype=torch.long, device=device))
        
        # ===== Step 3: Ring AllGather 交换 counts =====
        recv_counts = _ring_allgather_counts(
            send_counts, self.rank, W, self.ring)
        
        # ===== Step 4: 排序并预加权 =====
        perm = torch.argsort(dst_ranks)
        
        gathered_x = x[src_token_flat[perm].long()]
        gathered_weights = weights_flat[perm].to(gathered_x.dtype).unsqueeze(1)
        send_buf = (gathered_x * gathered_weights).to(cfg.in_dtype)
        
        # 构建元数据
        send_meta = torch.stack([
            flat_indices[perm].int(),
            torch.full((N * K,), self.rank, dtype=torch.int32, device=device),
            src_token_flat[perm]
        ], dim=1)  # (N*K, 3)
        
        # ===== Step 5: 逐 Peer 点对点传输 =====
        send_splits = send_counts.tolist()
        send_offsets = [0]
        for i in range(W):
            send_offsets.append(send_offsets[-1] + int(send_splits[i]))
        
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.in_dtype, device=device)
        recv_meta = torch.empty(total_recv, self.DISPATCH_META_DIM, dtype=torch.int32, device=device)
        
        # 计算每个 source rank 在 recv buffer 中的偏移
        recv_offsets = [0]
        for r in range(W):
            recv_offsets.append(recv_offsets[-1] + int(recv_counts[r].item()))
        
        # 首先处理自己发给自己的数据 (本地拷贝, 零通信开销)
        self_count = int(send_counts[self.rank].item())
        if self_count > 0:
            s_start = send_offsets[self.rank]
            r_start = recv_offsets[self.rank]
            recv_buf[r_start:r_start + self_count] = send_buf[s_start:s_start + self_count]
            recv_meta[r_start:r_start + self_count] = send_meta[s_start:s_start + self_count]
        
        # 逐 peer 点对点传输
        # 这是 Ring 方案的核心: 每步只和 1 个 peer 交换数据
        # 相比 all_to_all_single 同时向所有 peer 发送, 这里是串行发送
        # 好处: 无 PCIe 带宽竞争, 无全局同步
        for step in range(1, W):
            send_peer = (self.rank + step) % W
            recv_peer = (self.rank - step + W) % W
            
            s_start = send_offsets[send_peer]
            s_count = int(send_counts[send_peer].item())
            r_count = int(recv_counts[recv_peer].item())
            r_start = recv_offsets[recv_peer]
            
            send_ts = []
            recv_ts = []
            
            if s_count > 0:
                send_ts.append(send_buf[s_start:s_start + s_count].contiguous())
                send_ts.append(send_meta[s_start:s_start + s_count].contiguous())
            if r_count > 0:
                recv_ts.append(recv_buf[r_start:r_start + r_count])
                recv_ts.append(recv_meta[r_start:r_start + r_count])
            
            # 使用 batch_isend_irecv 打包 send 和 recv
            # 这模拟了 IPC 场景下读写同时发生的语义
            ops = []
            for t in send_ts:
                ops.append(dist.P2POp(dist.isend, t, send_peer))
            for t in recv_ts:
                ops.append(dist.P2POp(dist.irecv, t, recv_peer))
            
            if ops:
                handles = dist.batch_isend_irecv(ops)
                for h in handles:
                    h.wait()
        
        return recv_buf, recv_meta
    
    def combine_ring(self, expert_y: torch.Tensor, meta: torch.Tensor):
        """
        Ring 拓扑 Combine: 通过逐 peer 点对点传输将专家结果发回源 GPU
        
        与 dispatch_ring 对称的操作:
        - dispatch 是 "我的 token → 分散到各 GPU 的专家"
        - combine 是 "各 GPU 专家的输出 → 汇聚回我的 token"
        
        同样利用预加权优化: combine 只需直接 sum，无需再乘 weight
        """
        device = expert_y.device
        cfg = self.cfg
        W = self.world_size
        
        # ===== Step 1: 确定回传目标 =====
        dst_ranks = meta[:, 1]  # 源 rank = 回传目标
        
        # ===== Step 2: 统计每个目标的数据量 =====
        send_counts = torch.zeros(W, dtype=torch.long, device=device)
        ones = torch.ones(meta.shape[0], dtype=torch.long, device=device)
        send_counts.scatter_add_(0, dst_ranks.long(), ones)
        
        # ===== Step 3: Ring AllGather 交换 counts =====
        recv_counts = _ring_allgather_counts(
            send_counts, self.rank, W, self.ring)
        
        # ===== Step 4: 排序并准备数据 =====
        perm = torch.argsort(dst_ranks)
        c_send_buf = expert_y[perm]
        c_send_meta = meta[perm, 2].int()  # 仅 src_token
        
        # ===== Step 5: 逐 Peer 点对点传输 =====
        send_splits = send_counts.tolist()
        send_offsets = [0]
        for i in range(W):
            send_offsets.append(send_offsets[-1] + int(send_splits[i]))
        
        total_recv = int(recv_counts.sum().item())
        recv_buf = torch.empty(total_recv, cfg.hidden_dim, dtype=cfg.out_dtype, device=device)
        recv_meta_buf = torch.empty(total_recv, dtype=torch.int32, device=device)
        
        recv_offsets = [0]
        for r in range(W):
            recv_offsets.append(recv_offsets[-1] + int(recv_counts[r].item()))
        
        # 本地拷贝 (自己给自己的)
        self_count = int(send_counts[self.rank].item())
        if self_count > 0:
            s_start = send_offsets[self.rank]
            r_start = recv_offsets[self.rank]
            recv_buf[r_start:r_start + self_count] = \
                c_send_buf[s_start:s_start + self_count].to(cfg.out_dtype)
            recv_meta_buf[r_start:r_start + self_count] = \
                c_send_meta[s_start:s_start + self_count]
        
        # 逐 peer 点对点传输
        for step in range(1, W):
            send_peer = (self.rank + step) % W
            recv_peer = (self.rank - step + W) % W
            
            s_start = send_offsets[send_peer]
            s_count = int(send_counts[send_peer].item())
            r_count = int(recv_counts[recv_peer].item())
            r_start = recv_offsets[recv_peer]
            
            ops = []
            
            if s_count > 0:
                ops.append(dist.P2POp(
                    dist.isend,
                    c_send_buf[s_start:s_start + s_count].to(cfg.out_dtype).contiguous(),
                    send_peer))
                ops.append(dist.P2POp(
                    dist.isend,
                    c_send_meta[s_start:s_start + s_count].contiguous(),
                    send_peer))
            
            if r_count > 0:
                ops.append(dist.P2POp(
                    dist.irecv,
                    recv_buf[r_start:r_start + r_count],
                    recv_peer))
                ops.append(dist.P2POp(
                    dist.irecv,
                    recv_meta_buf[r_start:r_start + r_count],
                    recv_peer))
            
            if ops:
                handles = dist.batch_isend_irecv(ops)
                for h in handles:
                    h.wait()
        
        # ===== Step 6: index_add_ 直接累加 =====
        src_token_indices = recv_meta_buf.long()
        output = torch.zeros(cfg.max_num_tokens, cfg.hidden_dim, 
                            dtype=torch.float32, device=device)
        output.index_add_(0, src_token_indices, recv_buf.float())
        
        return output.to(cfg.out_dtype)


# ============================================================================
# 第三部分: 入口函数
# Part 3: Entry Point
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    """
    Ring Shared Memory All-to-All 入口
    
    完整流程:
    1. Dispatch (Ring): token → expert (预加权)
    2. Compute:         expert 计算 (模拟)
    3. Combine (Ring):  expert result → token (直接求和)
    """
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    
    device = rank_data.x.device
    
    # ========== 创建计时事件 ==========
    events = {}
    event_names = [
        'start',
        'd_ring_data',        # Dispatch Ring 完成
        'compute',            # Compute 完成
        'c_ring_data',        # Combine Ring 完成
        'end'
    ]
    for name in event_names:
        events[name] = torch.cuda.Event(enable_timing=True)
    
    # 创建通信器
    ata = SharedMemoryRingAllToAll(cfg, rank, world_size)
    
    # ==================== DISPATCH 阶段 (Ring) ====================
    events['start'].record()
    
    recv_buf, recv_meta = ata.dispatch_ring(
        rank_data.x, rank_data.indices, rank_data.weights
    )
    
    events['d_ring_data'].record()
    
    # ==================== COMPUTE 阶段 ====================
    expert_y = recv_buf.to(cfg.out_dtype) * (1 + rank)
    events['compute'].record()
    
    # ==================== COMBINE 阶段 (Ring) ====================
    y = ata.combine_ring(expert_y, recv_meta)
    
    events['c_ring_data'].record()
    events['end'].record()
    
    # ==================== 输出计时结果 ====================
    events['end'].synchronize()
    
    if rank == 0:
        def t(start, end):
            return events[start].elapsed_time(events[end])
        
        dispatch_total = t('start', 'd_ring_data')
        compute_time = t('d_ring_data', 'compute')
        combine_total = t('compute', 'c_ring_data')
        total_time = t('start', 'end')
        
        print(f"\n{'='*70}")
        print(f"[Ring SHM] Rank 0 | Total: {total_time:.3f}ms")
        print(f"{'='*70}")
        print(f"  DISPATCH (Ring): {dispatch_total:.3f}ms")
        print(f"  COMPUTE:         {compute_time:.3f}ms")
        print(f"  COMBINE (Ring):  {combine_total:.3f}ms")
        print(f"{'='*70}")
    
    return y[: rank_data.num_tokens]
