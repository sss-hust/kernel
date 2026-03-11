import torch
import os
import torch.distributed as dist
import dataclasses
from task import input_t, output_t


# ---------------- MoE config ----------------
@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16


class PyTorchAllToAll:
    def __init__(self, cfg: MoEConfig, rank: int, world_size: int):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.num_local_experts = cfg.num_experts // world_size
        self.max_recv = cfg.max_num_tokens * world_size

    def dispatch(self, dp_x: torch.Tensor, indices: torch.Tensor):
        device = dp_x.device
        cfg = self.cfg
        # 提取需要的数据，每个GPU有多少token去往其他GPU，每个gpu将获取的token和元数据
        send_counts = [0] * self.world_size
        token_map = [[]for _ in range(self.world_size)]
        meta_map = [[]for _ in range(self.world_size)]
        for t,expert_list in enumerate(indices.tolist()):
            for k,e in enumerate(expert_list):
                dst_rank = e //self.num_local_experts
                send_counts[dst_rank] += 1
                token_map[dst_rank].append(t)
                meta_map[dst_rank].extend(
                    [e,self.rank,t,k,0]
                )
        send_counts_t = torch.tensor(send_counts,dtype=torch.long,device=device)
        recv_counts_t = torch.empty(self.world_size,dtype=torch.long,device=device)
        dist.all_to_all_single(recv_counts_t,send_counts_t)

        total_recv = int(recv_counts_t.sum().item())
        send_buf = torch.cat([dp_x[idx_list]for idx_list in token_map],dim=0)
        recv_buf = torch.empty(
            total_recv,cfg.hidden_dim,dtype=cfg.in_dtype,device=device
        )

        dist.all_to_all_single(
            recv_buf,
            send_buf,
            output_split_sizes=recv_counts_t.tolist(),
            input_split_sizes=send_counts_t.tolist(),
        )
        dist.all_to_all_single(
            recv_meta.view(-1),
            send_meta.view(-1),
            output_split_sizes=[c*self.META_DIM for c in recv_counts_t.tolist()],
            input_split_sizes=[c*self.META_DIM for c in send_counts_t.tolist()],
        )
        recv_meta = recv_meta.view(-1,self.META_DIM)

        expert_num_tokens = torch.zeros(
            self.num_local_experts,dtype=torch.int16,device=device
        )
        expert_x = torch.empty(
            (self.num_local_experts,self.max_recv,cfg.hidden_dim),
            dtype=cfg.in_dtype,
            device=device
        )
        expert_meta = torch.empty(
            (self.num_local_experts,self.max_recv,self.META_DIM),
            dtype=torch.int16,
            device=device,
        )
        for i in range(total_recv):
            global_eid = int(recv_meta[i,0].item())
            local_eid = global_eid % self.num_local_experts
            expert_x[local_eid,expert_num_tokens[local_eid]] = recv_buf[i]
            expert_meta[local_eid,expert_num_tokens[local_eid]] = recv_meta[i]
            expert_num_tokens[local_eid] += 1
        return expert_x,expert_meta,expert_num_tokens
        
    def combine(self,expert_x,expert_meta,expert_num_tokens):
        device = expert_x.device
        cfg = self.cfg
        
        send_counts = [0]*self.world_size
        y_map = [[]for _ in range(self.world_size)]
        meta_map = [[]for _ in range(self.world_size)]
        
        for local_eid in range(self.num_local_experts):
            num_tokens = int(expert_num_tokens[local_eid].item())
            for i in range(num_tokens):
                meta = expert_meta[local_eid,i]
                src_rank = int(meta[1].item())
                src_token = int(meta[2].item())
                src_k = int(meta[3].item())
                send_counts[src_rank] += 1
                y_map[src_rank].append(src_token)
                meta_map[src_rank].extend(
                    [src_token,src_k,0]
                )
        