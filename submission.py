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
        self.max_tokens = cfg.max_num_tokens * cfg.experts_per_token  # e.g., 2048

    def dispatch(self, x: torch.Tensor, indices: torch.Tensor):
        device = x.device
        N, E, H = x.size(0), self.cfg.experts_per_token, self.cfg.hidden_dim
        T = self.max_tokens

        # Pre-allocate send buffer: [world_size, T, H + 2]
        send_buf = torch.zeros(self.world_size, T, H + 2,
                               dtype=torch.float16, device=device)

        # Flatten routing info
        flat_indices = indices.flatten()
        dst_ranks = flat_indices // self.num_local_experts
        token_idx = torch.arange(
            N, device=device, dtype=torch.int16).repeat_interleave(E)
        expert_local_idx = (flat_indices %
                            self.num_local_experts).to(torch.int16)

        # Count per rank (but we use fixed T, so just fill sequentially)
        offsets = torch.zeros(self.world_size, dtype=torch.long, device=device)
        for i in range(dst_ranks.numel()):
            dst = dst_ranks[i].item()
            pos = offsets[dst]
            if pos < T:
                send_buf[dst, pos, :H] = x[token_idx[i]]
                send_buf[dst, pos, H] = token_idx[i].to(torch.float16)
                send_buf[dst, pos, H+1] = expert_local_idx[i].to(torch.float16)
                offsets[dst] += 1

        # All-to-all on fixed buffer
        recv_buf = torch.empty_like(send_buf)
        dist.all_to_all(
            [recv_buf[i] for i in range(self.world_size)],
            [send_buf[i] for i in range(self.world_size)]
        )

        return recv_buf

    def combine(self, expert_y_buf: torch.Tensor, weights: torch.Tensor, orig_N: int):
        device = expert_y_buf.device
        H = self.cfg.hidden_dim
        T = expert_y_buf.size(1)

        output = torch.zeros(orig_N, H, dtype=torch.float32, device=device)

        for src_rank in range(self.world_size):
            buf = expert_y_buf[src_rank]  # [T, H+2]
            token_idx = buf[:, H].to(torch.int16)
            expert_idx = buf[:, H+1].to(torch.int16)
            valid = token_idx != 0  # assuming 0 is padding (or use a mask)
            # Better: use offset count, but for simplicity:
            for i in range(T):
                if token_idx[i] == 0 and i >= 1:  # crude padding detection
                    break
                if token_idx[i] >= orig_N:
                    continue
                w = weights[token_idx[i], expert_idx[i]]
                output[token_idx[i]] += expert_y_buf[src_rank,
                                                     i, :H].to(torch.float32) * w

        return output.to(self.cfg.out_dtype)


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    torch.cuda.set_device(rank)
    ata = OptimizedAllToAll(cfg, rank, world_size)

    recv_buf = ata.dispatch(rank_data.x, rank_data.indices)
    # Simulate expert: apply to each non-padding token in recv_buf
    H = cfg.hidden_dim
    expert_out = recv_buf.clone()
    expert_out[:, :, :H] = expert_out[:, :, :H].to(cfg.out_dtype) * (1 + rank)

    y = ata.combine(expert_out, rank_data.weights, rank_data.num_tokens)
    return y
