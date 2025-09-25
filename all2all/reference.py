# pytorch_all2all.py
import os
import torch
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


# ---------------- data per dp rank ----------------
class RankTestData:
    def __init__(self, cfg: MoEConfig, rng: torch.Generator, rank: int):
        device = torch.device(f"cuda:{rank}")
        self.num_tokens = int(
            torch.randint(
                1, cfg.max_num_tokens, [1], generator=rng, device=device
            ).item()
        )
        # token expert map
        self.indices = torch.empty(
            self.num_tokens, cfg.experts_per_token, dtype=torch.int32, device=device
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(
                cfg.num_experts, generator=rng, device=device)
            self.indices[i] = perm[: cfg.experts_per_token]
        # topk weights
        self.weights = torch.rand(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.float32,
            generator=rng,
            device=device,
        )
        # dp tokens, input of dispatch
        self.x = torch.randn(
            self.num_tokens,
            cfg.hidden_dim,
            dtype=cfg.in_dtype,
            generator=rng,
            device=device,
        )


def generate_input(
    num_experts, experts_per_token, hidden_dim, max_num_tokens, seed, rank, world_size
):
    device = torch.device(f"cuda:{rank}")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + rank)

    cfg = MoEConfig(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        hidden_dim=hidden_dim,
        max_num_tokens=max_num_tokens,
        in_dtype=torch.float16,
        out_dtype=torch.float16,
    )
    rank_data = RankTestData(cfg, gen, rank)
    return cfg, rank_data, rank, world_size


def ref_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data

    ata = PyTorchAllToAll(cfg, rank, world_size)

    expert_num, expert_x, expert_meta = ata.dispatch(
        rank_data.x, rank_data.indices)
    expert_y = expert_x.to(cfg.out_dtype) * (1 + rank)
    y = torch.zeros(
        cfg.max_num_tokens,
        cfg.hidden_dim,
        dtype=cfg.out_dtype,
        device=rank_data.x.device,
    )

    ata.combine(y, rank_data.weights, expert_meta, expert_y, expert_num)

    return y[: rank_data.num_tokens]


def check_implementation(data: input_t, output: output_t):
    expected = ref_kernel(data)
    if output.device != expected.device:
        return False, f"Output device mismatch: {output.device} != {expected.device}"
    res = torch.allclose(output, expected, rtol=1e-2, atol=5e-3)
    if not res:
        return False, f"Output values mismatch, {output} != {expected}"

    return True, ""
