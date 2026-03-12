# test_ring_shm.py
"""
Ring SHM All-to-All 方案的本地测试脚本
使用与 local_eval.py 类似的框架，但导入 submission_ring_shm 而非 submission
"""
import os
import sys
import math
import multiprocessing
import time
import torch
import torch.distributed as dist
from utils import set_seed, clear_l2_cache
from test_config import TestCase


# 颜色输出支持
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def colored(text, color):
    return f"{color}{text}{Colors.ENDC}"


def _clone_data(data, rank):
    if isinstance(data, tuple):
        return tuple(_clone_data(x, rank) for x in data)
    elif isinstance(data, torch.Tensor):
        return data.clone().to(f"cuda:{rank}")
    else:
        return data


def _run_correctness_test(args):
    """在单个进程中运行正确性测试"""
    test_args, rank, world_size = args

    # 导入 ring shm 版本
    from submission_ring_shm import custom_kernel
    from reference import generate_input, check_implementation

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12357"

    try:
        dist.init_process_group(
            "nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )

        data = generate_input(**test_args, rank=rank)
        torch.cuda.synchronize()
        output = custom_kernel(_clone_data(data, rank))
        torch.cuda.synchronize()
        return check_implementation(data, output)
    except Exception as e:
        return False, f"Exception: {e}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_benchmark(args):
    """在单个进程中运行性能测试"""
    test_args, rank, world_size, max_repeats = args

    from submission_ring_shm import custom_kernel
    from reference import generate_input, check_implementation

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12357"

    try:
        dist.init_process_group(
            "nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )

        data = generate_input(**test_args, rank=rank)

        # 正确性验证
        output = custom_kernel(_clone_data(data, rank))
        good, msg = check_implementation(data, output)
        if not good:
            return f"Correctness failed: {msg}"

        # 预热
        for _ in range(3):
            custom_kernel(_clone_data(data, rank))
        torch.cuda.synchronize()
        dist.barrier()

        # 基准测试
        durations = []
        for i in range(max_repeats):
            clear_l2_cache()
            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                start = time.perf_counter_ns()

            output = custom_kernel(_clone_data(data, rank))

            torch.cuda.synchronize()
            dist.barrier()

            if rank == 0:
                end = time.perf_counter_ns()
                durations.append(end - start)

            del output

        if rank == 0:
            avg = sum(durations) / len(durations)
            best = min(durations)
            return avg, best
        return 0.0, 0.0

    except Exception as e:
        return f"Exception: {e}"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    quick = "--quick" in sys.argv

    print(f"\n{'='*60}")
    print(f"  Ring SHM All-to-All 测试 (world_size={world_size})")
    print(f"{'='*60}")

    # 测试用例
    if quick:
        cases = [
            TestCase(8, 2, 2048, 4, 1236, world_size),
            TestCase(16, 4, 2048, 8, 1234, world_size),
        ]
    else:
        cases = [
            TestCase(8, 2, 2048, 4, 1236, world_size),
            TestCase(16, 4, 2048, 8, 1234, world_size),
            TestCase(32, 4, 2048, 16, 542, world_size),
            TestCase(64, 6, 2048, 8, 347, world_size),
        ]

    # ============ 正确性测试 ============
    print(f"\n🧪 正确性测试")
    print("-" * 50)

    all_passed = True
    for i, case in enumerate(cases):
        test_args = {
            "num_experts": case.num_experts,
            "experts_per_token": case.experts_per_token,
            "hidden_dim": case.hidden_dim,
            "max_num_tokens": case.max_num_tokens,
            "seed": case.seed,
            "world_size": world_size,
        }
        print(f"  [{i+1}/{len(cases)}] experts={case.num_experts}, "
              f"tokens={case.max_num_tokens}, hidden={case.hidden_dim}...", end=" ")
        sys.stdout.flush()

        mp_ctx = multiprocessing.get_context('spawn')
        with mp_ctx.Pool(world_size) as pool:
            args_list = [(test_args, r, world_size) for r in range(world_size)]
            results = pool.map(_run_correctness_test, args_list)

        passed = all(r[0] for r in results)
        if passed:
            print(colored("✓ PASS", Colors.GREEN))
        else:
            print(colored("✗ FAIL", Colors.RED))
            for r_idx, r in enumerate(results):
                if not r[0]:
                    print(f"    rank {r_idx}: {r[1]}")
            all_passed = False

    # ============ 性能测试 ============
    if all_passed:
        print(f"\n⚡ 性能基准测试")
        print("-" * 50)

        for i, case in enumerate(cases):
            test_args = {
                "num_experts": case.num_experts,
                "experts_per_token": case.experts_per_token,
                "hidden_dim": case.hidden_dim,
                "max_num_tokens": case.max_num_tokens,
                "seed": case.seed,
                "world_size": world_size,
            }
            print(f"  [{i+1}/{len(cases)}] experts={case.num_experts}, "
                  f"tokens={case.max_num_tokens}...", end=" ")
            sys.stdout.flush()

            mp_ctx = multiprocessing.get_context('spawn')
            with mp_ctx.Pool(world_size) as pool:
                args_list = [(test_args, r, world_size, 20) for r in range(world_size)]
                results = pool.map(_run_benchmark, args_list)

            # rank 0 的结果
            r0 = results[0]
            if isinstance(r0, tuple):
                avg, best = r0
                print(colored(f"avg={avg/1e6:.3f}ms, best={best/1e6:.3f}ms", Colors.CYAN))
            else:
                print(colored(f"✗ {r0}", Colors.RED))

    print(f"\n{'='*60}")
    if all_passed:
        print(colored("✓ 所有测试通过!", Colors.GREEN + Colors.BOLD))
    else:
        print(colored("✗ 部分测试失败", Colors.RED + Colors.BOLD))
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
