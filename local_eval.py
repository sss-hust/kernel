# local_eval.py
"""
本地测试评估脚本 - 不依赖 POPCORN_FD，直接在终端输出结果
"""
import copy
import dataclasses
import multiprocessing
import time
import os
import sys
import math
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import torch
import torch.cuda

from utils import set_seed, clear_l2_cache


# 颜色输出支持
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colored(text: str, color: str) -> str:
    """添加颜色"""
    return f"{color}{text}{Colors.ENDC}"


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float
    
    def __str__(self):
        return f"mean: {self.mean/1e6:.3f}ms, std: {self.std/1e6:.3f}ms, best: {self.best/1e6:.3f}ms (runs: {self.runs})"


def calculate_stats(durations: list):
    """计算统计数据"""
    runs = len(durations)
    total = sum(durations)
    best = min(durations)
    worst = max(durations)
    avg = total / runs
    variance = sum(map(lambda x: (x - avg) ** 2, durations))
    std = math.sqrt(variance / (runs - 1)) if runs > 1 else 0
    err = std / math.sqrt(runs) if runs > 0 else 0
    return Stats(runs=runs, mean=avg, std=std, err=err, best=float(best), worst=float(worst))


def _clone_data(data, rank: int):
    """递归克隆数据到指定 GPU"""
    if isinstance(data, tuple):
        return tuple(_clone_data(x, rank) for x in data)
    elif isinstance(data, list):
        return [_clone_data(x, rank) for x in data]
    elif isinstance(data, dict):
        return {k: _clone_data(v, rank) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        device = f"cuda:{rank}"
        return data.clone().to(device)
    else:
        return data


def wrap_check_implementation(data, submission_output):
    """包装正确性检查"""
    from reference import check_implementation
    result = check_implementation(data, submission_output)
    if isinstance(result, tuple):
        return result
    else:
        return not bool(result), result


def _run_distributed_test(args):
    """在单个进程中运行分布式测试"""
    test_args, rank, world_size = args
    test_args_copy = test_args.copy()
    submission_module = test_args_copy.pop("submission_module", "submission")
    
    import importlib
    sub_mod = importlib.import_module(submission_module)
    custom_kernel = sub_mod.custom_kernel
    from reference import generate_input
    import torch.distributed as dist
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    
    try:
        dist.init_process_group(
            "nccl", 
            init_method="env://", 
            rank=rank, 
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )
        
        data = generate_input(**test_args_copy, rank=rank)
        torch.cuda.synchronize()
        submission_output = custom_kernel(_clone_data(data, rank))
        torch.cuda.synchronize()
        return wrap_check_implementation(data, submission_output)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_benchmark(args):
    """在单个进程中运行分布式基准测试"""
    test_args, rank, world_size, max_repeats, max_time_ns = args
    test_args_copy = test_args.copy()
    submission_module = test_args_copy.pop("submission_module", "submission")
    
    import importlib
    sub_mod = importlib.import_module(submission_module)
    custom_kernel = sub_mod.custom_kernel
    from reference import generate_input
    import torch.distributed as dist
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"
    
    try:
        dist.init_process_group(
            "nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=torch.device(f'cuda:{rank}')
        )
        
        durations = []
        data = generate_input(**test_args_copy, rank=rank)
        check_copy = _clone_data(data, rank)
        
        # 正确性检查
        output = custom_kernel(_clone_data(data, rank))
        good, message = wrap_check_implementation(check_copy, output)
        if not good:
            return message
        
        # 基准测试
        bm_start_time = time.perf_counter_ns()
        for i in range(max_repeats):
            clear_l2_cache()
            torch.cuda.synchronize()
            dist.barrier()
            
            if rank == 0:
                start_time = time.perf_counter_ns()
            
            output = custom_kernel(_clone_data(data, rank))
            
            torch.cuda.synchronize()
            dist.barrier()
            
            if rank == 0:
                end_time = time.perf_counter_ns()
                duration = end_time - start_time
                durations.append(duration)
            
            del output
            
            if rank == 0 and i > 1:
                total_bm_duration = time.perf_counter_ns() - bm_start_time
                stats = calculate_stats(durations)
                should_stop = (
                    stats.err / stats.mean < 0.001 or
                    stats.mean * stats.runs > max_time_ns or
                    total_bm_duration > 60e9
                )
            else:
                should_stop = False
            
            stop_tensor = torch.tensor(should_stop, dtype=torch.bool, device=f'cuda:{rank}')
            dist.broadcast(stop_tensor, 0)
            
            if stop_tensor.item():
                break
        
        if rank == 0:
            return calculate_stats(durations)
        else:
            return Stats(runs=len(durations), mean=0.0, std=0.0, err=0.0, best=0.0, worst=0.0)
    
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class LocalEvaluator:
    """本地评估器"""
    
    def __init__(self, world_size: int = None, gpu_indices: list = None):
        """
        初始化评估器
        
        Args:
            world_size: GPU 数量，默认自动检测
            gpu_indices: 指定使用的 GPU 索引，默认使用全部
        """
        if world_size is None:
            world_size = torch.cuda.device_count()
        
        self.world_size = world_size
        self.gpu_indices = gpu_indices or list(range(world_size))
        
        # 设置 CUDA 可见设备
        if gpu_indices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    
    def run_test(self, test_args: dict) -> tuple:
        """
        运行单个测试用例
        
        Returns:
            (passed: bool, message: str)
        """
        test_args = dict(test_args)
        test_args["world_size"] = self.world_size
        
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(self.world_size) as pool:
            args_list = [(test_args, i, self.world_size) for i in range(self.world_size)]
            results = pool.map(_run_distributed_test, args_list)
        
        # 检查所有 rank 的结果
        all_passed = all(r[0] for r in results)
        errors = [f"rank {i}: {r[1]}" for i, r in enumerate(results) if not r[0]]
        
        return all_passed, "\n".join(errors) if errors else ""
    
    def run_benchmark(self, test_args: dict, max_repeats: int = 100, max_time_ns: float = 10e9) -> Stats:
        """
        运行单个基准测试用例
        
        Returns:
            Stats 对象或错误信息
        """
        test_args = dict(test_args)
        test_args["world_size"] = self.world_size
        
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(self.world_size) as pool:
            args_list = [
                (test_args, i, self.world_size, max_repeats, max_time_ns)
                for i in range(self.world_size)
            ]
            results = pool.map(_run_distributed_benchmark, args_list)
        
        # rank 0 的结果是有效的统计数据
        for i, result in enumerate(results):
            if not isinstance(result, Stats):
                return result  # 返回错误信息
        
        return results[0]  # 返回 rank 0 的统计


def run_tests(cases: list, world_size: int = None, verbose: bool = True, submission_module: str = 'submission') -> bool:
    """
    运行所有测试用例
    
    Args:
        cases: TestCase 对象列表
        world_size: GPU 数量
        verbose: 是否打印详细信息
        submission_module: 评测模块名
    
    Returns:
        是否全部通过
    """
    evaluator = LocalEvaluator(world_size=world_size)
    
    print(f"\n🧪 {colored('正确性测试', Colors.BOLD)} (world_size={evaluator.world_size})")
    print("=" * 60)
    
    all_passed = True
    for i, case in enumerate(cases):
        test_args = {
            "num_experts": case.num_experts,
            "experts_per_token": case.experts_per_token,
            "hidden_dim": case.hidden_dim,
            "max_num_tokens": case.max_num_tokens,
            "seed": case.seed,
            "submission_module": submission_module,
        }
        
        if verbose:
            print(f"  [{i+1}/{len(cases)}] Testing: experts={case.num_experts}, "
                  f"tokens={case.max_num_tokens}, hidden={case.hidden_dim}...", end=" ")
            sys.stdout.flush()
        
        try:
            passed, message = evaluator.run_test(test_args)
            if passed:
                print(colored("✓ PASS", Colors.GREEN))
            else:
                print(colored("✗ FAIL", Colors.RED))
                if message:
                    print(f"    Error: {message}")
                all_passed = False
        except Exception as e:
            print(colored("✗ ERROR", Colors.RED))
            print(f"    Exception: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print(colored("✓ 所有测试通过!", Colors.GREEN + Colors.BOLD))
    else:
        print(colored("✗ 部分测试失败", Colors.RED + Colors.BOLD))
    
    return all_passed


def run_benchmarks(cases: list, world_size: int = None, verbose: bool = True, submission_module: str = 'submission') -> list:
    """
    运行所有基准测试
    
    Args:
        cases: TestCase 对象列表
        world_size: GPU 数量
        verbose: 是否打印详细信息
        submission_module: 评测模块名
    
    Returns:
        Stats 对象列表
    """
    evaluator = LocalEvaluator(world_size=world_size)
    
    print(f"\n⚡ {colored('性能基准测试', Colors.BOLD)} (world_size={evaluator.world_size})")
    print("=" * 60)
    
    results = []
    
    # 预热
    if cases:
        print("  🔥 Warming up...", end=" ")
        sys.stdout.flush()
        warm_args = {
            "num_experts": cases[0].num_experts,
            "experts_per_token": cases[0].experts_per_token,
            "hidden_dim": cases[0].hidden_dim,
            "max_num_tokens": cases[0].max_num_tokens,
            "seed": cases[0].seed,
            "submission_module": submission_module,
        }
        evaluator.run_benchmark(warm_args, max_repeats=10, max_time_ns=1e8)
        print("Done")
    
    for i, case in enumerate(cases):
        test_args = {
            "num_experts": case.num_experts,
            "experts_per_token": case.experts_per_token,
            "hidden_dim": case.hidden_dim,
            "max_num_tokens": case.max_num_tokens,
            "seed": case.seed,
            "submission_module": submission_module,
        }
        
        if verbose:
            print(f"  [{i+1}/{len(cases)}] Benchmarking: experts={case.num_experts}, "
                  f"tokens={case.max_num_tokens}, hidden={case.hidden_dim}...", end=" ")
            sys.stdout.flush()
        
        try:
            result = evaluator.run_benchmark(test_args)
            if isinstance(result, Stats):
                print(colored(f"✓ {result.mean/1e6:.3f}ms", Colors.CYAN))
                results.append(result)
            else:
                print(colored(f"✗ {result}", Colors.RED))
                results.append(None)
        except Exception as e:
            print(colored(f"✗ ERROR: {e}", Colors.RED))
            results.append(None)
    
    print("=" * 60)
    
    # 计算几何平均
    valid_results = [r for r in results if r is not None]
    if valid_results:
        geo_mean = math.exp(sum(math.log(r.mean) for r in valid_results) / len(valid_results))
        print(f"📊 几何平均: {colored(f'{geo_mean/1e6:.3f}ms', Colors.CYAN + Colors.BOLD)}")
    
    return results


if __name__ == "__main__":
    # 简单测试
    set_seed(42)
    from test_config import get_test_cases, get_benchmark_cases
    
    cases = get_test_cases(world_size=2, quick=True)
    run_tests(cases, world_size=2)
