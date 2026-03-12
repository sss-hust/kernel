#!/usr/bin/env python3
# run_test.py
"""
一键测试入口脚本 - 让你专注于算子优化

使用方法:
    python run_test.py                  # 运行完整测试（正确性 + 性能）
    python run_test.py --mode test      # 只运行正确性测试
    python run_test.py --mode benchmark # 只运行性能基准测试
    python run_test.py --quick          # 快速测试模式
    python run_test.py --world-size 4   # 指定 GPU 数量
    python run_test.py --gpus 6,7       # 指定使用的 GPU
    python run_test.py --info           # 查看设备信息
    python run_test.py --mode benchmark --gpus 0,1,2,3 --module submission_ring_shm

"""
import argparse
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║           🚀 NVIDIA Multi-GPU Kernel Test Framework          ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def show_device_info():
    """显示设备信息"""
    from device_manager import get_device_manager
    dm = get_device_manager()
    dm.print_info()
    return dm


def parse_gpu_list(gpu_str: str) -> list:
    """解析 GPU 列表字符串，如 '0,1,2' 或 '6,7'"""
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='一键测试你的 CUDA 算子',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_test.py                    # 完整测试
  python run_test.py --mode test        # 只测正确性
  python run_test.py --mode benchmark   # 只测性能
  python run_test.py --quick            # 快速模式
  python run_test.py -w 4               # 使用4个GPU
  python run_test.py --module submission_ring_shm  # 指定测试特定模块
  python run_test.py --gpus 0,1,2,3     # 指定GPU
  python run_test.py --info             # 查看设备信息
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['all', 'test', 'benchmark', 'profile'],
        default='all',
        help='测试模式: all(全部), test(正确性), benchmark(性能), profile(分析)'
    )
    
    parser.add_argument(
        '--world-size', '-w',
        type=int,
        default=None,
        help='使用的 GPU 数量（默认自动检测）'
    )
    
    parser.add_argument(
        '--gpus', '-g',
        type=str,
        default=None,
        help='指定使用的 GPU 索引，如 "0,1,2" 或 "6,7"'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='快速测试模式（减少测试用例）'
    )
    
    parser.add_argument(
        '--module',
        type=str,
        default='submission',
        help='指定评测的模块名称 (默认: submission)'
    )
    
    parser.add_argument(
        '--case', '-c',
        type=int,
        default=None,
        help='只运行指定的测试用例（索引从0开始）'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='显示 GPU 设备信息'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # 显示设备信息
    dm = show_device_info()
    
    if args.info:
        return 0
    
    # 处理 GPU 选择
    gpu_indices = parse_gpu_list(args.gpus)
    
    if gpu_indices:
        world_size = len(gpu_indices)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"📌 使用指定 GPU: {gpu_indices}")
    elif args.world_size:
        world_size = args.world_size
        print(f"📌 使用 {world_size} 个 GPU")
    else:
        world_size = dm.get_recommended_world_size()
        print(f"📌 自动检测到 {world_size} 个 GPU")
    
    if world_size == 0:
        print("❌ 错误: 未检测到 GPU，无法运行测试")
        return 1
    
    # 设置随机种子
    from utils import set_seed
    set_seed(args.seed)
    
    # 获取测试用例
    from test_config import get_test_cases, get_benchmark_cases
    
    test_cases = get_test_cases(world_size=world_size, quick=args.quick)
    benchmark_cases = get_benchmark_cases(world_size=world_size, quick=args.quick)
    
    # 如果指定了单个用例
    if args.case is not None:
        if args.case < len(test_cases):
            test_cases = [test_cases[args.case]]
        if args.case < len(benchmark_cases):
            benchmark_cases = [benchmark_cases[args.case]]
    
    # 导入评估模块
    from local_eval import run_tests, run_benchmarks
    
    success = True
    
    # 运行测试
    if args.mode in ['all', 'test']:
        passed = run_tests(test_cases, world_size=world_size, submission_module=args.module)
        success = success and passed
    
    if args.mode in ['all', 'benchmark']:
        results = run_benchmarks(benchmark_cases, world_size=world_size, submission_module=args.module)
        # 检查是否有失败的基准测试
        if any(r is None for r in results):
            success = False
    
    if args.mode == 'profile':
        print("\n📊 性能分析模式（Profile）尚未在本地评估器中实现")
        print("   请使用 PyTorch Profiler 手动分析")
    
    print()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
