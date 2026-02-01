# test_config.py
"""
测试配置模块 - 定义适配 NVIDIA GPU 的测试用例
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestCase:
    """测试用例配置"""
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    seed: int
    world_size: int
    
    def to_spec_string(self) -> str:
        """转换为规格字符串格式"""
        return (f"num_experts: {self.num_experts}; "
                f"experts_per_token: {self.experts_per_token}; "
                f"hidden_dim: {self.hidden_dim}; "
                f"max_num_tokens: {self.max_num_tokens}; "
                f"seed: {self.seed}; "
                f"world_size: {self.world_size}")


def get_test_cases(world_size: int = 8, quick: bool = False) -> List[TestCase]:
    """
    获取测试用例列表
    
    Args:
        world_size: GPU 数量
        quick: 是否使用快速测试模式（减少测试用例数量）
    """
    if quick:
        # 快速测试模式 - 只运行少量小规模用例
        return [
            TestCase(8, 2, 2048, 4, 1236, world_size),
            TestCase(64, 4, 2048, 8, 1234, world_size),
        ]
    
    # 完整测试用例（根据 NVIDIA GPU 显存调整）
    return [
        TestCase(8, 2, 6144, 4, 1236, world_size),
        TestCase(64, 6, 2048, 4, 1234, world_size),
        TestCase(64, 6, 2048, 8, 542, world_size),
        TestCase(128, 4, 2880, 16, 347, world_size),
        TestCase(128, 4, 2880, 32, 51, world_size),
        TestCase(128, 8, 4096, 64, 175, world_size),
        TestCase(128, 8, 4096, 128, 534, world_size),
        TestCase(256, 8, 7168, 64, 897, world_size),
        TestCase(256, 8, 7168, 128, 4, world_size),
    ]


def get_benchmark_cases(world_size: int = 8, quick: bool = False) -> List[TestCase]:
    """
    获取基准测试用例列表
    
    Args:
        world_size: GPU 数量
        quick: 是否使用快速测试模式
    """
    if quick:
        return [
            TestCase(8, 2, 4096, 8, 6635, world_size),
            TestCase(64, 6, 2048, 16, 1234, world_size),
        ]
    
    return [
        TestCase(8, 2, 6144, 16, 6635, world_size),
        TestCase(64, 6, 2048, 32, 1234, world_size),
        TestCase(128, 4, 2880, 128, 51, world_size),
        TestCase(128, 8, 4096, 256, 175, world_size),
        TestCase(256, 8, 7168, 256, 4, world_size),
    ]


def write_test_file(cases: List[TestCase], filepath: str):
    """将测试用例写入文件"""
    with open(filepath, 'w') as f:
        for case in cases:
            f.write(case.to_spec_string() + '\n')
