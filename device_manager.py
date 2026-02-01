# device_manager.py
"""
GPU 设备管理器 - 自动检测和管理系统 GPU 资源
"""
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GPUInfo:
    """GPU 设备信息"""
    index: int
    name: str
    total_memory_gb: float
    compute_capability: Tuple[int, int]
    
    def __str__(self):
        return f"GPU {self.index}: {self.name} ({self.total_memory_gb:.1f}GB)"


class DeviceManager:
    """GPU 设备管理器"""
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self._detect_gpus()
    
    def _detect_gpus(self):
        """检测系统中所有可用的 GPU"""
        if not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，请检查 CUDA 环境")
            return
        
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            self.gpus.append(GPUInfo(
                index=i,
                name=props.name,
                total_memory_gb=props.total_memory / (1024**3),
                compute_capability=(props.major, props.minor)
            ))
    
    @property
    def num_gpus(self) -> int:
        return len(self.gpus)
    
    def print_info(self):
        """打印所有 GPU 信息"""
        print("\n" + "=" * 60)
        print("🖥️  系统 GPU 信息")
        print("=" * 60)
        
        if not self.gpus:
            print("❌ 未检测到任何 GPU")
            return
        
        for gpu in self.gpus:
            print(f"  {gpu}")
        
        # 按型号分组
        groups = self.group_by_model()
        if len(groups) > 1:
            print("\n📊 GPU 分组:")
            for model, indices in groups.items():
                print(f"  • {model}: GPU {indices}")
        
        print("=" * 60 + "\n")
    
    def group_by_model(self) -> dict:
        """按 GPU 型号分组"""
        groups = {}
        for gpu in self.gpus:
            key = gpu.name
            if key not in groups:
                groups[key] = []
            groups[key].append(gpu.index)
        return groups
    
    def get_recommended_world_size(self) -> int:
        """获取推荐的 world_size"""
        return self.num_gpus if self.num_gpus > 0 else 1
    
    def get_gpu_indices(self, gpu_type: Optional[str] = None) -> List[int]:
        """
        获取 GPU 索引列表
        
        Args:
            gpu_type: 可选，指定 GPU 类型（如 "4060" 或 "L20"）
        """
        if gpu_type is None:
            return [gpu.index for gpu in self.gpus]
        
        indices = []
        for gpu in self.gpus:
            if gpu_type.lower() in gpu.name.lower():
                indices.append(gpu.index)
        return indices
    
    def get_homogeneous_groups(self) -> List[List[int]]:
        """获取同型号 GPU 分组（用于避免异构 GPU 的性能差异问题）"""
        groups = self.group_by_model()
        return list(groups.values())
    
    def get_min_memory_gb(self, indices: Optional[List[int]] = None) -> float:
        """获取指定 GPU 中的最小显存"""
        if indices is None:
            indices = [gpu.index for gpu in self.gpus]
        return min(self.gpus[i].total_memory_gb for i in indices) if indices else 0


# 全局设备管理器实例
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """获取全局设备管理器实例"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


if __name__ == "__main__":
    dm = get_device_manager()
    dm.print_info()
