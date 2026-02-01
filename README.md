# NVIDIA 多 GPU 算子测试框架

## 🚀 快速开始

### 一条命令测试所有
```bash
python run_test.py
```

### 只测试正确性
```bash
python run_test.py --mode test
```

### 只测试性能
```bash
python run_test.py --mode benchmark
```

## 📋 常用命令

| 命令 | 说明 |
|------|------|
| `python run_test.py` | 完整测试（正确性+性能）|
| `python run_test.py --quick` | 快速测试模式 |
| `python run_test.py --mode test` | 只测正确性 |
| `python run_test.py --mode benchmark` | 只测性能 |
| `python run_test.py --world-size 4` | 指定 GPU 数量 |
| `python run_test.py --gpus 6,7` | 使用指定 GPU |
| `python run_test.py --info` | 查看设备信息 |

## 🛠️ 开发流程

1. **修改算子**: 编辑 `submission.py` 中的 `custom_kernel` 函数
2. **快速验证**: `python run_test.py --quick`
3. **完整测试**: `python run_test.py`
4. **查看结果**: 关注几何平均时间，持续优化

## 📁 文件结构

```
amd_kernel/
├── run_test.py        # 一键测试入口 ⭐
├── submission.py      # 你的算子实现 ✏️
├── reference.py       # 参考实现
├── local_eval.py      # 本地评估器
├── device_manager.py  # GPU 设备管理
├── test_config.py     # 测试配置
└── utils.py           # 工具函数
```

## 🔧 硬件适配

当前服务器配置：
- 6 × RTX 4060 (24GB)
- 2 × L20 (48GB)

框架会自动检测所有 GPU，你也可以手动指定：

```bash
# 只用 L20 测试
python run_test.py --gpus 6,7

# 只用 4060 测试
python run_test.py --gpus 0,1,2,3,4,5
```

## 💡 提示

- 建议使用同型号 GPU 进行测试，避免性能瓶颈
- 快速模式适合开发迭代，完整测试用于最终验证
- 所有输出直接在终端显示，无需额外配置
