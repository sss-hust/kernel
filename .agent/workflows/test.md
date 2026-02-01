---
description: 一键测试算子的正确性和性能
---

# 测试算子

// turbo-all

## 快速开始

1. 运行完整测试（正确性 + 性能）:
```bash
python run_test.py
```

2. 只测试正确性:
```bash
python run_test.py --mode test
```

3. 只测试性能:
```bash
python run_test.py --mode benchmark
```

## 常用选项

- 快速模式（减少测试用例）:
```bash
python run_test.py --quick
```

- 指定 GPU 数量:
```bash
python run_test.py --world-size 4
```

- 指定特定 GPU:
```bash
python run_test.py --gpus 6,7
```

- 查看设备信息:
```bash
python run_test.py --info
```

## 开发工作流

1. 修改 `submission.py` 中的 `custom_kernel` 函数
2. 运行 `python run_test.py --quick` 快速验证
3. 通过后运行 `python run_test.py` 完整测试
4. 查看性能数据，继续优化
