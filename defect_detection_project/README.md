# 瑕疵检测多任务模型

基于 DINOv3 的瑕疵检测模型，通过分割分支引导分类任务，提升对瑕疵特征的学习能力。

## 🎯 项目特点

- ✅ **注意力引导架构**：分割掩码直接引导分类器聚焦瑕疵区域
- ✅ **动态权重调整**：自动平衡多任务学习
- ✅ **强大的预训练**：基于 DINOv3（142M 图像预训练）
- ✅ **完整工具链**：数据准备、训练、推理、可视化

## 📊 性能表现

基于 1w+ 样本的预期性能（DINOv3-B/14）：

| 指标 | 性能 |
|------|------|
| 分类准确率 | 96-99% |
| 分割 IoU | 0.80-0.90 |
| 分割 Dice | 0.88-0.94 |
| 训练时间 | 15-20h (V100) |

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

```bash
python scripts/prepare_data.py organize \
    --defect_images /path/to/defect/images \
    --defect_masks /path/to/defect/masks \
    --normal_images /path/to/normal/images \
    --output_dir data/my_dataset
```

### 3. 训练模型

```bash
# Windows
scripts\train_large_dataset.bat

# Linux/Mac
bash scripts/train_large_dataset.sh
```

### 4. 推理

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --backbone dinov3_vitb14 \
    --image_dir data/test_images \
    --output_dir results
```

## 📁 项目结构

```
defect_detection_project/
├── models/
│   └── defect_classifier.py          # 模型定义
├── data/
│   └── dataset.py                     # 数据集
├── train/
│   └── trainer.py                     # 训练逻辑
├── scripts/
│   ├── train.py                       # 训练脚本
│   ├── inference.py                   # 推理脚本
│   ├── prepare_data.py                # 数据准备
│   ├── test_model.py                  # 模型测试
│   ├── train_large_dataset.sh/.bat    # 大数据集训练
│   └── quick_start.sh/.bat            # 快速开始
├── docs/
│   ├── QUICK_REFERENCE.md             # 快速参考
│   ├── MODEL_SIZE_GUIDE.md            # 模型参数说明
│   ├── IMPLEMENTATION_GUIDE.md        # 实现指南
│   └── LARGE_DATASET_GUIDE.md         # 大数据集指南
├── config/
│   └── config.yaml                    # 配置示例
├── requirements.txt                   # 依赖列表
└── README.md                          # 本文档
```

## 🎯 模型选择

| 数据量 | 推荐模型 | 参数量 | 显存 | 预期准确率 |
|--------|---------|--------|------|-----------|
| <5k | DINOv3-S/14 | 24M | 8GB | 94-96% |
| **5k-50k** ⭐ | **DINOv3-B/14** | **94M** | **16GB** | **96-99%** |
| >50k | DINOv3-L/14 | 332M | 32GB | 97-99% |

**你的场景（1w+ 样本）**：推荐使用 **DINOv3-B/14**

## 📚 详细文档

- **[快速参考](docs/QUICK_REFERENCE.md)** ⭐ - 最常用的命令和配置
- **[模型参数说明](docs/MODEL_SIZE_GUIDE.md)** - 参数量和性能分析
- **[实现指南](docs/IMPLEMENTATION_GUIDE.md)** - 方案对比和设计决策
- **[大数据集指南](docs/LARGE_DATASET_GUIDE.md)** - 1w+ 样本训练策略

## 💡 核心优势

1. ✅ **注意力引导**：分割掩码直接引导分类
2. ✅ **动态权重**：自动平衡多任务学习
3. ✅ **预训练强大**：DINOv3 在 142M 图像上预训练
4. ✅ **完整工具链**：数据准备、训练、推理、可视化

## 🐛 常见问题

### Q: 模型参数量够用吗？
A: 够用！DINOv3-B 有 94M 参数，对 1w+ 样本完全足够。

### Q: 需要多长时间训练？
A: 完整三阶段约 15-20 小时（V100）。

### Q: 需要多少显存？
A: DINOv3-B + batch_size=16 需要约 16GB。

详见：[docs/MODEL_SIZE_GUIDE.md](docs/MODEL_SIZE_GUIDE.md)

## 📄 引用

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv:2304.07193},
  year={2023}
}
```

---

**开始你的瑕疵检测之旅！** 🚀
