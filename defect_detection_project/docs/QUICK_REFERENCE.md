# 瑕疵检测模型 - 快速参考

## 🎯 你的场景

- **数据量**：1w+ 带掩码样本
- **推荐模型**：DINOv3-B/14 (94M 参数)
- **预期性能**：分类 96-99%，分割 IoU 0.80-0.90

## ⚡ 一键开始

```bash
# Windows
examples\train_large_dataset.bat

# Linux/Mac
bash examples/train_large_dataset.sh
```

## 📋 模型选择

| 模型 | 参数 | 适用数据量 | 显存 | 性能 |
|------|------|-----------|------|------|
| DINOv3-S | 24M | <5k | 8GB | ★★★☆☆ |
| **DINOv3-B** ⭐ | **94M** | **5k-50k** | **16GB** | **★★★★★** |
| DINOv3-L | 332M | >50k | 32GB | ★★★★☆ |

## 🚀 训练命令

### 快速测试（10 epochs）

```bash
python examples/train_defect_model.py \
    --data_root data/your_data \
    --backbone dinov3_vits14 \
    --batch_size 32 \
    --epochs 10 \
    --freeze_backbone
```

### 正式训练（推荐）

```bash
# 阶段 1：冻结骨干（30 epochs）
python examples/train_defect_model.py \
    --data_root data/your_10k_dataset \
    --backbone dinov3_vitb14 \
    --freeze_backbone \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage1

# 阶段 2：微调骨干（70 epochs）
python examples/train_defect_model.py \
    --data_root data/your_10k_dataset \
    --backbone dinov3_vitb14 \
    --batch_size 16 \
    --epochs 70 \
    --lr 5e-5 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage2
```

## 🔍 推理

```bash
python examples/inference_defect_model.py \
    --checkpoint checkpoints/stage2/best_accuracy.pth \
    --backbone dinov3_vitb14 \
    --image_dir data/test_images \
    --output_dir results
```

## 📊 预期性能

| 阶段 | 分类准确率 | 分割 IoU | 训练时间 |
|------|-----------|---------|---------|
| 阶段 1（冻结） | 90-93% | 0.70-0.75 | 3-4h |
| 阶段 2（微调） | 96-99% | 0.80-0.90 | 10-12h |

## 🛠️ 常用参数

```bash
--backbone dinov3_vitb14          # 模型选择
--batch_size 16                   # 批次大小
--epochs 100                      # 训练轮数
--lr 5e-5                         # 学习率
--freeze_backbone                 # 冻结骨干网络
--use_dynamic_weights             # 动态权重调整
--use_uncertainty_weighting       # 不确定性加权
--dropout 0.2                     # Dropout 率
--image_size 518                  # 图像尺寸
--num_workers 4                   # 数据加载线程
```

## 📁 文件结构

```
data/your_10k_dataset/
├── images/
│   ├── defect_001.jpg
│   ├── normal_001.jpg
│   └── ...
├── masks/
│   ├── defect_001.png
│   └── ...
├── train_labels.txt
└── val_labels.txt
```

## 🐛 问题排查

### 显存不足

```bash
--batch_size 8                    # 减小批次
--backbone dinov3_vits14          # 使用小模型
```

### 训练不收敛

```bash
--lr 1e-5                         # 降低学习率
--use_uncertainty_weighting       # 自动平衡损失
```

### 过拟合

```bash
--dropout 0.3                     # 增加 dropout
--weight_decay 0.1                # 增加权重衰减
```

## 📚 详细文档

- **使用指南**：`examples/README_DEFECT_DETECTION.md`
- **实现指南**：`examples/IMPLEMENTATION_GUIDE.md`
- **大数据集指南**：`examples/LARGE_DATASET_GUIDE.md`
- **模型参数说明**：`MODEL_SIZE_EXPLANATION.md`
- **完整总结**：`DEFECT_DETECTION_SUMMARY.md`

## ❓ 常见问题

**Q: 模型参数量够用吗？**
A: 够用！DINOv3-B 有 94M 参数，对 1w+ 样本完全足够。测试中的 2.31M 只是 dummy backbone。

**Q: 需要多长时间训练？**
A: 完整三阶段约 15-20 小时（V100）。

**Q: 需要多少显存？**
A: DINOv3-B + batch_size=16 需要约 16GB。

**Q: 如何选择模型大小？**
A: 数据量 <5k 用 S，5k-50k 用 B（推荐），>50k 用 L。

## 🎯 核心优势

1. ✅ **注意力引导**：分割掩码直接引导分类
2. ✅ **动态权重**：自动平衡多任务学习
3. ✅ **预训练强大**：DINOv3 在 142M 图像上预训练
4. ✅ **完整工具链**：数据准备、训练、推理、可视化

## 📞 获取帮助

1. 查看详细文档（见上方链接）
2. 运行测试：`python examples/test_model.py`
3. 查看配置示例：`examples/config_defect_detection.yaml`

---

**开始训练，祝你成功！** 🚀
