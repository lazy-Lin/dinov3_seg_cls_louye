# 大数据集训练指南（1w+ 样本）

## 🎯 针对你的场景

- **数据量**：1w+ 带掩码的瑕疵样本
- **推荐模型**：DINOv3-B/14（~94M 参数）
- **预期效果**：分类准确率 96-99%，分割 IoU 0.80-0.90

## 📊 模型选择对比

### 参数量和性能

| 模型 | 总参数 | 显存需求 | 训练速度 | 推荐场景 |
|------|--------|---------|---------|---------|
| **DINOv3-S/14** | ~24M | 8GB | 快 | 数据量 <5k，快速原型 |
| **DINOv3-B/14** ⭐ | ~94M | 16GB | 中 | 数据量 5k-50k，**推荐** |
| **DINOv3-L/14** | ~332M | 32GB | 慢 | 数据量 >50k，追求极致性能 |
| **DINOv3-g/14** | ~1.2B | 80GB | 很慢 | 数据量 >100k，研究用途 |

### 为什么推荐 DINOv3-B/14？

1. **参数量适中**：94M 参数足够学习复杂的瑕疵特征
2. **性能优秀**：在 1w+ 样本上能达到 SOTA 性能
3. **训练效率**：单卡 V100/A100 可以训练
4. **泛化能力强**：DINOv3 预训练质量高，迁移效果好

## 🚀 推荐训练流程

### 阶段 1：冻结骨干网络（30 epochs）

快速训练分类和分割头，验证数据质量。

```bash
python examples/train_defect_model.py \
    --data_root data/your_10k_dataset \
    --backbone dinov3_vitb14 \
    --freeze_backbone \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --weight_decay 0.01 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage1_frozen
```

**预期结果**：
- 分类准确率：90-93%
- 分割 IoU：0.70-0.75
- 训练时间：~3-4 小时（V100）

### 阶段 2：微调骨干网络（70 epochs）

解冻骨干网络，端到端微调。

```bash
python examples/train_defect_model.py \
    --data_root data/your_10k_dataset \
    --backbone dinov3_vitb14 \
    --batch_size 16 \
    --epochs 70 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage2_finetune
```

**预期结果**：
- 分类准确率：96-99%
- 分割 IoU：0.80-0.90
- 训练时间：~10-12 小时（V100）

### 阶段 3（可选）：不确定性加权微调（20 epochs）

进一步优化多任务平衡。

```bash
python examples/train_defect_model.py \
    --data_root data/your_10k_dataset \
    --backbone dinov3_vitb14 \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-5 \
    --use_uncertainty_weighting \
    --save_dir checkpoints/stage3_uncertainty
```

## 💾 显存优化策略

### 如果显存不足（<16GB）

#### 方案 1：减小 batch size + 梯度累积

```bash
python examples/train_defect_model.py \
    --backbone dinov3_vitb14 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    # 等效 batch_size=32
```

#### 方案 2：使用混合精度训练

```bash
python examples/train_defect_model.py \
    --backbone dinov3_vitb14 \
    --batch_size 32 \
    --mixed_precision \
    # 显存减少 ~40%
```

#### 方案 3：使用 DINOv3-S/14

```bash
python examples/train_defect_model.py \
    --backbone dinov3_vits14 \
    --batch_size 64 \
    # 只需 8GB 显存
```

### 如果显存充足（>32GB）

可以尝试 DINOv3-L/14：

```bash
python examples/train_defect_model.py \
    --backbone dinov3_vitl14 \
    --batch_size 16 \
    --epochs 100
```

## 📈 性能预期

### 基于 1w+ 样本的预期性能

| 指标 | 阶段 1（冻结） | 阶段 2（微调） | 阶段 3（优化） |
|------|--------------|--------------|--------------|
| **分类准确率** | 90-93% | 96-99% | 97-99% |
| **分割 IoU** | 0.70-0.75 | 0.80-0.90 | 0.82-0.92 |
| **分割 Dice** | 0.80-0.85 | 0.88-0.94 | 0.90-0.95 |
| **训练时间** | 3-4h | 10-12h | 3-4h |

*基于 V100 GPU，DINOv3-B/14*

### 不同模型的性能对比

| 模型 | 分类准确率 | 分割 IoU | 推理速度 |
|------|-----------|---------|---------|
| DINOv3-S/14 | 94-96% | 0.75-0.82 | 50 FPS |
| **DINOv3-B/14** ⭐ | **96-99%** | **0.80-0.90** | **30 FPS** |
| DINOv3-L/14 | 97-99% | 0.85-0.92 | 15 FPS |

## 🔧 超参数调优建议

### 学习率

```python
# 阶段 1（冻结骨干）
lr = 1e-3  # 可以用较大学习率

# 阶段 2（微调骨干）
lr = 5e-5  # 需要小心微调
# 如果不稳定，降到 1e-5

# 阶段 3（精细调整）
lr = 1e-5  # 非常小的学习率
```

### Batch Size

```python
# 根据显存选择
batch_size = 32  # 16GB 显存
batch_size = 16  # 12GB 显存
batch_size = 8   # 8GB 显存（需要梯度累积）

# 有效 batch size 建议：32-64
```

### 损失权重

```python
# 动态权重（推荐）
use_dynamic_weights = True
alpha_start = 0.5, alpha_end = 1.0
beta_start = 1.0, beta_end = 0.3

# 或固定权重
alpha = 1.0  # 分类
beta = 0.5   # 分割

# 如果分割效果差
beta = 1.0 或 2.0

# 如果分类效果差
alpha = 2.0
```

### 数据增强

对于 1w+ 样本，可以适当减少增强强度：

```python
# 在 defect_dataset.py 中调整
A.ShiftScaleRotate(
    shift_limit=0.05,  # 从 0.1 减小
    scale_limit=0.1,   # 从 0.2 减小
    rotate_limit=30,   # 从 45 减小
    p=0.3              # 从 0.5 减小
)
```

## 📊 训练监控

### 关键指标

1. **训练损失下降**：应该平滑下降
2. **验证准确率上升**：应该稳定上升
3. **分割 IoU 提升**：应该逐步提升
4. **两个任务平衡**：cls_loss 和 seg_loss 应该在同一数量级

### 异常情况处理

#### 情况 1：训练损失不下降

```bash
# 降低学习率
--lr 1e-5

# 检查数据
python examples/prepare_defect_data.py organize --check_data
```

#### 情况 2：验证准确率震荡

```bash
# 增加权重衰减
--weight_decay 0.1

# 使用余弦退火学习率
# 已默认使用
```

#### 情况 3：过拟合（训练准确率 >> 验证准确率）

```bash
# 增加 dropout
--dropout 0.3

# 增加数据增强
# 在 defect_dataset.py 中调整

# 增加权重衰减
--weight_decay 0.1
```

## 🎯 实战示例

### 完整训练脚本

```bash
#!/bin/bash

# 设置变量
DATA_ROOT="data/your_10k_dataset"
BACKBONE="dinov3_vitb14"

# 阶段 1：冻结骨干（快速验证）
echo "Stage 1: Training with frozen backbone..."
python examples/train_defect_model.py \
    --data_root $DATA_ROOT \
    --backbone $BACKBONE \
    --freeze_backbone \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-3 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage1

# 阶段 2：微调骨干（提升性能）
echo "Stage 2: Fine-tuning entire model..."
python examples/train_defect_model.py \
    --data_root $DATA_ROOT \
    --backbone $BACKBONE \
    --batch_size 16 \
    --epochs 70 \
    --lr 5e-5 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage2

# 阶段 3：不确定性加权（可选）
echo "Stage 3: Uncertainty weighting..."
python examples/train_defect_model.py \
    --data_root $DATA_ROOT \
    --backbone $BACKBONE \
    --batch_size 16 \
    --epochs 20 \
    --lr 1e-5 \
    --use_uncertainty_weighting \
    --save_dir checkpoints/stage3

# 推理测试
echo "Running inference..."
python examples/inference_defect_model.py \
    --checkpoint checkpoints/stage3/best_accuracy.pth \
    --backbone $BACKBONE \
    --image_dir data/test_images \
    --output_dir results

echo "Training completed!"
```

### Windows 版本

```batch
@echo off

REM 设置变量
set DATA_ROOT=data/your_10k_dataset
set BACKBONE=dinov3_vitb14

REM 阶段 1
echo Stage 1: Training with frozen backbone...
python examples/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --freeze_backbone --batch_size 32 --epochs 30 --lr 1e-3 --use_dynamic_weights --save_dir checkpoints/stage1

REM 阶段 2
echo Stage 2: Fine-tuning entire model...
python examples/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --batch_size 16 --epochs 70 --lr 5e-5 --use_dynamic_weights --save_dir checkpoints/stage2

REM 阶段 3
echo Stage 3: Uncertainty weighting...
python examples/train_defect_model.py --data_root %DATA_ROOT% --backbone %BACKBONE% --batch_size 16 --epochs 20 --lr 1e-5 --use_uncertainty_weighting --save_dir checkpoints/stage3

REM 推理
echo Running inference...
python examples/inference_defect_model.py --checkpoint checkpoints/stage3/best_accuracy.pth --backbone %BACKBONE% --image_dir data/test_images --output_dir results

echo Training completed!
pause
```

## 📝 总结

对于你的 1w+ 样本场景：

1. ✅ **使用 DINOv3-B/14**（94M 参数）
2. ✅ **三阶段训练**：冻结 → 微调 → 优化
3. ✅ **预期性能**：分类 96-99%，分割 IoU 0.80-0.90
4. ✅ **训练时间**：总共 ~15-20 小时（V100）
5. ✅ **显存需求**：16GB（batch_size=16）

**模型完全够用，性能会非常好！** 🚀
