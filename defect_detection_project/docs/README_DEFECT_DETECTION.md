# 瑕疵检测多任务模型使用指南

基于 DINOv3 的瑕疵检测模型，结合分割分支引导分类任务，提升对瑕疵特征的学习能力。

## 模型架构

### 核心设计
- **共享骨干网络**: DINOv3 (ViT-S/14)
- **分割分支**: 轻量级解码器，生成瑕疵区域掩码
- **注意力引导**: 使用分割掩码引导分类器聚焦瑕疵区域
- **多任务学习**: 同时优化分类和分割任务

### 优势
1. **精准定位**: 分割分支帮助模型学习瑕疵的精确位置
2. **特征聚焦**: 注意力机制引导分类器关注关键区域
3. **鲁棒性强**: 多任务学习提升模型泛化能力
4. **可解释性**: 可视化注意力图和分割结果

## 数据准备

### 数据组织结构

```
data_root/
├── images/
│   ├── defect_001.jpg      # 带瑕疵的图片
│   ├── defect_002.jpg
│   ├── normal_001.jpg      # 正常图片
│   └── ...
├── masks/
│   ├── defect_001.png      # 瑕疵区域掩码（二值图）
│   ├── defect_002.png
│   └── ...                 # 正常图片无需 mask
├── train_labels.txt        # 训练集标签
└── val_labels.txt          # 验证集标签
```

### 标签文件格式

`train_labels.txt` 和 `val_labels.txt` 格式：
```
defect_001.jpg,1
defect_002.jpg,1
normal_001.jpg,0
normal_002.jpg,0
```

- 第一列：图片文件名
- 第二列：标签（0=正常，1=瑕疵）

### 掩码要求

- 格式：PNG 或 JPG
- 类型：灰度图或二值图
- 值：瑕疵区域为白色（255），背景为黑色（0）
- 尺寸：与原图相同或任意尺寸（会自动 resize）

## 训练

### 基础训练

```bash
python examples/train_defect_model.py \
    --data_root /path/to/your/data \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir checkpoints/defect_model
```

### 推荐配置

#### 1. 初期训练（冻结骨干网络）

适合数据量较小的情况：

```bash
python examples/train_defect_model.py \
    --data_root /path/to/your/data \
    --freeze_backbone \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage1
```

#### 2. 微调阶段（解冻骨干网络）

在初期训练基础上继续训练：

```bash
python examples/train_defect_model.py \
    --data_root /path/to/your/data \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --use_dynamic_weights \
    --save_dir checkpoints/stage2
```

#### 3. 使用不确定性加权

自动平衡分类和分割损失：

```bash
python examples/train_defect_model.py \
    --data_root /path/to/your/data \
    --batch_size 32 \
    --epochs 100 \
    --use_uncertainty_weighting \
    --save_dir checkpoints/uncertainty
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data_root` | 数据根目录 | 必需 |
| `--image_size` | 输入图像尺寸 | 518 |
| `--batch_size` | 批次大小 | 32 |
| `--epochs` | 训练轮数 | 100 |
| `--lr` | 学习率 | 1e-4 |
| `--weight_decay` | 权重衰减 | 0.05 |
| `--freeze_backbone` | 冻结骨干网络 | False |
| `--dropout` | Dropout 率 | 0.2 |
| `--use_dynamic_weights` | 动态权重调整 | False |
| `--use_uncertainty_weighting` | 不确定性加权 | False |
| `--save_dir` | 模型保存目录 | checkpoints |

## 推理

### 单张图像推理

```bash
python examples/inference_defect_model.py \
    --checkpoint checkpoints/defect_model/best_accuracy.pth \
    --image /path/to/test_image.jpg \
    --output_dir results
```

### 批量推理

```bash
python examples/inference_defect_model.py \
    --checkpoint checkpoints/defect_model/best_accuracy.pth \
    --image_dir /path/to/test_images \
    --output_dir results
```

### 输出结果

推理会生成：
1. **可视化图像**: 包含原图、分割掩码、注意力图和分类结果
2. **结果摘要**: `results_summary.txt` 包含所有图像的预测结果

## 训练策略建议

### 1. 动态权重调整

使用 `--use_dynamic_weights` 启用：

- **初期（0-20 epochs）**: 分割权重较大（β=1.0），分类权重较小（α=0.5）
  - 帮助模型先学习瑕疵的位置和形状
- **后期（20+ epochs）**: 分类权重增大（α=1.0），分割权重减小（β=0.3）
  - 提升分类性能

### 2. 不确定性加权

使用 `--use_uncertainty_weighting` 启用：

- 自动学习两个任务的相对重要性
- 适合不确定最优权重比例的情况

### 3. 渐进式训练

**阶段一**: 冻结骨干网络，只训练分类和分割头
```bash
--freeze_backbone --epochs 30 --lr 1e-3
```

**阶段二**: 解冻骨干网络，端到端微调
```bash
--epochs 70 --lr 1e-4
```

## 性能优化建议

### 数据增强

模型已内置以下增强策略：
- 几何变换：翻转、旋转、缩放
- 颜色变换：亮度、对比度、色调
- 噪声：高斯噪声、模糊

### 类别不平衡

如果正常样本远多于瑕疵样本，可以：
1. 使用 Focal Loss（在 `MultiTaskLoss` 中设置 `focal_loss=True`）
2. 调整采样策略
3. 增加瑕疵样本的数据增强

### 弱监督方案

如果没有精确的分割标注：
1. 使用边界框转换为粗略掩码
2. 使用 DINOv3 的注意力图作为伪标签
3. 使用 CAM/Grad-CAM 生成掩码

## 评估指标

训练过程会输出：
- **分类指标**: Accuracy
- **分割指标**: IoU, Dice
- **损失**: Total Loss, Classification Loss, Segmentation Loss

## 常见问题

### Q1: 分割效果好但分类效果差？
- 增大分类损失权重（α）
- 检查分类头的容量是否足够
- 尝试使用 Focal Loss

### Q2: 分类效果好但分割效果差？
- 增大分割损失权重（β）
- 检查掩码标注质量
- 增加分割解码器的深度

### Q3: 两个任务都不收敛？
- 降低学习率
- 使用渐进式训练
- 检查数据质量和标注一致性

### Q4: 显存不足？
- 减小 batch_size
- 减小 image_size
- 使用梯度累积

## 代码结构

```
dinov3/
├── models/
│   └── defect_classifier.py          # 模型定义
├── data/
│   └── defect_dataset.py              # 数据集
└── train/
    └── train_defect_classifier.py     # 训练逻辑

examples/
├── train_defect_model.py              # 训练脚本
├── inference_defect_model.py          # 推理脚本
└── README_DEFECT_DETECTION.md         # 本文档
```

## 引用

如果使用本代码，请引用 DINOv3：

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
```
