# 项目设置指南

## 📁 完整项目结构

本项目是一个**独立的瑕疵检测项目**，不修改 DINOv3 原始代码。

```
defect_detection_project/          # 独立项目目录
├── models/                         # 模型定义
│   ├── __init__.py
│   └── defect_classifier.py       # 注意力引导分类器
│
├── data/                           # 数据处理
│   ├── __init__.py
│   └── dataset.py                 # 数据集类
│
├── train/                          # 训练逻辑
│   ├── __init__.py
│   └── trainer.py                 # 训练器
│
├── scripts/                        # 可执行脚本
│   ├── train.py                   # 训练脚本
│   ├── inference.py               # 推理脚本
│   ├── prepare_data.py            # 数据准备
│   ├── test_model.py              # 模型测试
│   ├── train_large_dataset.sh     # Linux 训练脚本
│   ├── train_large_dataset.bat    # Windows 训练脚本
│   ├── quick_start.sh             # Linux 快速开始
│   └── quick_start.bat            # Windows 快速开始
│
├── docs/                           # 文档
│   ├── QUICK_REFERENCE.md         # 快速参考
│   ├── MODEL_SIZE_GUIDE.md        # 模型参数说明
│   ├── IMPLEMENTATION_GUIDE.md    # 实现指南
│   └── LARGE_DATASET_GUIDE.md     # 大数据集指南
│
├── config/                         # 配置文件
│   └── config.yaml                # 配置示例
│
├── requirements.txt                # Python 依赖
├── README.md                       # 项目说明
└── SETUP_GUIDE.md                 # 本文档
```

## 🚀 快速设置

### 1. 安装依赖

```bash
cd defect_detection_project
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python scripts/test_model.py
```

### 3. 准备数据

```bash
python scripts/prepare_data.py organize \
    --defect_images /path/to/defect/images \
    --defect_masks /path/to/defect/masks \
    --normal_images /path/to/normal/images \
    --output_dir ../data/my_dataset
```

### 4. 开始训练

```bash
# Windows
scripts\train_large_dataset.bat

# Linux/Mac
bash scripts/train_large_dataset.sh
```

## 📝 与 DINOv3 的关系

本项目**依赖但不修改** DINOv3：

1. **使用 DINOv3 作为骨干网络**：
   ```python
   import torch
   backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14')
   ```

2. **DINOv3 原始代码保持不变**：
   - `dinov3/` 目录中的所有文件保持原样
   - 不需要修改任何 DINOv3 的代码

3. **独立的项目结构**：
   - 所有瑕疵检测相关代码在 `defect_detection_project/` 目录
   - 可以独立开发、测试和部署

## 🔧 导入方式

在项目中使用模型：

```python
# 方式 1：直接导入（推荐）
import sys
sys.path.append('defect_detection_project')

from models import AttentionGuidedDefectClassifier, MultiTaskLoss
from data import DefectDataset
from train import train_multitask_model

# 方式 2：使用相对导入
from defect_detection_project.models import AttentionGuidedDefectClassifier
```

## 📦 依赖说明

### 核心依赖

- **torch >= 2.0.0**: PyTorch 框架
- **torchvision >= 0.15.0**: 视觉工具
- **DINOv3**: 通过 torch.hub 自动下载

### 数据处理

- **albumentations >= 1.3.0**: 数据增强
- **opencv-python >= 4.7.0**: 图像处理
- **pillow >= 9.5.0**: 图像读取

### 训练工具

- **tqdm >= 4.65.0**: 进度条
- **scikit-learn >= 1.2.0**: 数据划分

### 可视化

- **matplotlib >= 3.7.0**: 结果可视化

## 🎯 使用场景

### 场景 1：快速原型（小数据集）

```bash
python scripts/train.py \
    --data_root ../data/demo \
    --backbone dinov3_vits14 \
    --batch_size 32 \
    --epochs 30 \
    --freeze_backbone
```

### 场景 2：生产环境（大数据集）

```bash
# 使用提供的脚本
bash scripts/train_large_dataset.sh
```

### 场景 3：自定义训练

```python
import torch
from models import AttentionGuidedDefectClassifier, MultiTaskLoss
from data import create_dataloaders
from train import train_multitask_model

# 加载骨干网络
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14')

# 创建模型
model = AttentionGuidedDefectClassifier(
    backbone=backbone,
    embed_dim=768,
    num_classes=2
)

# 创建数据加载器
train_loader, val_loader = create_dataloaders(
    data_root='../data/my_dataset',
    batch_size=16
)

# 训练
trained_model = train_multitask_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

## 🔍 项目特点

### 1. 独立性
- ✅ 不修改 DINOv3 原始代码
- ✅ 可以独立版本控制
- ✅ 易于维护和更新

### 2. 模块化
- ✅ 清晰的目录结构
- ✅ 独立的模型、数据、训练模块
- ✅ 易于扩展和定制

### 3. 完整性
- ✅ 包含所有必要的代码
- ✅ 详细的文档和示例
- ✅ 开箱即用的脚本

## 📚 文档导航

1. **[README.md](README.md)** - 项目概览和快速开始
2. **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - 常用命令参考
3. **[docs/MODEL_SIZE_GUIDE.md](docs/MODEL_SIZE_GUIDE.md)** - 模型选择指南
4. **[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md)** - 实现细节
5. **[docs/LARGE_DATASET_GUIDE.md](docs/LARGE_DATASET_GUIDE.md)** - 大数据集训练

## 🐛 常见问题

### Q: 为什么要独立项目？
A: 保持 DINOv3 原始代码不变，便于维护和更新。

### Q: 如何更新 DINOv3？
A: 直接更新 DINOv3 仓库，不影响本项目。

### Q: 可以在其他项目中使用吗？
A: 可以！只需复制 `defect_detection_project/` 目录即可。

### Q: 如何添加新功能？
A: 在 `defect_detection_project/` 目录中添加新模块，不影响 DINOv3。

## 📧 获取帮助

1. 查看文档：`docs/` 目录
2. 运行测试：`python scripts/test_model.py`
3. 查看示例：`scripts/` 目录中的脚本

---

**开始使用独立的瑕疵检测项目！** 🚀
