"""
快速开始示例
演示如何使用已创建的核心模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import AttentionGuidedDefectClassifier, MultiTaskLoss, DynamicWeightScheduler
from data import DefectDataset, create_dataloaders


def example_1_create_model():
    """示例 1：创建模型"""
    print("=" * 50)
    print("示例 1：创建模型")
    print("=" * 50)
    
    # 加载 DINOv3 骨干网络
    print("加载 DINOv3-B/14 骨干网络...")
    backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14')
    
    # 创建瑕疵检测模型
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=768,  # ViT-B: 768
        num_classes=2,
        seg_channels=1,
        dropout=0.2
    )
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ 模型创建成功！")
    print(f"  总参数量: {total_params / 1e6:.2f}M")
    print(f"  可训练参数: {trainable_params / 1e6:.2f}M")
    print()
    
    return model


def example_2_create_loss():
    """示例 2：创建损失函数"""
    print("=" * 50)
    print("示例 2：创建损失函数")
    print("=" * 50)
    
    # 固定权重损失
    criterion_fixed = MultiTaskLoss(alpha=1.0, beta=0.5)
    print("✓ 固定权重损失函数创建成功")
    print(f"  分类权重 (alpha): 1.0")
    print(f"  分割权重 (beta): 0.5")
    
    # 不确定性加权损失
    criterion_uncertainty = MultiTaskLoss(
        alpha=1.0, 
        beta=0.5,
        use_uncertainty_weighting=True
    )
    print("✓ 不确定性加权损失函数创建成功")
    
    # 动态权重调度器
    scheduler = DynamicWeightScheduler(
        alpha_start=0.5,
        alpha_end=1.0,
        beta_start=1.0,
        beta_end=0.3,
        warmup_epochs=20
    )
    print("✓ 动态权重调度器创建成功")
    print(f"  初始权重: alpha=0.5, beta=1.0")
    print(f"  最终权重: alpha=1.0, beta=0.3")
    print()
    
    return criterion_fixed, scheduler


def example_3_create_dataset():
    """示例 3：创建数据集（需要准备数据）"""
    print("=" * 50)
    print("示例 3：创建数据集")
    print("=" * 50)
    
    # 数据路径（需要先准备数据）
    data_root = "../data/my_dataset"  # 修改为你的数据路径
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            data_root=data_root,
            batch_size=16,
            num_workers=4,
            image_size=518
        )
        
        print(f"✓ 数据加载器创建成功！")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print()
        
        return train_loader, val_loader
        
    except FileNotFoundError:
        print(f"⚠ 数据目录不存在: {data_root}")
        print(f"  请先准备数据，或使用 scripts/prepare_data.py")
        print()
        return None, None


def example_4_forward_pass():
    """示例 4：前向传播"""
    print("=" * 50)
    print("示例 4：前向传播测试")
    print("=" * 50)
    
    # 创建模型
    print("创建模型...")
    backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14')
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=768,
        num_classes=2
    )
    model.eval()
    
    # 创建测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 518, 518)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(x, return_attention=True)
    
    print(f"✓ 前向传播成功！")
    print(f"  分类输出形状: {outputs['classification'].shape}")
    print(f"  分割输出形状: {outputs['segmentation'].shape}")
    print(f"  注意力图形状: {outputs['attention_map'].shape}")
    print()


def example_5_training_loop():
    """示例 5：简单的训练循环"""
    print("=" * 50)
    print("示例 5：训练循环示例")
    print("=" * 50)
    
    print("""
简单的训练循环示例代码：

```python
import torch
from models import AttentionGuidedDefectClassifier, MultiTaskLoss
from data import create_dataloaders

# 1. 创建模型
backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb14')
model = AttentionGuidedDefectClassifier(backbone=backbone, embed_dim=768)
model = model.cuda()

# 2. 创建数据加载器
train_loader, val_loader = create_dataloaders(
    data_root='data/my_dataset',
    batch_size=16
)

# 3. 创建优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = MultiTaskLoss(alpha=1.0, beta=0.5)

# 4. 训练循环
for epoch in range(100):
    model.train()
    for batch in train_loader:
        images = batch['image'].cuda()
        labels = batch['label'].cuda()
        masks = batch['mask'].cuda()
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, labels, masks)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {losses['total_loss'].item():.4f}")
```

提示：完整的训练代码请参考根目录的 dinov3/train/train_defect_classifier.py
或从 examples/train_defect_model.py 复制到 scripts/train.py
    """)
    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 50)
    print("瑕疵检测模型 - 快速开始示例")
    print("=" * 50 + "\n")
    
    # 示例 1：创建模型
    model = example_1_create_model()
    
    # 示例 2：创建损失函数
    criterion, scheduler = example_2_create_loss()
    
    # 示例 3：创建数据集（可选，需要数据）
    train_loader, val_loader = example_3_create_dataset()
    
    # 示例 4：前向传播
    example_4_forward_pass()
    
    # 示例 5：训练循环示例
    example_5_training_loop()
    
    print("=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)
    print()
    print("下一步：")
    print("1. 准备数据：python scripts/prepare_data.py")
    print("2. 训练模型：python scripts/train.py")
    print("3. 推理测试：python scripts/inference.py")
    print()
    print("或查看文档：")
    print("- README.md - 项目概览")
    print("- SETUP_GUIDE.md - 设置指南")
    print("- FILE_MANIFEST.md - 文件清单")
    print()


if __name__ == '__main__':
    main()
