"""
测试模型架构和数据流
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from defect_detection_project.models.defect_classifier import (
    AttentionGuidedDefectClassifier,
    MultiTaskLoss,
    DynamicWeightScheduler
)


def test_model_forward():
    """测试模型前向传播"""
    print("=" * 50)
    print("测试 1: 模型前向传播")
    print("=" * 50)
    
    # 创建简单的骨干网络（用于测试）
    class DummyBackbone(torch.nn.Module):
        def __init__(self, embed_dim=384):
            super().__init__()
            self.embed_dim = embed_dim
            
        def forward_features(self, x):
            B, C, H, W = x.shape
            num_patches = (H // 14) * (W // 14)  # 假设 patch size = 14
            
            return {
                'x_norm_clstoken': torch.randn(B, self.embed_dim),
                'x_norm_patchtokens': torch.randn(B, num_patches, self.embed_dim)
            }
    
    # 创建模型
    backbone = DummyBackbone(embed_dim=384)
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=384,
        num_classes=2,
        seg_channels=1
    )
    
    # 测试输入
    batch_size = 4
    x = torch.randn(batch_size, 3, 518, 518)
    
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_attention=True)
    
    print(f"分类输出形状: {outputs['classification'].shape}")
    print(f"分割输出形状: {outputs['segmentation'].shape}")
    print(f"注意力图形状: {outputs['attention_map'].shape}")
    
    # 验证形状
    assert outputs['classification'].shape == (batch_size, 2)
    assert outputs['segmentation'].shape == (batch_size, 1, 518, 518)
    
    print("✓ 模型前向传播测试通过！\n")


def test_loss_function():
    """测试损失函数"""
    print("=" * 50)
    print("测试 2: 损失函数")
    print("=" * 50)
    
    batch_size = 4
    
    # 模拟输出
    outputs = {
        'classification': torch.randn(batch_size, 2),
        'segmentation': torch.randn(batch_size, 1, 518, 518)
    }
    
    # 模拟标签
    cls_labels = torch.randint(0, 2, (batch_size,))
    seg_masks = torch.rand(batch_size, 1, 518, 518)
    
    # 测试固定权重
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5)
    losses = criterion(outputs, cls_labels, seg_masks)
    
    print(f"总损失: {losses['total_loss'].item():.4f}")
    print(f"分类损失: {losses['cls_loss'].item():.4f}")
    print(f"分割损失: {losses['seg_loss'].item():.4f}")
    
    assert 'total_loss' in losses
    assert 'cls_loss' in losses
    assert 'seg_loss' in losses
    
    print("✓ 固定权重损失测试通过！")
    
    # 测试不确定性加权
    criterion_unc = MultiTaskLoss(
        alpha=1.0, 
        beta=0.5, 
        use_uncertainty_weighting=True
    )
    losses_unc = criterion_unc(outputs, cls_labels, seg_masks)
    
    print(f"\n不确定性加权总损失: {losses_unc['total_loss'].item():.4f}")
    print(f"log_var_cls: {criterion_unc.log_var_cls.item():.4f}")
    print(f"log_var_seg: {criterion_unc.log_var_seg.item():.4f}")
    
    print("✓ 不确定性加权损失测试通过！\n")


def test_weight_scheduler():
    """测试权重调度器"""
    print("=" * 50)
    print("测试 3: 动态权重调度器")
    print("=" * 50)
    
    scheduler = DynamicWeightScheduler(
        alpha_start=0.5,
        alpha_end=1.0,
        beta_start=1.0,
        beta_end=0.3,
        warmup_epochs=20,
        total_epochs=100
    )
    
    print("Epoch | Alpha | Beta")
    print("-" * 30)
    
    test_epochs = [0, 5, 10, 15, 20, 30, 50, 100]
    for epoch in test_epochs:
        alpha, beta = scheduler.get_weights(epoch)
        print(f"{epoch:5d} | {alpha:.3f} | {beta:.3f}")
    
    # 验证边界值
    alpha_0, beta_0 = scheduler.get_weights(0)
    assert abs(alpha_0 - 0.5) < 0.01
    assert abs(beta_0 - 1.0) < 0.01
    
    alpha_20, beta_20 = scheduler.get_weights(20)
    assert abs(alpha_20 - 1.0) < 0.01
    assert abs(beta_20 - 0.3) < 0.01
    
    print("\n✓ 权重调度器测试通过！\n")


def test_model_parameters():
    """测试模型参数量"""
    print("=" * 50)
    print("测试 4: 模型参数统计")
    print("=" * 50)
    
    class DummyBackbone(torch.nn.Module):
        def __init__(self, embed_dim=384):
            super().__init__()
            self.embed_dim = embed_dim
            self.dummy_param = torch.nn.Parameter(torch.randn(1000, embed_dim))
            
        def forward_features(self, x):
            B, C, H, W = x.shape
            num_patches = (H // 14) * (W // 14)
            return {
                'x_norm_clstoken': torch.randn(B, self.embed_dim),
                'x_norm_patchtokens': torch.randn(B, num_patches, self.embed_dim)
            }
    
    backbone = DummyBackbone(embed_dim=384)
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=384,
        num_classes=2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 冻结骨干网络
    for param in backbone.parameters():
        param.requires_grad = False
    
    trainable_params_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"冻结骨干后可训练参数: {trainable_params_frozen / 1e6:.2f}M")
    
    print("\n✓ 参数统计测试通过！\n")


def test_gradient_flow():
    """测试梯度流"""
    print("=" * 50)
    print("测试 5: 梯度流")
    print("=" * 50)
    
    class DummyBackbone(torch.nn.Module):
        def __init__(self, embed_dim=384):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv = torch.nn.Conv2d(3, embed_dim, 1)
            
        def forward_features(self, x):
            B, C, H, W = x.shape
            num_patches = (H // 14) * (W // 14)
            return {
                'x_norm_clstoken': torch.randn(B, self.embed_dim, requires_grad=True),
                'x_norm_patchtokens': torch.randn(B, num_patches, self.embed_dim, requires_grad=True)
            }
    
    backbone = DummyBackbone(embed_dim=384)
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=384,
        num_classes=2
    )
    
    # 前向传播
    x = torch.randn(2, 3, 518, 518, requires_grad=True)
    outputs = model(x)
    
    # 模拟损失
    cls_labels = torch.tensor([0, 1])
    seg_masks = torch.rand(2, 1, 518, 518)
    
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5)
    losses = criterion(outputs, cls_labels, seg_masks)
    
    # 反向传播
    losses['total_loss'].backward()
    
    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            print(f"✓ {name}: 梯度正常")
            break
    
    assert has_grad, "没有检测到梯度！"
    
    print("\n✓ 梯度流测试通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始测试瑕疵检测模型")
    print("=" * 50 + "\n")
    
    try:
        test_model_forward()
        test_loss_function()
        test_weight_scheduler()
        test_model_parameters()
        test_gradient_flow()
        
        print("=" * 50)
        print("✓ 所有测试通过！")
        print("=" * 50)
        print("\n模型架构验证成功，可以开始训练！\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
