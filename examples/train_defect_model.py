"""
瑕疵检测模型训练示例
使用 DINOv3 作为骨干网络
硬编码配置版本
"""

import torch

# 导入 DINOv3
import dinov3

# 导入自定义模块
from dinov3.models.defect_classifier import AttentionGuidedDefectClassifier
from dinov3.data.defect_dataset import create_dataloaders
from dinov3.train.train_defect_classifier import train_multitask_model


# ==================== 硬编码配置 ====================
class Config:
    """训练配置（硬编码）"""

    # 数据参数
    data_root = 'data_louye_cls'  # 数据集根目录
    image_size = 518               # 输入图像尺寸
    batch_size = 16                # 批次大小
    num_workers = 4                # 数据加载线程数

    # 模型参数
    backbone = 'dinov3_vitb14'     # 骨干网络: dinov3_vits14, dinov3_vitb14, dinov3_vitl14, dinov3_vitg14
    freeze_backbone = False        # 是否冻结骨干网络
    dropout = 0.2                  # Dropout率

    # 训练参数
    epochs = 100                   # 训练轮数
    lr = 1e-4                      # 学习率
    weight_decay = 0.05            # 权重衰减
    patience = 15                  # Early stopping patience (None 表示不使用)

    # 损失权重策略
    use_dynamic_weights = True     # 使用动态权重调度
    use_uncertainty_weighting = False  # 使用不确定性加权

    # 保存参数
    save_dir = 'checkpoints'       # 模型保存目录


def main():
    # 使用硬编码配置
    config = Config()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 打印配置
    print("\n" + "="*50)
    print("训练配置:")
    print("="*50)
    print(f"数据集路径: {config.data_root}")
    print(f"图像尺寸: {config.image_size}")
    print(f"批次大小: {config.batch_size}")
    print(f"骨干网络: {config.backbone}")
    print(f"冻结骨干: {config.freeze_backbone}")
    print(f"训练轮数: {config.epochs}")
    print(f"学习率: {config.lr}")
    print(f"Early Stopping: {'启用 (patience=' + str(config.patience) + ')' if config.patience else '禁用'}")
    print(f"动态权重: {config.use_dynamic_weights}")
    print(f"不确定性加权: {config.use_uncertainty_weighting}")
    print("="*50 + "\n")

    # 加载预训练的 DINOv3 模型
    print("Loading DINOv3 backbone...")

    # 根据配置选择不同大小的模型
    backbone_configs = {
        'dinov3_vits14': {'model': 'dinov3_vits14', 'embed_dim': 384},
        'dinov3_vitb14': {'model': 'dinov3_vitb14', 'embed_dim': 768},
        'dinov3_vitl14': {'model': 'dinov3_vitl14', 'embed_dim': 1024},
        'dinov3_vitg14': {'model': 'dinov3_vitg14', 'embed_dim': 1536},
    }

    backbone_config = backbone_configs[config.backbone]
    backbone = torch.hub.load('facebookresearch/dinov3', backbone_config['model'])
    print(f"Loaded {config.backbone} with embed_dim={backbone_config['embed_dim']}")

    # 冻结骨干网络（可选，初期训练时推荐）
    if config.freeze_backbone:
        print("Freezing backbone parameters...")
        for param in backbone.parameters():
            param.requires_grad = False

    # 创建多任务模型
    print("Creating multi-task model...")
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=backbone_config['embed_dim'],
        num_classes=2,
        seg_channels=1,
        dropout=config.dropout
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # 创建数据加载器
    print(f"Loading data from {config.data_root}...")
    train_loader, val_loader = create_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 开始训练
    print("\nStarting training...")
    trained_model = train_multitask_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=device,
        save_dir=config.save_dir,
        use_dynamic_weights=config.use_dynamic_weights,
        use_uncertainty_weighting=config.use_uncertainty_weighting,
        patience=config.patience
    )

    print("Training completed!")


if __name__ == '__main__':
    main()
