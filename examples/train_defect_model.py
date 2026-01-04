"""
瑕疵检测模型训练示例
使用 DINOv3 作为骨干网络
"""

import torch
import argparse
from pathlib import Path

# 导入 DINOv3
import dinov3

# 导入自定义模块
from dinov3.models.defect_classifier import AttentionGuidedDefectClassifier
from dinov3.data.defect_dataset import create_dataloaders
from dinov3.train.train_defect_classifier import train_multitask_model


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载预训练的 DINOv3 模型
    print("Loading DINOv3 backbone...")
    
    # 根据配置选择不同大小的模型
    backbone_configs = {
        'dinov3_vits14': {'model': 'dinov3_vits14', 'embed_dim': 384},
        'dinov3_vitb14': {'model': 'dinov3_vitb14', 'embed_dim': 768},
        'dinov3_vitl14': {'model': 'dinov3_vitl14', 'embed_dim': 1024},
        'dinov3_vitg14': {'model': 'dinov3_vitg14', 'embed_dim': 1536},
    }
    
    config = backbone_configs[args.backbone]
    backbone = torch.hub.load('facebookresearch/dinov3', config['model'])
    print(f"Loaded {args.backbone} with embed_dim={config['embed_dim']}")
    
    # 冻结骨干网络（可选，初期训练时推荐）
    if args.freeze_backbone:
        print("Freezing backbone parameters...")
        for param in backbone.parameters():
            param.requires_grad = False
    
    # 创建多任务模型
    print("Creating multi-task model...")
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=config['embed_dim'],
        num_classes=2,
        seg_channels=1,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 创建数据加载器
    print(f"Loading data from {args.data_root}...")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 开始训练
    print("\nStarting training...")
    trained_model = train_multitask_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.save_dir,
        use_dynamic_weights=args.use_dynamic_weights,
        use_uncertainty_weighting=args.use_uncertainty_weighting
    )
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train defect detection model')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of dataset')
    parser.add_argument('--image_size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='dinov3_vitb14',
                        choices=['dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14'],
                        help='DINOv3 backbone model')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone parameters')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    
    # 损失权重策略
    parser.add_argument('--use_dynamic_weights', action='store_true',
                        help='Use dynamic weight scheduling')
    parser.add_argument('--use_uncertainty_weighting', action='store_true',
                        help='Use uncertainty-based weighting')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    main(args)
