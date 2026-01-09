"""
瑕疵检测模型训练示例
使用 DINOv3 作为骨干网络
"""

import torch
import argparse
from pathlib import Path
import sys
import os
import mlflow
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 导入 DINOv3
import dinov3

# 导入自定义模块
from defect_detection_project.models.defect_classifier import AttentionGuidedDefectClassifier
from defect_detection_project.data.defect_dataset import create_dataloaders
from defect_detection_project.train.train_defect_classifier import train_multitask_model


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        # 尝试在默认路径查找
        default_config = Path(project_root) / 'defect_detection_project' / 'configs' / 'config_defect_detection.yaml'
        if default_config.exists():
            print(f"Config not found at {config_path}, using default: {default_config}")
            config_path = default_config
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
    config = load_config(config_path)
    
    # 优先使用命令行参数覆盖配置
    data_root = args.data_root if args.data_root else config['data']['data_root']
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载预训练的 DINOv3 模型
    backbone_name = config['model']['backbone']
    print(f"Loading DINOv3 backbone: {backbone_name}...")
    # 使用本地 hubconf 加载模型，确保使用本地缓存的权重
    backbone = torch.hub.load(project_root, backbone_name, source='local', pretrained=True)
    
    # 冻结骨干网络
    freeze_backbone = args.freeze_backbone or config['model']['freeze_backbone']
    if freeze_backbone:
        print("Freezing backbone parameters...")
        for param in backbone.parameters():
            param.requires_grad = False
    
    # 创建多任务模型
    print("Creating multi-task model...")
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=config['model']['embed_dim'],
        num_classes=config['model']['num_classes'],
        seg_channels=config['model']['seg_channels'],
        dropout=config['model']['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 创建数据加载器
    print(f"Loading data from {data_root}...")
    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 配置 MLflow
    mlflow_config = config['logging']['mlflow']
    enable_mlflow = args.enable_mlflow or mlflow_config['enabled']
    
    if enable_mlflow:
        tracking_uri = args.mlflow_tracking_uri if args.mlflow_tracking_uri else mlflow_config['tracking_uri']
        experiment_name = args.mlflow_experiment_name if args.mlflow_experiment_name else mlflow_config['experiment_name']
        
        print(f"Configuring MLflow with URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        
        # 记录超参数
        mlflow.log_params({
            "backbone": backbone_name,
            "embed_dim": config['model']['embed_dim'],
            "batch_size": config['data']['batch_size'],
            "epochs": config['training']['epochs'],
            "lr": config['training']['lr'],
            "weight_decay": config['training']['weight_decay'],
            "image_size": config['data']['image_size'],
            "freeze_backbone": freeze_backbone,
            "use_dynamic_weights": config['training']['use_dynamic_weights'],
            "use_uncertainty_weighting": config['training']['use_uncertainty_weighting']
        })

    # 开始训练
    print("\nStarting training...")
    try:
        trained_model = train_multitask_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay']),
            device=device,
            save_dir=config['checkpoint']['save_dir'],
            use_dynamic_weights=config['training']['use_dynamic_weights'],
            use_uncertainty_weighting=config['training']['use_uncertainty_weighting'],
            use_mlflow=enable_mlflow
        )
    finally:
        if enable_mlflow:
            mlflow.end_run()
    
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train defect detection model')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='defect_detection_project/configs/config_defect_detection.yaml',
                        help='Path to config file')
    
    # 覆盖参数（可选）
    parser.add_argument('--data_root', type=str, help='Override data root')
    parser.add_argument('--freeze_backbone', action='store_true', help='Override freeze backbone')
    
    # MLflow 覆盖参数
    parser.add_argument('--enable_mlflow', action='store_true', help='Force enable MLflow')
    parser.add_argument('--mlflow_tracking_uri', type=str, help='Override MLflow tracking URI')
    parser.add_argument('--mlflow_experiment_name', type=str, help='Override MLflow experiment name')

    args = parser.parse_args()
    main(args)
