"""
瑕疵检测模型训练示例
使用 DINOv3 作为骨干网络
"""

import torch
import sys
import os
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便导入 defect_detection_project
current_dir = os.path.dirname(os.path.abspath(__file__))
# scripts -> defect_detection_project -> root
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import yaml

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
    # 硬编码参数配置，替代 argparse
    class Args:
        # 配置文件路径
        config = 'defect_detection_project/configs/config_defect_detection.yaml'
        
        # 数据集根目录 (None 表示使用配置文件中的值)
        data_root = None
        # data_root = r"C:\Users\Admin\Desktop\dinov3_seg_cls\defect_detection_project\data"
        
        # 是否冻结骨干网络
        freeze_backbone = False
        
        # MLflow 设置
        enable_mlflow = False  # 强制开启
        no_mlflow = True       # 强制关闭 (优先级更高)
        mlflow_tracking_uri = None
        mlflow_experiment_name = None

    args = Args()
    
    # 打印当前配置
    print("Running with hardcoded arguments:")
    print(f"  config: {args.config}")
    print(f"  data_root: {args.data_root}")
    print(f"  freeze_backbone: {args.freeze_backbone}")
    print(f"  mlflow: enable={args.enable_mlflow}, disable={args.no_mlflow}")
    
    main(args)
