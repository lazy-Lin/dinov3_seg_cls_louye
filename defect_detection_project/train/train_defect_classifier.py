"""
瑕疵检测多任务模型训练脚本
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import mlflow

from defect_detection_project.models.defect_classifier import (
    AttentionGuidedDefectClassifier,
    MultiTaskLoss,
    DynamicWeightScheduler
)


def calculate_metrics(outputs, cls_labels, seg_masks, threshold=0.5):
    """计算分类和分割指标"""
    # 分类指标
    cls_logits = outputs['classification']
    cls_preds = torch.argmax(cls_logits, dim=1)
    cls_acc = (cls_preds == cls_labels).float().mean().item()
    
    # 分割指标（IoU）
    seg_logits = outputs['segmentation']
    seg_preds = (torch.sigmoid(seg_logits) > threshold).float()
    
    intersection = (seg_preds * seg_masks).sum(dim=[1, 2, 3])
    union = (seg_preds + seg_masks).clamp(0, 1).sum(dim=[1, 2, 3])
    iou = (intersection / (union + 1e-6)).mean().item()
    
    # Dice 系数
    dice = (2 * intersection / (seg_preds.sum(dim=[1, 2, 3]) + seg_masks.sum(dim=[1, 2, 3]) + 1e-6)).mean().item()
    
    return {
        'accuracy': cls_acc,
        'iou': iou,
        'dice': dice
    }


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    total_metrics = {'accuracy': 0, 'iou': 0, 'dice': 0}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        cls_labels = batch['label'].to(device)
        seg_masks = batch['mask'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, cls_labels, seg_masks)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # 统计
        total_loss += losses['total_loss'].item()
        total_cls_loss += losses['cls_loss'].item()
        total_seg_loss += losses['seg_loss'].item()
        
        # 计算指标
        with torch.no_grad():
            metrics = calculate_metrics(outputs, cls_labels, seg_masks)
            for k, v in metrics.items():
                total_metrics[k] += v
        
        # 更新进度条
        pbar.set_postfix({
            'loss': losses['total_loss'].item(),
            'cls_loss': losses['cls_loss'].item(),
            'seg_loss': losses['seg_loss'].item(),
            'acc': metrics['accuracy']
        })
    
    num_batches = len(train_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'accuracy': total_metrics['accuracy'] / num_batches,
        'iou': total_metrics['iou'] / num_batches,
        'dice': total_metrics['dice'] / num_batches
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    
    total_loss = 0
    total_cls_loss = 0
    total_seg_loss = 0
    total_metrics = {'accuracy': 0, 'iou': 0, 'dice': 0}
    
    for batch in tqdm(val_loader, desc='Validating'):
        images = batch['image'].to(device)
        cls_labels = batch['label'].to(device)
        seg_masks = batch['mask'].to(device)
        
        # 前向传播
        outputs = model(images)
        
        # 计算损失
        losses = criterion(outputs, cls_labels, seg_masks)
        
        total_loss += losses['total_loss'].item()
        total_cls_loss += losses['cls_loss'].item()
        total_seg_loss += losses['seg_loss'].item()
        
        # 计算指标
        metrics = calculate_metrics(outputs, cls_labels, seg_masks)
        for k, v in metrics.items():
            total_metrics[k] += v
    
    num_batches = len(val_loader)
    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'seg_loss': total_seg_loss / num_batches,
        'accuracy': total_metrics['accuracy'] / num_batches,
        'iou': total_metrics['iou'] / num_batches,
        'dice': total_metrics['dice'] / num_batches
    }


def train_multitask_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-4,
    weight_decay=0.05,
    device='cuda',
    save_dir='checkpoints',
    use_dynamic_weights=True,
    use_uncertainty_weighting=False,
    use_mlflow=False
):
    """
    多任务模型训练主函数
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        device: 设备
        save_dir: 模型保存目录
        use_dynamic_weights: 是否使用动态权重调整
        use_uncertainty_weighting: 是否使用不确定性加权
        use_mlflow: 是否使用 mlflow 记录
    """
    model = model.to(device)
    
    # 多卡支持
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=lr * 0.01
    )
    
    # 损失函数
    if use_dynamic_weights:
        weight_scheduler = DynamicWeightScheduler(
            alpha_start=0.5, alpha_end=1.0,
            beta_start=1.0, beta_end=0.3,
            warmup_epochs=20,
            total_epochs=epochs
        )
        criterion = MultiTaskLoss(
            alpha=0.5, beta=1.0,
            use_uncertainty_weighting=use_uncertainty_weighting
        )
    else:
        criterion = MultiTaskLoss(
            alpha=1.0, beta=0.5,
            use_uncertainty_weighting=use_uncertainty_weighting
        )
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_acc = 0
    best_val_iou = 0
    
    for epoch in range(1, epochs + 1):
        # 动态调整权重
        if use_dynamic_weights:
            alpha, beta = weight_scheduler.get_weights(epoch)
            criterion.alpha = alpha
            criterion.beta = beta
            print(f"\nEpoch {epoch}: alpha={alpha:.3f}, beta={beta:.3f}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        # 打印结果
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}")
        
        # MLflow logging
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_metrics['loss'],
                "train_cls_loss": train_metrics['cls_loss'],
                "train_seg_loss": train_metrics['seg_loss'],
                "train_acc": train_metrics['accuracy'],
                "train_iou": train_metrics['iou'],
                "train_dice": train_metrics['dice'],
                "val_loss": val_metrics['loss'],
                "val_cls_loss": val_metrics['cls_loss'],
                "val_seg_loss": val_metrics['seg_loss'],
                "val_acc": val_metrics['accuracy'],
                "val_iou": val_metrics['iou'],
                "val_dice": val_metrics['dice'],
                "learning_rate": optimizer.param_groups[0]['lr'],
                "loss_alpha": criterion.alpha,
                "loss_beta": criterion.beta
            }, step=epoch)

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            # 处理 DataParallel 的 state_dict
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, save_dir / 'best_accuracy.pth')
            print(f"✓ Saved best accuracy model: {best_val_acc:.4f}")
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            # 处理 DataParallel 的 state_dict
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, save_dir / 'best_iou.pth')
            print(f"✓ Saved best IoU model: {best_val_iou:.4f}")
        
        # 保存最后一个 epoch 的模型 (覆盖更新)
        # 处理 DataParallel 的 state_dict
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics
        }, save_dir / 'last.pth')
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    
    return model
