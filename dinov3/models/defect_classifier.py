"""
基于 DINOv3 的瑕疵检测多任务模型
结合分割分支引导分类任务，提升对瑕疵特征的学习能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGuidedDefectClassifier(nn.Module):
    """
    注意力引导的瑕疵分类器
    - 使用分割分支生成注意力图
    - 引导分类器聚焦于瑕疵区域
    """
    
    def __init__(
        self, 
        backbone, 
        embed_dim=768, 
        num_classes=2,
        seg_channels=1,
        dropout=0.2
    ):
        super().__init__()
        
        self.backbone = backbone
        self.embed_dim = embed_dim
        
        # 分割解码器（轻量级）
        self.seg_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, seg_channels, 1)
        )
        
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # 分类头（融合 cls_token 和掩码引导的特征）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, return_attention=False):
        B, C, H, W = x.shape
        
        # 提取 DINOv3 特征
        features = self.backbone.forward_features(x)
        cls_token = features['x_norm_clstoken']  # [B, embed_dim]
        patch_tokens = features['x_norm_patchtokens']  # [B, N, embed_dim]
        
        # 重塑 patch tokens 为空间特征图
        num_patches = int(patch_tokens.shape[1] ** 0.5)
        spatial_features = patch_tokens.transpose(1, 2).reshape(
            B, self.embed_dim, num_patches, num_patches
        )
        
        # 生成分割掩码
        seg_logits = self.seg_decoder(spatial_features)
        seg_logits_upsampled = F.interpolate(
            seg_logits, 
            size=(H, W), 
            mode='bilinear', 
            align_corners=False
        )
        seg_prob = torch.sigmoid(seg_logits)
        
        # 使用分割掩码增强特征
        enhanced_features = self.feature_enhancer(spatial_features)
        masked_features = enhanced_features * seg_prob
        
        # 全局平均池化（聚焦瑕疵区域）
        masked_global = masked_features.mean(dim=[2, 3])
        
        # 融合 cls_token 和掩码引导特征
        combined_features = torch.cat([cls_token, masked_global], dim=1)
        
        # 分类
        cls_logits = self.classifier(combined_features)
        
        outputs = {
            'classification': cls_logits,
            'segmentation': seg_logits_upsampled,
            'seg_prob': seg_prob
        }
        
        if return_attention:
            outputs['attention_map'] = seg_prob
            outputs['spatial_features'] = spatial_features
            
        return outputs


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    支持固定权重和不确定性自适应权重两种模式
    """
    
    def __init__(
        self, 
        alpha=1.0, 
        beta=1.0, 
        use_uncertainty_weighting=False,
        focal_loss=False,
        focal_alpha=0.25,
        focal_gamma=2.0
    ):
        super().__init__()
        
        self.alpha = alpha  # 分类损失权重
        self.beta = beta    # 分割损失权重
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.focal_loss = focal_loss
        
        # 不确定性加权参数
        if use_uncertainty_weighting:
            self.log_var_cls = nn.Parameter(torch.zeros(1))
            self.log_var_seg = nn.Parameter(torch.zeros(1))
        
        # 损失函数
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()
        
        # Focal Loss 参数（可选，用于处理类别不平衡）
        if focal_loss:
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
    
    def focal_loss_fn(self, logits, targets):
        """Focal Loss for classification"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs, cls_labels, seg_masks):
        cls_logits = outputs['classification']
        seg_logits = outputs['segmentation']
        
        # 计算分类损失
        if self.focal_loss:
            cls_loss = self.focal_loss_fn(cls_logits, cls_labels)
        else:
            cls_loss = self.cls_criterion(cls_logits, cls_labels)
        
        # 计算分割损失
        seg_loss = self.seg_criterion(seg_logits, seg_masks)
        
        # 组合损失
        if self.use_uncertainty_weighting:
            # 基于不确定性的自适应权重
            precision_cls = torch.exp(-self.log_var_cls)
            precision_seg = torch.exp(-self.log_var_seg)
            total_loss = (
                precision_cls * cls_loss + self.log_var_cls +
                precision_seg * seg_loss + self.log_var_seg
            )
        else:
            # 固定权重
            total_loss = self.alpha * cls_loss + self.beta * seg_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'seg_loss': seg_loss
        }


class DynamicWeightScheduler:
    """
    动态调整多任务损失权重
    初期：分割权重较大，帮助模型学习瑕疵位置
    后期：分类权重增大，提升分类性能
    """
    
    def __init__(
        self, 
        alpha_start=0.5, 
        alpha_end=1.0,
        beta_start=1.0, 
        beta_end=0.3,
        warmup_epochs=20,
        total_epochs=100
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
    
    def get_weights(self, epoch):
        """根据训练进度返回当前权重"""
        if epoch < self.warmup_epochs:
            # Warmup 阶段：线性增长
            progress = epoch / self.warmup_epochs
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
            beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            # 后期保持稳定
            alpha = self.alpha_end
            beta = self.beta_end
        
        return alpha, beta
