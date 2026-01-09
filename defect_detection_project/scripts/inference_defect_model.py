"""
瑕疵检测模型推理和可视化
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path
import cv2
import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import dinov3
from defect_detection_project.models.defect_classifier import AttentionGuidedDefectClassifier


def load_model(checkpoint_path, backbone_name='dinov3_vitb14', device='cuda'):
    """加载训练好的模型"""
    # 骨干网络配置
    backbone_configs = {
        'dinov3_vits14': {'model': 'dinov3_vits14', 'embed_dim': 384},
        'dinov3_vitb14': {'model': 'dinov3_vitb14', 'embed_dim': 768},
        'dinov3_vitl14': {'model': 'dinov3_vitl14', 'embed_dim': 1024},
        'dinov3_vitg14': {'model': 'dinov3_vitg14', 'embed_dim': 1536},
    }
    
    config = backbone_configs[backbone_name]
    
    # 加载骨干网络
    print(f"Loading {backbone_name}...")
    backbone = torch.hub.load('facebookresearch/dinov3', config['model'])
    
    # 创建模型
    model = AttentionGuidedDefectClassifier(
        backbone=backbone,
        embed_dim=config['embed_dim'],
        num_classes=2,
        seg_channels=1
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Validation metrics: {checkpoint.get('val_metrics', 'N/A')}")
    
    return model


def preprocess_image(image_path, image_size=518):
    """预处理图像"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize
    image = image.resize((image_size, image_size), Image.BILINEAR)
    
    # 转换为 tensor 并归一化
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, original_size


@torch.no_grad()
def predict(model, image_tensor, device='cuda'):
    """模型推理"""
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor, return_attention=True)
    
    # 分类结果
    cls_logits = outputs['classification']
    cls_probs = F.softmax(cls_logits, dim=1)
    cls_pred = torch.argmax(cls_probs, dim=1).item()
    cls_confidence = cls_probs[0, cls_pred].item()
    
    # 分割结果
    seg_logits = outputs['segmentation']
    seg_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    
    # 注意力图
    attention_map = outputs['attention_map'][0, 0].cpu().numpy()
    
    return {
        'class': cls_pred,
        'confidence': cls_confidence,
        'class_probs': cls_probs[0].cpu().numpy(),
        'segmentation': seg_prob,
        'attention': attention_map
    }


def visualize_results(image_path, predictions, save_path=None):
    """可视化预测结果"""
    # 读取原始图像
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. 原始图像
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. 分割掩码
    seg_mask = predictions['segmentation']
    seg_mask_resized = cv2.resize(seg_mask, (image_np.shape[1], image_np.shape[0]))
    axes[0, 1].imshow(image_np)
    axes[0, 1].imshow(seg_mask_resized, alpha=0.5, cmap='jet')
    axes[0, 1].set_title('Segmentation Mask')
    axes[0, 1].axis('off')
    
    # 3. 注意力图
    attention = predictions['attention']
    attention_resized = cv2.resize(attention, (image_np.shape[1], image_np.shape[0]))
    axes[1, 0].imshow(image_np)
    axes[1, 0].imshow(attention_resized, alpha=0.5, cmap='hot')
    axes[1, 0].set_title('Attention Map')
    axes[1, 0].axis('off')
    
    # 4. 分类结果
    class_names = ['Normal', 'Defect']
    class_probs = predictions['class_probs']
    
    axes[1, 1].barh(class_names, class_probs, color=['green', 'red'])
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_xlabel('Probability')
    axes[1, 1].set_title(f'Classification: {class_names[predictions["class"]]} '
                         f'({predictions["confidence"]:.2%})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    print("Loading model...")
    model = load_model(args.checkpoint, args.backbone, device)
    
    # 处理输入
    if args.image:
        # 单张图像
        image_paths = [Path(args.image)]
    elif args.image_dir:
        # 目录中的所有图像
        image_dir = Path(args.image_dir)
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    else:
        raise ValueError("Please provide --image or --image_dir")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 推理
    print(f"Processing {len(image_paths)} images...")
    results = []
    
    for image_path in image_paths:
        print(f"\nProcessing {image_path.name}...")
        
        # 预处理
        image_tensor, original_size = preprocess_image(image_path, args.image_size)
        
        # 推理
        predictions = predict(model, image_tensor, device)
        
        # 打印结果
        class_names = ['Normal', 'Defect']
        print(f"  Class: {class_names[predictions['class']]}")
        print(f"  Confidence: {predictions['confidence']:.2%}")
        print(f"  Probabilities: Normal={predictions['class_probs'][0]:.2%}, "
              f"Defect={predictions['class_probs'][1]:.2%}")
        
        # 可视化
        save_path = output_dir / f"{image_path.stem}_result.png"
        visualize_results(image_path, predictions, save_path)
        
        results.append({
            'image': image_path.name,
            'class': class_names[predictions['class']],
            'confidence': predictions['confidence']
        })
    
    # 保存结果摘要
    summary_path = output_dir / 'results_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Image,Class,Confidence\n")
        for result in results:
            f.write(f"{result['image']},{result['class']},{result['confidence']:.4f}\n")
    
    print(f"\nResults summary saved to {summary_path}")
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Defect detection inference')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='dinov3_vitb14',
                        choices=['dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14'],
                        help='DINOv3 backbone model (must match training)')
    parser.add_argument('--image', type=str,
                        help='Path to single image')
    parser.add_argument('--image_dir', type=str,
                        help='Path to directory of images')
    parser.add_argument('--image_size', type=int, default=518,
                        help='Input image size')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    main(args)
