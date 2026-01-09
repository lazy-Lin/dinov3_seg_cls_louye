"""
瑕疵检测数据集
支持带瑕疵（有 label 和 mask）和不带瑕疵的图片
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DefectDataset(Dataset):
    """
    瑕疵检测数据集
    
    数据组织结构：
    data_root/
        images/
            defect_001.jpg
            defect_002.jpg
            normal_001.jpg
            ...
        masks/
            defect_001.png  # 二值掩码
            defect_002.png
            # 正常样本无需 mask
        labels.txt  # 格式: image_name,label (0=正常, 1=瑕疵)
    """
    
    def __init__(
        self,
        data_root,
        split='train',
        image_size=518,
        augment=True
    ):
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / 'images'
        self.mask_dir = self.data_root / 'masks'
        self.image_size = image_size
        self.augment = augment and split == 'train'
        
        # 读取标签文件
        self.samples = self._load_samples(split)
        
        # 数据增强
        self.transform = self._get_transforms()
    
    def _load_samples(self, split):
        """加载样本列表"""
        label_file = self.data_root / f'{split}_labels.txt'
        
        samples = []
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                image_name = parts[0]
                label = int(parts[1])
                
                image_path = self.image_dir / image_name
                
                # 瑕疵样本需要 mask
                if label == 1:
                    mask_name = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = self.mask_dir / mask_name
                    if not mask_path.exists():
                        print(f"Warning: Mask not found for {image_name}, skipping...")
                        continue
                else:
                    mask_path = None
                
                samples.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'label': label
                })
        
        print(f"Loaded {len(samples)} samples for {split}")
        return samples
    
    def _get_transforms(self):
        """获取数据增强变换"""
        if self.augment:
            # 训练时的数据增强
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=45,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=7),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20,
                        sat_shift_limit=30,
                        val_shift_limit=20
                    ),
                ], p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # 验证/测试时只做基本变换
            transform = A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        
        return transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图像
        image = Image.open(sample['image_path']).convert('RGB')
        image = np.array(image)
        
        # 读取掩码（如果存在）
        if sample['mask_path'] is not None:
            mask = Image.open(sample['mask_path']).convert('L')
            mask = np.array(mask)
            # 二值化
            mask = (mask > 127).astype(np.float32)
        else:
            # 正常样本的掩码全为 0
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # 应用变换
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # 确保 mask 维度正确 [1, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        return {
            'image': image,
            'mask': mask,
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }


def create_dataloaders(
    data_root,
    batch_size=32,
    num_workers=4,
    image_size=518
):
    """创建训练和验证数据加载器"""
    
    train_dataset = DefectDataset(
        data_root=data_root,
        split='train',
        image_size=image_size,
        augment=True
    )
    
    val_dataset = DefectDataset(
        data_root=data_root,
        split='val',
        image_size=image_size,
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
