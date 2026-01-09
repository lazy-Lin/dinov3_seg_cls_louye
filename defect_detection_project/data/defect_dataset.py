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
            defect/  # 瑕疵图片文件夹
                xxx.jpg
            normal/  # 正常图片文件夹
                yyy.jpg
        masks/
            xxx.png  # 二值掩码
            # 正常样本无需 mask
        labels/      # 标签文件夹
            train_labels.txt  # 格式: image_name,label (0=正常, 1=瑕疵)
            val_labels.txt
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
        # 兼容旧路径（labels.txt 直接在 data_root 下）和新路径（labels/labels.txt）
        self.labels_dir = self.data_root / 'labels' if (self.data_root / 'labels').exists() else self.data_root
        
        self.image_size = image_size
        self.augment = augment and split == 'train'
        
        # 读取标签文件
        self.samples = self._load_samples(split)
        
        # 数据增强
        self.transform = self._get_transforms()
    
    def _load_samples(self, split):
        """加载样本列表"""
        label_file = self.labels_dir / f'{split}_labels.txt'
        
        samples = []
        try:
            # 尝试使用 UTF-8 编码
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # 如果失败，尝试使用 GBK 编码（兼容 Windows 生成的文件）
            print(f"Warning: Failed to read {label_file} with UTF-8, trying GBK...")
            with open(label_file, 'r', encoding='gbk') as f:
                lines = f.readlines()

        # 缓存 images 目录下的所有文件路径，以便快速查找
        # 假设 images 下面有子文件夹 (如 defect, normal) 或直接是图片
        print(f"Scanning images in {self.image_dir}...")
        image_path_map = {}
        for p in self.image_dir.rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.heic']:
                image_path_map[p.name] = p

        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            image_name = parts[0]
            label = int(parts[1])
            
            # 在 image_path_map 中查找图片路径
            if image_name in image_path_map:
                image_path = image_path_map[image_name]
            else:
                print(f"Warning: Image {image_name} not found in {self.image_dir} (recursive search), skipping...")
                continue
            
            # 瑕疵样本需要 mask
            if label == 1:
                # 尝试查找 mask 文件
                # 策略1: 去掉 _defect 后缀 (例如 xxx_defect.jpg -> xxx.png)
                mask_name_v1 = image_name.replace('.jpg', '.png').replace('.jpeg', '.png').replace('_defect', '')
                mask_path_v1 = self.mask_dir / mask_name_v1
                
                # 策略2: 直接替换后缀 (例如 xxx.jpg -> xxx.png)
                mask_name_v2 = image_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path_v2 = self.mask_dir / mask_name_v2

                # 策略3: 兼容复杂后缀 (例如 ..._Q90.jpg__defect.jpg -> ..._Q90.png)
                # 移除所有可能的扩展名后缀，然后加上 .png
                base_name = image_name
                # 反复移除可能的扩展名，直到没有为止
                while True:
                    stem = Path(base_name).stem
                    if stem == base_name:
                        break
                    base_name = stem
                
                # 移除 _defect
                base_name = base_name.replace('_defect', '')
                # 移除可能残留的 .jpg, .png 等（虽然stem应该已经处理了，但为了保险）
                base_name = base_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                
                mask_name_v3 = base_name + '.png'
                mask_path_v3 = self.mask_dir / mask_name_v3
                
                if mask_path_v1.exists():
                    mask_path = mask_path_v1
                elif mask_path_v2.exists():
                    mask_path = mask_path_v2
                elif mask_path_v3.exists():
                     mask_path = mask_path_v3
                else:
                    # 尝试模糊匹配，因为文件名可能经过了 url 编码或者截断等处理
                    # 只取前 20 个字符进行匹配
                    found = False
                    prefix = image_name[:20]
                    for m_path in self.mask_dir.glob(f"{prefix}*.png"):
                        mask_path = m_path
                        found = True
                        break
                    
                    if not found:
                        print(f"Warning: Mask not found for {image_name} (tried {mask_name_v1}, {mask_name_v2}, {mask_name_v3}), skipping...")
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
