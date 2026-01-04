"""
数据准备工具
帮助将原始数据组织成训练所需的格式
"""

import argparse
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random


def organize_data(
    defect_images_dir,
    defect_masks_dir,
    normal_images_dir,
    output_dir,
    val_split=0.2,
    seed=42
):
    """
    组织数据到标准格式
    
    Args:
        defect_images_dir: 瑕疵图片目录
        defect_masks_dir: 瑕疵掩码目录
        normal_images_dir: 正常图片目录
        output_dir: 输出目录
        val_split: 验证集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    # 收集瑕疵样本
    defect_images = []
    defect_images_dir = Path(defect_images_dir)
    defect_masks_dir = Path(defect_masks_dir)
    
    for img_path in defect_images_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 查找对应的 mask
            mask_name = img_path.stem + '.png'
            mask_path = defect_masks_dir / mask_name
            
            if not mask_path.exists():
                # 尝试其他扩展名
                mask_path = defect_masks_dir / (img_path.stem + '.jpg')
            
            if mask_path.exists():
                defect_images.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for {img_path.name}, skipping...")
    
    print(f"Found {len(defect_images)} defect samples")
    
    # 收集正常样本
    normal_images = []
    normal_images_dir = Path(normal_images_dir)
    
    for img_path in normal_images_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            normal_images.append(img_path)
    
    print(f"Found {len(normal_images)} normal samples")
    
    # 划分训练集和验证集
    defect_train, defect_val = train_test_split(
        defect_images, test_size=val_split, random_state=seed
    )
    normal_train, normal_val = train_test_split(
        normal_images, test_size=val_split, random_state=seed
    )
    
    print(f"\nTrain: {len(defect_train)} defect + {len(normal_train)} normal")
    print(f"Val:   {len(defect_val)} defect + {len(normal_val)} normal")
    
    # 复制文件并生成标签
    train_labels = []
    val_labels = []
    
    # 处理训练集瑕疵样本
    for idx, (img_path, mask_path) in enumerate(defect_train):
        new_name = f"defect_train_{idx:04d}{img_path.suffix}"
        mask_name = f"defect_train_{idx:04d}.png"
        
        shutil.copy(img_path, images_dir / new_name)
        shutil.copy(mask_path, masks_dir / mask_name)
        
        train_labels.append(f"{new_name},1\n")
    
    # 处理训练集正常样本
    for idx, img_path in enumerate(normal_train):
        new_name = f"normal_train_{idx:04d}{img_path.suffix}"
        
        shutil.copy(img_path, images_dir / new_name)
        
        train_labels.append(f"{new_name},0\n")
    
    # 处理验证集瑕疵样本
    for idx, (img_path, mask_path) in enumerate(defect_val):
        new_name = f"defect_val_{idx:04d}{img_path.suffix}"
        mask_name = f"defect_val_{idx:04d}.png"
        
        shutil.copy(img_path, images_dir / new_name)
        shutil.copy(mask_path, masks_dir / mask_name)
        
        val_labels.append(f"{new_name},1\n")
    
    # 处理验证集正常样本
    for idx, img_path in enumerate(normal_val):
        new_name = f"normal_val_{idx:04d}{img_path.suffix}"
        
        shutil.copy(img_path, images_dir / new_name)
        
        val_labels.append(f"{new_name},0\n")
    
    # 打乱标签顺序
    random.shuffle(train_labels)
    random.shuffle(val_labels)
    
    # 保存标签文件
    with open(output_dir / 'train_labels.txt', 'w') as f:
        f.writelines(train_labels)
    
    with open(output_dir / 'val_labels.txt', 'w') as f:
        f.writelines(val_labels)
    
    print(f"\nData organized successfully!")
    print(f"Output directory: {output_dir}")
    print(f"- images/: {len(list(images_dir.glob('*')))} files")
    print(f"- masks/: {len(list(masks_dir.glob('*')))} files")
    print(f"- train_labels.txt: {len(train_labels)} samples")
    print(f"- val_labels.txt: {len(val_labels)} samples")


def create_dummy_data(output_dir, num_defect=100, num_normal=100):
    """
    创建示例数据（用于测试）
    """
    import numpy as np
    from PIL import Image, ImageDraw
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    train_labels = []
    val_labels = []
    
    print("Creating dummy defect samples...")
    for i in range(num_defect):
        # 创建随机图像
        img = Image.new('RGB', (512, 512), color=(
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200)
        ))
        
        # 创建掩码
        mask = Image.new('L', (512, 512), color=0)
        draw = ImageDraw.Draw(mask)
        
        # 随机绘制瑕疵区域
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            x = random.randint(50, 450)
            y = random.randint(50, 450)
            r = random.randint(20, 80)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        
        # 保存
        split = 'train' if i < num_defect * 0.8 else 'val'
        img_name = f"defect_{split}_{i:04d}.jpg"
        mask_name = f"defect_{split}_{i:04d}.png"
        
        img.save(images_dir / img_name)
        mask.save(masks_dir / mask_name)
        
        label = f"{img_name},1\n"
        if split == 'train':
            train_labels.append(label)
        else:
            val_labels.append(label)
    
    print("Creating dummy normal samples...")
    for i in range(num_normal):
        # 创建随机图像
        img = Image.new('RGB', (512, 512), color=(
            random.randint(150, 250),
            random.randint(150, 250),
            random.randint(150, 250)
        ))
        
        # 保存
        split = 'train' if i < num_normal * 0.8 else 'val'
        img_name = f"normal_{split}_{i:04d}.jpg"
        
        img.save(images_dir / img_name)
        
        label = f"{img_name},0\n"
        if split == 'train':
            train_labels.append(label)
        else:
            val_labels.append(label)
    
    # 保存标签
    with open(output_dir / 'train_labels.txt', 'w') as f:
        f.writelines(train_labels)
    
    with open(output_dir / 'val_labels.txt', 'w') as f:
        f.writelines(val_labels)
    
    print(f"\nDummy data created successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Train samples: {len(train_labels)}")
    print(f"Val samples: {len(val_labels)}")


def main():
    parser = argparse.ArgumentParser(description='Prepare defect detection data')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # 组织真实数据
    organize_parser = subparsers.add_parser('organize', help='Organize real data')
    organize_parser.add_argument('--defect_images', type=str, required=True,
                                 help='Directory of defect images')
    organize_parser.add_argument('--defect_masks', type=str, required=True,
                                 help='Directory of defect masks')
    organize_parser.add_argument('--normal_images', type=str, required=True,
                                 help='Directory of normal images')
    organize_parser.add_argument('--output_dir', type=str, required=True,
                                 help='Output directory')
    organize_parser.add_argument('--val_split', type=float, default=0.2,
                                 help='Validation split ratio')
    organize_parser.add_argument('--seed', type=int, default=42,
                                 help='Random seed')
    
    # 创建示例数据
    dummy_parser = subparsers.add_parser('dummy', help='Create dummy data for testing')
    dummy_parser.add_argument('--output_dir', type=str, required=True,
                             help='Output directory')
    dummy_parser.add_argument('--num_defect', type=int, default=100,
                             help='Number of defect samples')
    dummy_parser.add_argument('--num_normal', type=int, default=100,
                             help='Number of normal samples')
    
    args = parser.parse_args()
    
    if args.command == 'organize':
        organize_data(
            defect_images_dir=args.defect_images,
            defect_masks_dir=args.defect_masks,
            normal_images_dir=args.normal_images,
            output_dir=args.output_dir,
            val_split=args.val_split,
            seed=args.seed
        )
    elif args.command == 'dummy':
        create_dummy_data(
            output_dir=args.output_dir,
            num_defect=args.num_defect,
            num_normal=args.num_normal
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
