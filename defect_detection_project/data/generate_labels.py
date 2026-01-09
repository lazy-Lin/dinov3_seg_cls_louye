"""
生成数据集标签文件脚本
扫描 images 目录，根据文件名规则生成 train_labels.txt 和 val_labels.txt
"""

import os
import argparse
import random
from pathlib import Path

def generate_labels(
    data_root, 
    val_split=0.2, 
    seed=42,
    defect_keyword='defect',
    normal_keyword='normal'
):
    """
    生成标签文件
    
    Args:
        data_root: 数据集根目录（必须包含 images 子目录）
        val_split: 验证集比例
        seed: 随机种子
        defect_keyword: 文件名包含此关键字视为瑕疵样本 (Label 1)
        normal_keyword: 文件名包含此关键字视为正常样本 (Label 0)
    """
    data_root = Path(data_root)
    images_dir = data_root / 'images'
    
    if not images_dir.exists():
        print(f"错误: 找不到图片目录: {images_dir}")
        print("请确保数据结构如下:")
        print(f"{data_root}/")
        print("  └── images/")
        return

    # 扫描图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    all_images = []
    
    print(f"正在扫描 {images_dir} ...")
    
    for f in images_dir.iterdir():
        if f.suffix.lower() in valid_exts:
            all_images.append(f.name)
            
    if not all_images:
        print("未找到任何图片文件！")
        return

    print(f"共找到 {len(all_images)} 张图片")

    # 打标
    samples = []
    defect_count = 0
    normal_count = 0
    unknown_count = 0

    for img_name in all_images:
        name_lower = img_name.lower()
        
        if defect_keyword in name_lower:
            label = 1
            defect_count += 1
        elif normal_keyword in name_lower:
            label = 0
            normal_count += 1
        else:
            # 默认逻辑：如果既不包含 defect 也不包含 normal
            # 这里的策略是：打印警告并默认设为 1 (瑕疵)，或者跳过？
            # 为了安全起见，我们将其设为 -1 并让用户决定，或者默认为瑕疵
            # 这里我们选择默认为瑕疵，但在控制台提示
            print(f"警告: 文件名 '{img_name}' 不包含 '{defect_keyword}' 或 '{normal_keyword}'，默认标记为瑕疵(1)")
            label = 1
            unknown_count += 1
            
        samples.append((img_name, label))

    print(f"\n统计:")
    print(f"  瑕疵样本 (1): {defect_count + unknown_count}")
    print(f"  正常样本 (0): {normal_count}")
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(samples)
    
    # 划分数据集
    val_size = int(len(samples) * val_split)
    train_samples = samples[val_size:]
    val_samples = samples[:val_size]
    
    print(f"\n划分结果 (验证集比例 {val_split}):")
    print(f"  训练集: {len(train_samples)}")
    print(f"  验证集: {len(val_samples)}")
    
    # 写入文件
    def write_txt(filename, data):
        path = data_root / filename
        with open(path, 'w') as f:
            for name, label in data:
                f.write(f"{name},{label}\n")
        print(f"已生成: {path}")

    write_txt('train_labels.txt', train_samples)
    write_txt('val_labels.txt', val_samples)
    print("\n完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成数据集标签文件')
    parser.add_argument('--data_root', type=str, default='.', help='数据集根目录 (包含 images 文件夹)')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--defect_keyword', type=str, default='defect', help='瑕疵样本文件名关键字')
    parser.add_argument('--normal_keyword', type=str, default='normal', help='正常样本文件名关键字')
    
    args = parser.parse_args()
    
    generate_labels(
        args.data_root,
        args.val_split,
        args.seed,
        args.defect_keyword,
        args.normal_keyword
    )
