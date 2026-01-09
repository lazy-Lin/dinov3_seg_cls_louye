"""
生成数据集标签文件脚本
扫描 images 目录，根据文件名规则生成 train_labels.txt 和 val_labels.txt
"""

import os
import argparse
import random
import yaml
import sys
from pathlib import Path

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def generate_labels(
    data_root=None, 
    val_split=None, 
    seed=42,
    defect_keyword='defect',
    normal_keyword='normal',
    config_path=None
):
    """
    生成标签文件
    """
    # 优先使用 config 中的参数
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            config = load_config(config_path)
            if data_root is None:
                data_root = config.get('data', {}).get('data_root')
            if val_split is None:
                val_split = config.get('data', {}).get('val_split', 0.2)
        else:
             print(f"Warning: Config file {config_path} not found.")

    if data_root is None:
        data_root = '.'
    if val_split is None:
        val_split = 0.2

    data_root = Path(data_root)
    images_dir = data_root / 'images'
    
    if not images_dir.exists():
        print(f"错误: 找不到图片目录: {images_dir}")
        print("请确保数据结构如下:")
        print(f"{data_root}/")
        print("  └── images/")
        return

    # 扫描图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.heic'}
    all_images = []
    
    print(f"正在扫描 {images_dir} ...")
    
    # 递归扫描所有子目录
    for f in images_dir.rglob('*'):
        if f.is_file() and f.suffix.lower() in valid_exts:
            # 记录相对路径或者直接记录文件名（如果文件名唯一）
            # 根据 defect_dataset.py 的逻辑，它只存储文件名，然后通过 map 查找
            # 所以这里我们只需要存储文件名，或者存储相对路径，但 dataset 那边用的是 name
            # 为了兼容性，我们存储文件名
            all_images.append(f)
            
    if not all_images:
        print("未找到任何图片文件！")
        return

    print(f"共找到 {len(all_images)} 张图片")

    # 打标
    samples = []
    defect_count = 0
    normal_count = 0
    unknown_count = 0

    for img_path in all_images:
        img_name = img_path.name
        # 优先使用父文件夹名称作为类别判定依据
        # 注意：这里的 parent 是直接父目录。
        # 如果结构是 images/louye/images/xxx.jpg，那么 parent 是 images
        # 这种情况下我们需要往上找一级，直到找到 defect_keyword 或 normal_keyword
        
        # 获取从 images 目录开始的相对路径
        try:
            rel_path = img_path.relative_to(images_dir)
            # rel_path.parts 会是一个元组，例如 ('louye', 'images', 'xxx.jpg')
            path_parts = [p.lower() for p in rel_path.parts]
        except ValueError:
            # 如果 img_path 不在 images_dir 下（不太可能，因为我们是用 rglob 找的），回退到直接父目录
            path_parts = [img_path.parent.name.lower()]
            rel_path = img_path.parent.name
            
        name_lower = img_name.lower()
        
        # 判定规则：
        # 1. 直接检查父文件夹名称
        # 如果父文件夹名称包含 normal_keyword，则为正常样本(0)
        # 否则默认为瑕疵样本(1)
        
        # 获取直接父目录名称
        parent_dir = img_path.parent.name.lower()
        
        if normal_keyword in parent_dir:
            label = 0
            normal_count += 1
        else:
            # 只要不是 normal，就认为是 defect
            # 即使父文件夹是 defect_keyword 或者其他名称（如 louye），都算瑕疵
            label = 1
            defect_count += 1
            
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
    
    # 确保 labels 目录存在
    labels_dir = data_root / 'labels'
    labels_dir.mkdir(exist_ok=True)
    
    # 写入文件
    def write_txt(filename, data):
        path = labels_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            for name, label in data:
                f.write(f"{name},{label}\n")
        print(f"已生成: {path}")

    write_txt('train_labels.txt', train_samples)
    write_txt('val_labels.txt', val_samples)
    print("\n完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成数据集标签文件')
    
    # 优先使用配置文件
    parser.add_argument('--config', type=str, default='defect_detection_project/configs/config_defect_detection.yaml',
                        help='配置文件路径')
    
    # 命令行参数可覆盖配置
    parser.add_argument('--data_root', type=str, help='数据集根目录 (包含 images 文件夹)')
    parser.add_argument('--val_split', type=float, help='验证集比例')
    
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--defect_keyword', type=str, default='defect', help='瑕疵样本文件名关键字')
    parser.add_argument('--normal_keyword', type=str, default='normal', help='正常样本文件名关键字')
    
    args = parser.parse_args()
    
    # 构造绝对路径的 config
    if args.config:
        if os.path.isabs(args.config):
            config_path = args.config
        else:
             # 假设在项目根目录运行或相对于当前脚本
             # 尝试相对于项目根目录
            config_path = os.path.join(project_root, args.config)
    else:
        config_path = None
    
    generate_labels(
        data_root=args.data_root,
        val_split=args.val_split,
        seed=args.seed,
        defect_keyword=args.defect_keyword,
        normal_keyword=args.normal_keyword,
        config_path=config_path
    )
