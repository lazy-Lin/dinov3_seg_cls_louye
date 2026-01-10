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
    生成标签划分文件 (train.txt / val.txt)
    
    扫描 images/normal 和 images/defect 目录，生成训练和验证集的划分文件。
    生成的文件内容为相对于 images/ 目录的路径，例如：
    defect/001.jpg
    normal/002.jpg
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
        print("      ├── defect/")
        print("      └── normal/")
        return

    # 扫描图片
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.heic'}
    all_images = []
    
    print(f"正在扫描 {images_dir} ...")
    
    # 分别扫描 defect 和 normal 目录
    defect_dir = images_dir / 'defect'
    normal_dir = images_dir / 'normal'
    
    defect_count = 0
    normal_count = 0
    
    # 扫描 defect
    if defect_dir.exists():
        for f in defect_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in valid_exts:
                # 获取相对于 images 的路径
                rel_path = f.relative_to(images_dir)
                all_images.append(str(rel_path).replace('\\', '/'))
                defect_count += 1
    else:
        print(f"Warning: Defect directory not found: {defect_dir}")

    # 扫描 normal
    if normal_dir.exists():
        for f in normal_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in valid_exts:
                # 获取相对于 images 的路径
                rel_path = f.relative_to(images_dir)
                all_images.append(str(rel_path).replace('\\', '/'))
                normal_count += 1
    else:
        print(f"Warning: Normal directory not found: {normal_dir}")
            
    if not all_images:
        print("未找到任何图片文件！")
        return

    print(f"共找到 {len(all_images)} 张图片")
    print(f"  瑕疵样本 (1): {defect_count}")
    print(f"  正常样本 (0): {normal_count}")
    
    # 随机打乱
    random.seed(seed)
    random.shuffle(all_images)
    
    # 划分数据集
    val_size = int(len(all_images) * val_split)
    train_samples = all_images[val_size:]
    val_samples = all_images[:val_size]
    
    print(f"\n划分结果 (验证集比例 {val_split}):")
    print(f"  训练集: {len(train_samples)}")
    print(f"  验证集: {len(val_samples)}")
    
    # 确保 splits 目录存在
    splits_dir = data_root / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    # 写入文件
    def write_txt(filename, data):
        path = splits_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            for line in data:
                f.write(f"{line}\n")
        print(f"已生成: {path}")

    write_txt('train.txt', train_samples)
    write_txt('val.txt', val_samples)
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
