# 项目索引

## 📖 快速导航

### 🚀 新手入门（按顺序阅读）

1. **[README.md](README.md)** - 从这里开始！
   - 项目概览
   - 快速开始步骤
   - 性能表现
   - 模型选择

2. **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - 设置指南
   - 完整项目结构
   - 安装步骤
   - 与 DINOv3 的关系
   - 导入方式

3. **[QUICK_START_EXAMPLE.py](QUICK_START_EXAMPLE.py)** - 运行示例
   - 可执行的代码示例
   - 5 个实用示例
   - 快速验证安装

### 📚 项目文档

4. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目总结
   - 已完成的工作
   - 使用方式
   - 性能预期
   - 核心优势

5. **[FILE_MANIFEST.md](FILE_MANIFEST.md)** - 文件清单
   - 已创建的文件列表
   - 待补充的文件
   - 如何补充文件
   - 最小可用配置

6. **[INDEX.md](INDEX.md)** - 本文档
   - 所有文档的索引
   - 快速查找指南

### 🔧 核心模块

7. **[models/defect_classifier.py](models/defect_classifier.py)** - 模型定义
   - `AttentionGuidedDefectClassifier` - 主模型
   - `MultiTaskLoss` - 损失函数
   - `DynamicWeightScheduler` - 权重调度器

8. **[models/__init__.py](models/__init__.py)** - 模块导出

9. **[data/dataset.py](data/dataset.py)** - 数据集
   - `DefectDataset` - 数据集类
   - `create_dataloaders` - 数据加载器

10. **[data/__init__.py](data/__init__.py)** - 模块导出

### 📦 配置文件

11. **[requirements.txt](requirements.txt)** - Python 依赖
    - 核心依赖
    - 数据处理
    - 训练工具
    - 可视化

## 🗂️ 根目录参考文档

以下文档在项目根目录（上一级目录），包含完整的实现细节：

### 快速参考
- **QUICK_REFERENCE.md** - 最常用的命令和配置
- **MODEL_SIZE_EXPLANATION.md** - 模型参数详细说明（回答你的问题）

### 详细指南
- **examples/README_DEFECT_DETECTION.md** - 完整使用指南
- **examples/IMPLEMENTATION_GUIDE.md** - 实现细节和方案对比
- **examples/LARGE_DATASET_GUIDE.md** - 大数据集（1w+）训练策略

### 脚本和工具
- **examples/train_defect_model.py** - 训练脚本
- **examples/inference_defect_model.py** - 推理脚本
- **examples/prepare_defect_data.py** - 数据准备工具
- **examples/test_model.py** - 模型测试
- **examples/train_large_dataset.sh/.bat** - 大数据集训练脚本
- **examples/quick_start.sh/.bat** - 快速开始脚本

### 配置示例
- **examples/config_defect_detection.yaml** - YAML 配置文件

### 训练模块
- **dinov3/train/train_defect_classifier.py** - 完整训练逻辑

## 🎯 按需求查找

### 我想...

#### 快速了解项目
→ 阅读 [README.md](README.md)

#### 设置开发环境
→ 阅读 [SETUP_GUIDE.md](SETUP_GUIDE.md)

#### 运行示例代码
→ 运行 [QUICK_START_EXAMPLE.py](QUICK_START_EXAMPLE.py)

#### 了解已完成的工作
→ 阅读 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

#### 知道还需要哪些文件
→ 阅读 [FILE_MANIFEST.md](FILE_MANIFEST.md)

#### 理解模型参数量
→ 阅读根目录的 **MODEL_SIZE_EXPLANATION.md**

#### 训练大数据集（1w+ 样本）
→ 阅读根目录的 **examples/LARGE_DATASET_GUIDE.md**

#### 了解实现细节
→ 阅读根目录的 **examples/IMPLEMENTATION_GUIDE.md**

#### 查看常用命令
→ 阅读根目录的 **QUICK_REFERENCE.md**

#### 准备数据
→ 使用根目录的 **examples/prepare_defect_data.py**

#### 开始训练
→ 使用根目录的 **examples/train_defect_model.py**
→ 或使用 **examples/train_large_dataset.sh/.bat**

#### 推理测试
→ 使用根目录的 **examples/inference_defect_model.py**

#### 自定义配置
→ 参考根目录的 **examples/config_defect_detection.yaml**

## 📊 文档类型分类

### 入门文档 🚀
- README.md
- SETUP_GUIDE.md
- QUICK_START_EXAMPLE.py

### 参考文档 📚
- PROJECT_SUMMARY.md
- FILE_MANIFEST.md
- INDEX.md (本文档)

### 技术文档 🔧
- models/defect_classifier.py
- data/dataset.py
- requirements.txt

### 外部参考 📖
- 根目录的 QUICK_REFERENCE.md
- 根目录的 MODEL_SIZE_EXPLANATION.md
- 根目录的 examples/*.md

## 🔍 按主题查找

### 模型相关
- models/defect_classifier.py - 模型实现
- 根目录 MODEL_SIZE_EXPLANATION.md - 参数说明
- 根目录 examples/IMPLEMENTATION_GUIDE.md - 实现细节

### 数据相关
- data/dataset.py - 数据集实现
- 根目录 examples/prepare_defect_data.py - 数据准备
- README.md - 数据组织结构

### 训练相关
- 根目录 dinov3/train/train_defect_classifier.py - 训练逻辑
- 根目录 examples/train_defect_model.py - 训练脚本
- 根目录 examples/LARGE_DATASET_GUIDE.md - 大数据集训练
- 根目录 examples/train_large_dataset.sh/.bat - 训练脚本

### 推理相关
- 根目录 examples/inference_defect_model.py - 推理脚本
- README.md - 推理示例

### 配置相关
- requirements.txt - 依赖配置
- 根目录 examples/config_defect_detection.yaml - 训练配置
- SETUP_GUIDE.md - 环境配置

## 💡 推荐阅读路径

### 路径 1：快速上手（30 分钟）
1. README.md - 了解项目
2. QUICK_START_EXAMPLE.py - 运行示例
3. 开始编码！

### 路径 2：完整学习（2 小时）
1. README.md - 项目概览
2. SETUP_GUIDE.md - 详细设置
3. PROJECT_SUMMARY.md - 项目总结
4. 根目录 MODEL_SIZE_EXPLANATION.md - 理解模型
5. 根目录 examples/IMPLEMENTATION_GUIDE.md - 实现细节
6. 根目录 examples/LARGE_DATASET_GUIDE.md - 训练策略

### 路径 3：生产部署（1 天）
1. 阅读所有文档
2. 从根目录复制完整文件（参考 FILE_MANIFEST.md）
3. 准备数据
4. 训练模型
5. 评估和优化

## 📞 获取帮助

### 遇到问题？

1. **查看文档**：按主题在本索引中查找
2. **运行示例**：`python QUICK_START_EXAMPLE.py`
3. **查看清单**：FILE_MANIFEST.md 了解文件状态
4. **参考根目录**：完整的代码和文档都在根目录

### 常见问题快速链接

- **模型参数够用吗？** → 根目录 MODEL_SIZE_EXPLANATION.md
- **如何训练大数据集？** → 根目录 examples/LARGE_DATASET_GUIDE.md
- **需要哪些文件？** → FILE_MANIFEST.md
- **如何设置环境？** → SETUP_GUIDE.md
- **有示例代码吗？** → QUICK_START_EXAMPLE.py

## 🎉 开始使用

**推荐起点**：

```bash
# 1. 阅读项目概览
cat README.md

# 2. 运行示例
python QUICK_START_EXAMPLE.py

# 3. 查看设置指南
cat SETUP_GUIDE.md

# 4. 开始开发！
```

---

**所有文档都在这里，开始探索吧！** 🚀
