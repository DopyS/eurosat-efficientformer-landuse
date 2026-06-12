# 技术设计

## 技术栈

- Python 3.11
- PyTorch
- torchvision
- timm
- scikit-learn
- matplotlib
- Streamlit

## 数据集

EuroSAT 是遥感土地利用图像分类数据集。项目使用 RGB 图像版本，目标是完成 10 类场景分类。

计划类别包括：

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## 数据流

1. 下载或读取 EuroSAT RGB 数据。
2. 划分训练集、验证集和测试集。
3. 对训练集应用增强策略。
4. 将图像输入 EfficientFormerV2。
5. 输出 10 类分类概率。
6. 计算准确率、损失、混淆矩阵和预测示例。
7. 将训练产物保存到本地 `outputs/`，但不提交到 GitHub。
8. 使用独立评估脚本加载 checkpoint，在验证集或测试集上生成整体指标和逐类指标。

默认数据划分比例：

- 训练集：70%
- 验证集：15%
- 测试集：15%

划分使用 `project.seed` 控制随机种子，保证后续实验可复现。

## 模型方案

基线模型使用 timm 提供的 EfficientFormerV2 预训练模型，替换分类头为 10 类输出。

当前 timm 版本使用模型名 `efficientformerv2_s0`。如果配置中出现带预训练标签的旧模型名，模型构建模块会尝试解析为当前 timm 可用的基础模型名。

基线阶段：

- 加载 ImageNet 预训练权重。
- 冻结或半冻结 backbone 作为低成本启动方案。
- 训练分类头并记录验证集表现。
- 先使用限制 batch 的短程训练验证完整链路，再逐步扩大训练轮数和数据量。

增强阶段：

- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter
- Mixup
- Label Smoothing
- Cosine Annealing 学习率策略

## 评估指标

- Top-1 Accuracy
- Validation Loss
- Test Accuracy
- Confusion Matrix
- Per-class Accuracy
- Top-3 Prediction Examples

## 输出约定

实验输出统一保存到 `outputs/`，包括：

- `outputs/checkpoints/`
- `outputs/figures/`
- `outputs/predictions/`
- `outputs/metrics/`

这些文件默认被 `.gitignore` 排除，只在报告中引用整理后的图表或结果描述。
