# 报告章节草稿

本文档用于沉淀结题报告正文素材。内容不是最终定稿，而是可直接扩写为报告的章节草稿、图表位置和写作要点。

## 题目

基于 EfficientFormerV2 与增强训练策略的 EuroSAT 遥感土地利用图像分类系统设计与实现

## 摘要草稿

遥感土地利用图像分类是遥感智能解译中的基础任务，在农业监测、城市规划、生态环境分析和灾害评估等场景中具有重要应用价值。针对遥感图像类别差异细微、背景复杂以及模型部署成本等问题，本文设计并实现了一个基于 EfficientFormerV2 的 EuroSAT 遥感土地利用图像分类系统。

系统以 EuroSAT RGB 数据集为实验对象，将数据划分为训练集、验证集和测试集，使用 timm 提供的 EfficientFormerV2-S0 预训练模型作为主干网络，并替换分类头以适配 10 类土地利用分类任务。在训练策略方面，本文实现了基线训练方案，并进一步加入 ColorJitter、Mixup 和 Label Smoothing 等增强训练策略进行对比实验。系统同时提供命令行训练、独立评估、实验汇总、图表生成、单图预测和 Streamlit Web 演示界面。

初步实验结果表明，EfficientFormerV2-S0 能够在较短训练流程下学习到有效的遥感图像特征。当前 100 batch 基线模型在验证集 480 个样本上达到 0.7729 准确率，在完整测试集 4050 个样本上达到 0.7748 准确率。增强训练策略在短训练条件下未超过基线准确率，但获得了更低的验证损失，说明其可能改善模型输出分布和泛化潜力。本文最后对系统实现、实验结果、类别混淆现象和后续优化方向进行了总结。

关键词：遥感图像分类；EuroSAT；EfficientFormerV2；数据增强；深度学习；Streamlit

## 第 1 章 绪论

### 1.1 研究背景与意义

遥感影像能够从宏观尺度记录地表覆盖和土地利用状态，是地理信息科学、生态环境监测和智慧城市建设的重要数据来源。随着卫星成像技术和公开遥感数据集的发展，利用深度学习方法自动识别遥感图像中的土地利用类别，已经成为遥感智能处理的重要研究方向。

传统遥感图像分类方法通常依赖人工特征、光谱指数或浅层机器学习算法，对复杂场景的表达能力有限。深度卷积网络和视觉 Transformer 相关模型能够自动学习多层次图像特征，在自然图像与遥感图像分类任务中均取得了较好效果。但是，遥感图像存在视角俯视、尺度变化、类别间纹理相似和类内差异较大等特点，仍需要结合合适的数据增强、迁移学习和评估分析方法。

本文选择 EfficientFormerV2 作为主干模型，是因为该模型兼顾了 Transformer 表达能力和轻量化推理效率，适合用于中小规模遥感图像分类系统。通过在 EuroSAT 数据集上完成训练、评估和演示系统实现，可以形成一个完整的课程实践项目，也为后续进一步改进模型性能提供基础。

### 1.2 研究内容

本文主要完成以下工作：

- 构建 EuroSAT RGB 数据加载、固定随机划分和增强预处理流程。
- 使用 EfficientFormerV2-S0 预训练模型实现 10 类土地利用分类。
- 实现基线训练、增强训练、独立评估、实验汇总和图表生成工具。
- 对比 baseline 与 enhanced 配置在限制 batch 条件下的实验结果。
- 实现命令行单图预测和 Streamlit Web 演示界面。
- 整理开发日志、技术文档、实验结果和报告素材，保证项目过程可追溯。

### 1.3 章节安排

第 1 章介绍研究背景、项目意义和主要工作。第 2 章介绍 EuroSAT 数据集、EfficientFormerV2 模型和增强训练策略。第 3 章说明系统总体设计、模块划分和关键实现。第 4 章给出实验设置、结果对比和类别表现分析。第 5 章介绍 Web 演示系统。最后总结项目成果、不足和后续改进方向。

## 第 2 章 相关技术与数据集

### 2.1 EuroSAT 数据集

EuroSAT 是基于 Sentinel-2 卫星图像构建的遥感土地利用分类数据集。本文使用 RGB 图像版本，包含 10 个类别：

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

项目中使用固定随机种子将数据划分为训练集、验证集和测试集，比例为 70%、15%、15%。当前划分结果为训练集 18900 张、验证集 4050 张、测试集 4050 张。固定划分能够保证不同训练配置之间的结果具有可比性。

### 2.2 EfficientFormerV2 模型

EfficientFormerV2 是一种兼顾效率与表达能力的视觉模型结构。与传统 CNN 相比，它能够利用更强的特征建模能力；与大型视觉 Transformer 相比，它在参数量和推理成本上更加轻量。本项目使用 timm 库中的 `efficientformerv2_s0` 模型，并加载 ImageNet 预训练权重作为迁移学习起点。

模型输入为经过 resize、crop 和 normalize 的 RGB 图像，输出为 10 维类别 logits。项目通过替换分类头，使预训练模型能够适配 EuroSAT 的 10 类土地利用任务。

### 2.3 增强训练策略

本文实现的增强策略包括：

- RandomResizedCrop：提升模型对空间尺度变化的适应能力。
- RandomHorizontalFlip：增强模型对水平翻转场景的鲁棒性。
- ColorJitter：模拟亮度、对比度和颜色变化。
- Mixup：将两张样本及其标签线性混合，缓解过拟合。
- Label Smoothing：降低模型对单一标签的过度置信。

基线配置关闭 Mixup 和 ColorJitter，增强配置启用 Mixup、ColorJitter 和 Label Smoothing，用于形成同规模对比实验。

## 第 3 章 系统设计与模型实现

### 3.1 系统总体结构

项目采用模块化结构组织：

- `configs/`：保存 baseline、enhanced 和 default 配置。
- `src/eurosat_landuse/data.py`：数据集下载、划分、transform 和 dataloader 构建。
- `src/eurosat_landuse/model.py`：EfficientFormerV2 模型构建和参数统计。
- `src/eurosat_landuse/train.py`：训练循环、验证循环、Mixup 和 checkpoint 保存。
- `src/eurosat_landuse/evaluate.py`：加载 checkpoint 并输出整体指标、逐类准确率和混淆矩阵。
- `src/eurosat_landuse/predict.py`：单图 Top-k 预测。
- `src/eurosat_landuse/summarize_experiments.py`：汇总 metrics JSON 为 CSV 和 Markdown。
- `src/eurosat_landuse/plot_experiments.py`：生成准确率、损失、混淆矩阵和每类准确率图。
- `app/streamlit_app.py`：Web 演示界面。

### 3.2 数据处理流程

系统读取 EuroSAT RGB 数据后，先根据配置中的随机种子生成固定划分，再分别为训练、验证和测试阶段构建图像变换。训练阶段可启用随机裁剪、翻转和颜色扰动；验证与测试阶段使用确定性 resize、center crop 和 normalize，以保证评估结果稳定。

### 3.3 模型训练流程

训练入口读取 YAML 配置，构建 dataloader、模型、损失函数和优化器。训练过程中按 epoch 统计训练损失和准确率，并在验证阶段计算验证损失和准确率。当验证准确率优于历史最佳结果时，系统将 checkpoint 保存到 `outputs/checkpoints/`。训练指标保存为 JSON，便于后续汇总和绘图。

### 3.4 模型评估流程

评估脚本从本地 checkpoint 恢复模型，在指定 split 上运行推理，并输出 loss、accuracy、per-class accuracy 和 confusion matrix。该流程独立于训练脚本，能够复核训练时的验证结果，也能在测试集上进行最终评估。

### 3.5 预测与演示流程

命令行预测入口接收单张图像路径和 checkpoint，输出 Top-k 类别及置信度。Streamlit 界面复用同一套预测逻辑，提供 checkpoint 选择、设备选择、Top-k 调整和图像上传功能，便于展示模型实际效果。

## 第 4 章 实验设计与结果分析

### 4.1 实验环境

当前实验环境如下：

| Item | Value |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.12.0 |
| torchvision | 0.27.0 |
| timm | 1.0.27 |
| 运行设备 | Apple MPS |
| 数据集 | EuroSAT RGB |
| 模型 | EfficientFormerV2-S0 |

### 4.2 实验设置

项目设置 baseline 和 enhanced 两组核心配置。baseline 关闭 Mixup 和 ColorJitter；enhanced 启用 Mixup、Label Smoothing 和 ColorJitter。由于本地训练资源有限，当前阶段主要采用限制 batch 的短程实验验证完整方法链路，并使用独立评估脚本复核 checkpoint 表现。

### 4.3 当前实验结果

当前核心结果如下：

| Run | Split | Train Acc | Eval Loss | Eval Acc | Eval Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| quick_baseline | val | 0.0500 | 2.3112 | 0.0625 | 16 |
| medium_baseline_20b | val | 0.3000 | 2.0115 | 0.3563 | 160 |
| enhanced_20b | val | 0.1906 | 2.1020 | 0.2812 | 160 |
| baseline_100b | val | 0.5681 | 6.7928 | 0.7729 | 480 |
| enhanced_100b | val | 0.2944 | 3.1420 | 0.7167 | 480 |
| baseline_100b | test | 0.5681 | 4.7938 | 0.7748 | 4050 |

完整实验结果整理见 `docs/experiment_results.md`。本地可引用图表包括：

- `outputs/figures/experiment_accuracy.png`
- `outputs/figures/experiment_loss.png`
- `outputs/figures/baseline_100b_confusion_matrix.png`
- `outputs/figures/baseline_100b_per_class_accuracy.png`
- `outputs/figures/baseline_100b_test_confusion_matrix.png`
- `outputs/figures/baseline_100b_test_per_class_accuracy.png`
- `outputs/figures/baseline_100b_test_full_confusion_matrix.png`
- `outputs/figures/baseline_100b_test_full_per_class_accuracy.png`
- `outputs/figures/streamlit_demo_ui.png`

### 4.4 结果分析

从实验结果可以看出，随着训练 batch 数增加，模型准确率显著提升。`quick_baseline` 的验证准确率仅为 0.0625，而 `baseline_100b` 的验证准确率达到 0.7729，说明 EfficientFormerV2-S0 能够有效学习 EuroSAT 图像特征。

在 20 batch 和 100 batch 设置下，增强配置的验证准确率均低于基线配置，但 100 batch 增强配置的验证损失明显低于基线配置。这说明 Mixup 和 Label Smoothing 在短训练条件下可能增加优化难度，使准确率提升不明显；同时它们也可能降低模型过度置信，使损失表现更稳定。后续如扩大训练轮数，增强策略仍可能体现更好的泛化价值。

当前最好模型 `baseline_100b_best.pt` 在完整测试集 4050 个样本上达到 0.7748 准确率，与验证集 0.7729 接近，说明模型具有一定泛化能力。从类别表现看，`River`、`Pasture`、`SeaLake`、`PermanentCrop` 和 `Forest` 是测试集上相对较弱的类别，后续可以结合混淆矩阵进一步分析其与相似纹理类别之间的混淆关系。

## 第 5 章 Web 演示系统

### 5.1 系统功能

Web 演示界面基于 Streamlit 实现，主要功能包括：

- 自动扫描本地 `outputs/checkpoints/` 下的 checkpoint。
- 选择推理设备，包括 auto、cpu、mps 和 cuda。
- 上传单张遥感图像。
- 设置 Top-k 输出数量。
- 展示预测类别、置信度和候选类别排序。

### 5.2 运行方式

Web 界面启动命令为：

```bash
streamlit run app/streamlit_app.py
```

界面截图已保存到本地 `outputs/figures/streamlit_demo_ui.png`，可作为报告第 5 章插图素材。由于 `outputs/` 不提交到 GitHub，报告中应使用截图导出文件或自行插入图片。

### 5.3 系统价值

相比只提供训练脚本，Web 演示界面能够更直观地展示模型预测流程，使项目从单纯实验代码扩展为可交互的应用原型。该界面也方便后续进行模型对比、错误样本分析和课程展示。

## 结论与展望

本文完成了一个基于 EfficientFormerV2 与增强训练策略的 EuroSAT 遥感土地利用图像分类系统。系统覆盖数据加载、模型训练、增强策略、独立评估、实验汇总、图表生成、命令行预测和 Web 演示等完整流程。当前实验结果表明，EfficientFormerV2-S0 在短程训练下已经能够取得较好的分类效果，完整测试集准确率达到 0.7748。

项目仍存在一些不足。第一，当前实验主要采用限制 batch 的短训练方式，尚未完成完整 epoch 或多 epoch 训练。第二，增强策略在短训练条件下没有超过基线准确率，需要在更充分训练条件下重新验证。第三，报告中的类别混淆分析还可以结合更多样本可视化进一步展开。

后续改进方向包括：

- 扩大训练规模，运行完整 epoch 或多 epoch 实验。
- 加入学习率调度器并比较不同优化策略。
- 对混淆类别进行样本级可视化分析。
- 增加 Grad-CAM 或注意力可视化，解释模型关注区域。
- 将 Web 界面扩展为多模型对比和批量预测工具。

## 图表与表格清单

| 编号 | 名称 | 来源 |
| --- | --- | --- |
| 图 1 | 项目总体流程图 | 可根据第 3 章模块结构绘制 |
| 图 2 | 数据处理流程图 | `src/eurosat_landuse/data.py` |
| 图 3 | 实验准确率对比图 | `outputs/figures/experiment_accuracy.png` |
| 图 4 | 实验损失对比图 | `outputs/figures/experiment_loss.png` |
| 图 5 | 基线模型混淆矩阵 | `outputs/figures/baseline_100b_confusion_matrix.png` |
| 图 6 | 基线模型完整测试集混淆矩阵 | `outputs/figures/baseline_100b_test_full_confusion_matrix.png` |
| 图 7 | Streamlit 演示界面 | `outputs/figures/streamlit_demo_ui.png` |
| 表 1 | EuroSAT 类别列表 | 第 2 章 |
| 表 2 | 数据集划分结果 | `docs/experiment_results.md` |
| 表 3 | baseline/enhanced 配置对比 | `configs/baseline.yaml`、`configs/enhanced.yaml` |
| 表 4 | 实验结果对比表 | `docs/experiment_results.md` |
