# 实验结果整理

本文档汇总当前阶段可用于结题报告第 4 章的实验设置、结果表、图表路径和初步分析结论。`outputs/` 下的原始实验产物不提交到 GitHub，但本地可直接查看和引用。

## 实验环境

- Python：3.11
- 深度学习框架：PyTorch 2.12.0
- 图像模型库：timm 1.0.27
- 数据集工具：torchvision 0.27.0
- 运行设备：Apple MPS
- 数据集：EuroSAT RGB
- 模型：EfficientFormerV2-S0
- 类别数：10

## 数据划分

EuroSAT 数据集按固定随机种子划分：

| Split | Samples |
| --- | ---: |
| Train | 18900 |
| Val | 4050 |
| Test | 4050 |

当前阶段为了快速验证方法链路，主要使用限制 batch 的验证实验：

- 20 batch 对比：训练 20 个 batch，验证 10 个 batch。
- 100 batch 对比：训练 100 个 batch，验证 30 个 batch。
- 完整测试集评估：使用当前表现最好的 `baseline_100b_best.pt` 在完整 test split 上评估 4050 个样本。

## 实验配置

| Config | Mixup | Label Smoothing | ColorJitter | Purpose |
| --- | ---: | ---: | ---: | --- |
| `configs/baseline.yaml` | 0.0 | 0.0 | false | 基线训练 |
| `configs/enhanced.yaml` | 0.2 | 0.1 | true | 增强训练策略 |

## 主要实验结果

完整实验汇总可由本地文件查看：

- `outputs/metrics/experiment_summary.csv`
- `outputs/metrics/experiment_summary.md`

当前核心结果如下：

| Run | Split | Train Acc | Eval Loss | Eval Acc | Eval Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| `quick_baseline` | val | 0.0500 | 2.3112 | 0.0625 | 16 |
| `medium_baseline_20b` | val | 0.3000 | 2.0115 | 0.3563 | 160 |
| `enhanced_20b` | val | 0.1906 | 2.1020 | 0.2812 | 160 |
| `baseline_100b` | val | 0.5681 | 6.7928 | 0.7729 | 480 |
| `enhanced_100b` | val | 0.2944 | 3.1420 | 0.7167 | 480 |
| `baseline_100b` | test | 0.5681 | 4.7938 | 0.7748 | 4050 |

## 图表路径

实验对比图：

- `outputs/figures/experiment_accuracy.png`
- `outputs/figures/experiment_loss.png`

100 batch 基线模型分析图：

- `outputs/figures/baseline_100b_confusion_matrix.png`
- `outputs/figures/baseline_100b_per_class_accuracy.png`
- `outputs/figures/baseline_100b_test_confusion_matrix.png`
- `outputs/figures/baseline_100b_test_per_class_accuracy.png`
- `outputs/figures/baseline_100b_test_full_confusion_matrix.png`
- `outputs/figures/baseline_100b_test_full_per_class_accuracy.png`

100 batch 增强模型分析图：

- `outputs/figures/enhanced_100b_confusion_matrix.png`
- `outputs/figures/enhanced_100b_per_class_accuracy.png`

## 初步分析结论

1. 随着训练 batch 数增加，模型准确率明显提升。`quick_baseline` 的验证准确率只有 0.0625，而 `baseline_100b` 达到 0.7729，说明 EfficientFormerV2-S0 能够有效学习 EuroSAT 图像特征。

2. 在相同 20 batch 设置下，增强策略的验证准确率低于基线。`medium_baseline_20b` 的验证准确率为 0.3563，`enhanced_20b` 为 0.2812。这说明 Mixup、ColorJitter 和 Label Smoothing 在极短训练下可能增加优化难度。

3. 在 100 batch 设置下，增强策略的验证准确率仍低于基线，但验证损失更低。`baseline_100b` 的验证准确率为 0.7729，验证损失为 6.7928；`enhanced_100b` 的验证准确率为 0.7167，验证损失为 3.1420。该现象说明增强策略可能改善了模型输出的损失表现或置信度分布，但短训练条件下尚未带来更高的分类准确率。

4. 当前最好模型 `baseline_100b_best.pt` 在完整 test split 的 4050 个样本上达到 0.7748 准确率，接近验证集 0.7729 的表现，说明当前模型具备一定泛化能力。完整测试集上相对薄弱的类别主要包括 `River`、`Pasture`、`SeaLake`、`PermanentCrop` 和 `Forest`。

5. 从基线模型的每类准确率看，`Residential`、`HerbaceousVegetation`、`AnnualCrop` 等类别表现较好，`River`、`SeaLake`、`PermanentCrop` 等类别相对较弱。后续可以通过更多训练轮数、类别级数据增强或混淆类别样本分析进一步提升这些类别的表现。

6. 完整测试集错误分析显示，主要混淆方向包括 `River -> Highway` 146 个、`Forest -> SeaLake` 76 个、`SeaLake -> AnnualCrop` 76 个、`PermanentCrop -> HerbaceousVegetation` 74 个、`Pasture -> Forest` 55 个。该结果可用于报告第 4 章的类别混淆分析。

## 错误分析输出

错误分析 Markdown 可由以下命令生成：

```bash
python3 -m src.eurosat_landuse.analyze_errors --eval-json outputs/metrics/baseline_100b_eval_test_full.json
```

本地输出路径：

- `outputs/metrics/baseline_100b_eval_test_full_error_analysis.md`

典型误分类样本可由以下命令导出：

```bash
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_100b_best.pt --split test --true-class River --predicted-class Highway --limit 12
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_100b_best.pt --split test --true-class Forest --predicted-class SeaLake --limit 12
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_100b_best.pt --split test --true-class PermanentCrop --predicted-class HerbaceousVegetation --limit 12
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_100b_best.pt --split test --true-class Pasture --predicted-class Forest --limit 12
```

本地输出路径：

- `outputs/error_samples/test_River_to_Highway/index.md`
- `outputs/error_samples/test_River_to_Highway/contact_sheet.png`
- `outputs/error_samples/test_Forest_to_SeaLake/contact_sheet.png`
- `outputs/error_samples/test_PermanentCrop_to_HerbaceousVegetation/contact_sheet.png`
- `outputs/error_samples/test_Pasture_to_Forest/contact_sheet.png`

## 后续实验建议

- 扩大训练规模，例如 300 batch 或完整 1 epoch，对比 baseline 和 enhanced 是否出现更稳定差异。
- 引入学习率调度器，如 CosineAnnealingLR。
- 保存并绘制完整训练曲线。
- 在更充分训练后继续对完整 test split 运行最终评估，避免只依据验证集结论。
- 针对混淆较多类别做样本可视化分析。
