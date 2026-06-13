# 提交前检查清单

本文档用于正式提交课程材料、同步 GitHub 或整理报告前检查项目状态。

## 1. GitHub 仓库检查

- 确认远程仓库地址为 `https://github.com/DopyS/eurosat-efficientformer-landuse.git`。
- 确认仓库为 Public 时，不包含课程原件、数据集、模型权重、实验输出和隐私信息。
- 提交前运行：

```bash
git status --short
git check-ignore -v 题目要求.jpeg 利兹2023级生产实习学生选题列表.pdf 工程实习结题报告模板.docx
git check-ignore -v data/eurosat outputs/checkpoints outputs/figures outputs/metrics
```

## 2. 代码功能检查

基础语法检查：

```bash
python3 -m compileall src app
```

数据读取检查：

```bash
python3 -m src.eurosat_landuse.data_smoke --config configs/default.yaml --check-only
python3 -m src.eurosat_landuse.data_smoke --config configs/default.yaml --download --batch-size 8
```

训练 smoke test：

```bash
python3 -m src.eurosat_landuse.train --config configs/default.yaml --smoke-test --download --batch-size 4
```

单图预测检查：

```bash
python3 -m src.eurosat_landuse.predict --config configs/default.yaml --checkpoint outputs/checkpoints/baseline_300b_best.pt --image path/to/image.jpg --top-k 3
```

Web 界面检查：

```bash
streamlit run app/streamlit_app.py
```

## 3. 实验结果检查

当前报告可引用的核心结果：

| Run | Split | Eval Loss | Eval Acc | Eval Samples |
| --- | --- | ---: | ---: | ---: |
| `baseline_100b` | test | 4.7938 | 0.7748 | 4050 |
| `baseline_300b` | val | 0.1958 | 0.9443 | 1920 |
| `enhanced_300b` | val | 0.8644 | 0.8620 | 1920 |
| `baseline_300b` | test | 0.2163 | 0.9356 | 4050 |

重新生成实验汇总：

```bash
python3 -m src.eurosat_landuse.summarize_experiments --config configs/default.yaml
```

重新生成实验图表：

```bash
python3 -m src.eurosat_landuse.plot_experiments --config configs/default.yaml
python3 -m src.eurosat_landuse.plot_experiments --config configs/default.yaml --eval-json outputs/metrics/baseline_300b_eval_test_full.json --output-prefix baseline_300b_test_full
```

重新生成错误分析：

```bash
python3 -m src.eurosat_landuse.analyze_errors --eval-json outputs/metrics/baseline_300b_eval_test_full.json
```

导出典型误分类样本：

```bash
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_300b_best.pt --split test --true-class PermanentCrop --predicted-class HerbaceousVegetation --limit 12
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_300b_best.pt --split test --true-class Highway --predicted-class River --limit 12
python3 -m src.eurosat_landuse.export_errors --config configs/baseline.yaml --checkpoint outputs/checkpoints/baseline_300b_best.pt --split test --true-class Pasture --predicted-class Forest --limit 12
```

## 4. 报告材料检查

核心报告文档：

- `docs/final_report_draft.md`：正式报告 Markdown 初稿。
- `docs/report_chapter_drafts.md`：章节草稿与素材池。
- `docs/experiment_results.md`：实验结果、图表路径和分析结论。
- `docs/report_outline.md`：报告提纲。

需要从本地插入报告模板的图片：

- `outputs/figures/experiment_accuracy.png`
- `outputs/figures/experiment_loss.png`
- `outputs/figures/baseline_300b_test_full_confusion_matrix.png`
- `outputs/figures/baseline_300b_test_full_per_class_accuracy.png`
- `outputs/figures/streamlit_demo_ui.png`
- `outputs/error_samples/baseline_300b_test_PermanentCrop_to_HerbaceousVegetation/contact_sheet.png`
- `outputs/error_samples/baseline_300b_test_Highway_to_River/contact_sheet.png`
- `outputs/error_samples/baseline_300b_test_Pasture_to_Forest/contact_sheet.png`

注意：`outputs/` 目录不会上传 GitHub，正式提交报告时应将需要的图表插入 DOCX/PDF，而不是依赖 GitHub 路径。

## 5. 建议提交材料

推荐提交：

- 整理后的结题报告 DOCX/PDF。
- GitHub 仓库链接。
- 项目源码压缩包，需排除 `data/`、`outputs/`、模型权重和课程原始材料。
- 若老师要求演示，可本地保留 `outputs/checkpoints/baseline_300b_best.pt` 和 EuroSAT 数据集。

不建议提交：

- `data/` 或 `datasets/`。
- `outputs/` 原始产物。
- `*.pt`、`*.pth`、`*.ckpt`、`*.onnx`、`*.safetensors`。
- `.env`、缓存文件、虚拟环境目录。
- 未整理的课程原始文件。

## 6. 当前不足说明

如果在报告答辩或说明中被问到实验限制，可以客观说明：

- 当前实验以限制 batch 的短程训练为主，主要用于验证系统链路和初步效果。
- 增强策略在短训练下准确率未超过基线，但验证损失更低，后续需要完整 epoch 或多 epoch 训练进一步确认。
- 测试集评估已覆盖完整 test split 的 4050 个样本，但训练本身仍是限制 batch 的短程训练，最终论文级结论建议继续扩大训练轮数。
