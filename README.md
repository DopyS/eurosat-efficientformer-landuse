# EfficientFormerV2 EuroSAT Land Use Classification

本项目题目为：**基于 EfficientFormerV2 与增强训练策略的 EuroSAT 遥感土地利用图像分类系统设计与实现**。

项目目标是在 EuroSAT 遥感图像数据集上构建一个土地利用图像分类系统，先完成 EfficientFormerV2 基线模型，再逐步加入增强训练策略，并最终提供评估结果、可视化图表和 Web 演示界面。

## 项目路径

- [docs/requirements.md](docs/requirements.md)：项目需求、交付物和课程验收要求。
- [docs/technical_design.md](docs/technical_design.md)：技术路线、数据流、模型方案和实验指标。
- [docs/development_standards.md](docs/development_standards.md)：开发规范、实验记录规范和 Git 同步规则。
- [docs/implementation_steps.md](docs/implementation_steps.md)：分阶段开发步骤和验收标准。
- [docs/report_outline.md](docs/report_outline.md)：结题报告章节结构和写作素材规划。
- [dev_logs/README.md](dev_logs/README.md)：开发日志使用说明。
- [dev_logs/2026-06-11.md](dev_logs/2026-06-11.md)：当前开发日志。
- [configs/default.yaml](configs/default.yaml)：默认实验配置。

## 当前阶段

当前处于第二阶段：最小代码骨架建立。

本阶段建立可导入、可检查的代码入口，不下载数据集、不训练模型、不提交课程原始材料。

## 当前可用入口

第二阶段提供最小代码入口，当前可运行的脚本包括：

```bash
python -m src.eurosat_landuse.train --config configs/default.yaml
python -m src.eurosat_landuse.evaluate --config configs/default.yaml
python -m src.eurosat_landuse.predict --config configs/default.yaml --image path/to/image.jpg
python -m src.eurosat_landuse.data_smoke --config configs/default.yaml --check-only
streamlit run app/streamlit_app.py
```

这些入口目前仍属于 scaffold，主要用于确认项目结构、配置读取和后续接线位置。

安装 PyTorch 和 torchvision 后，可以运行真实数据 smoke test：

```bash
python -m src.eurosat_landuse.data_smoke --config configs/default.yaml --download --batch-size 8
```

完成数据 smoke test 后，可以运行单 batch 训练 smoke test：

```bash
python -m src.eurosat_landuse.train --config configs/default.yaml --smoke-test --download --batch-size 4
```

短程 baseline 训练示例：

```bash
python -m src.eurosat_landuse.train --config configs/default.yaml --download --epochs 1 --batch-size 8 --max-train-batches 5 --max-val-batches 2 --run-name quick_baseline
```

增强训练策略对比示例：

```bash
python -m src.eurosat_landuse.train --config configs/enhanced.yaml --download --epochs 1 --batch-size 16 --max-train-batches 20 --max-val-batches 10 --run-name enhanced_20b
```

评估已保存 checkpoint：

```bash
python -m src.eurosat_landuse.evaluate --config configs/default.yaml --checkpoint outputs/checkpoints/quick_baseline_best.pt --split val --batch-size 8 --max-batches 2 --run-name quick_eval_val
```

汇总本地实验结果：

```bash
python -m src.eurosat_landuse.summarize_experiments --config configs/default.yaml
```

生成报告图表：

```bash
python -m src.eurosat_landuse.plot_experiments --config configs/default.yaml
python -m src.eurosat_landuse.plot_experiments --config configs/default.yaml --eval-json outputs/metrics/baseline_100b_eval_val.json --output-prefix baseline_100b
```

## 开发流程

1. 每次开发前查看当天 `dev_logs/YYYY-MM-DD.md`。
2. 按 [docs/implementation_steps.md](docs/implementation_steps.md) 中的阶段推进。
3. 每次开发结束更新当天日志，记录完成事项、关键决策、问题风险和下一步待办。
4. 每完成一个稳定小阶段提交一次 Git commit。
5. Public 仓库只同步代码、文档、配置和开发日志。

## GitHub 同步原则

本仓库为 Public。以下内容默认不上传：

- 课程原始材料，包括题目图片、选题列表 PDF、报告模板 DOCX。
- EuroSAT 数据集本体。
- 训练输出、模型权重、日志缓存和本地环境文件。
- 任何 token、账号、路径隐私或未整理的草稿。

## 后续命令规划

后续代码阶段会逐步补充以下命令：

```bash
python -m src.eurosat_landuse.train --config configs/default.yaml
python -m src.eurosat_landuse.evaluate --config configs/default.yaml
streamlit run app/streamlit_app.py
```

这些命令当前只是接口规划，第一阶段暂不实现完整训练逻辑。
