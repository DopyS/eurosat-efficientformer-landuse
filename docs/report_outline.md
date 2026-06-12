# 结题报告提纲

## 摘要

说明遥感土地利用分类的背景、本文使用的 EfficientFormerV2 模型、增强训练策略、实验结果和系统实现。

## 第 1 章 绪论

- 研究背景与意义
- 遥感图像分类应用价值
- 国内外研究现状
- 本文主要工作
- 章节安排

## 第 2 章 相关技术与数据集

- 遥感图像分类任务介绍
- EuroSAT 数据集介绍
- EfficientFormerV2 模型介绍
- 迁移学习方法
- 数据增强与正则化方法

## 第 3 章 系统设计与模型实现

- 项目总体架构
- 数据处理流程
- 模型构建流程
- 训练流程
- 评估流程

## 第 4 章 实验设计与结果分析

- 实验环境
- 数据划分
- 基线模型实验
- 增强训练策略实验
- 对比分析
- 混淆矩阵和类别表现分析

当前可参考 [experiment_results.md](experiment_results.md) 中的实验结果整理、图表路径和初步结论。

正文草稿可参考 [report_chapter_drafts.md](report_chapter_drafts.md)，其中已经整理摘要、第 1-5 章、结论与图表清单。

正式报告 Markdown 初稿见 [final_report_draft.md](final_report_draft.md)，可作为迁移到学校 DOCX 模板前的主要文字版本。提交前检查见 [submission_checklist.md](submission_checklist.md)。

## 第 5 章 Web 演示系统

- 系统功能设计
- 图像上传流程
- 预测结果展示
- 系统运行示例

当前界面入口为 `app/streamlit_app.py`，可通过 `streamlit run app/streamlit_app.py` 启动。

## 结论

- 总结已经完成的工作。
- 总结模型表现和增强策略效果。
- 说明不足与后续改进方向。

## 图表建议

- EuroSAT 样本类别示例图。
- 项目总体流程图。
- EfficientFormerV2 分类流程图。
- 训练损失曲线。
- 验证准确率曲线。
- 混淆矩阵。
- Web 界面截图。
- 实验对比表，可由 `outputs/metrics/experiment_summary.csv` 整理得到。
- 实验对比图，可由 `outputs/figures/experiment_accuracy.png` 和 `outputs/figures/experiment_loss.png` 整理得到。
- 每类准确率图和混淆矩阵图，可由指定评估 JSON 生成。
