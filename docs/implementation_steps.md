# 执行步骤

## 阶段 1：项目初始化

目标：建立项目目录、文档体系、开发日志和 GitHub 同步规则。

验收标准：

- `README.md` 能指向所有核心文档。
- `docs/` 包含需求、技术设计、开发规范、执行步骤和报告提纲。
- `dev_logs/` 包含模板和当天日志。
- `.gitignore` 能排除课程原件、数据、权重和输出。
- 项目成功推送到 GitHub。

## 阶段 2：代码骨架

目标：建立训练、评估、预测和界面入口，不立即跑完整实验。

验收标准：

- 配置文件可读取。
- Python 模块结构清晰。
- 代码能通过基础语法检查。
- README 补充最小运行命令。

## 阶段 3：EuroSAT 数据加载

目标：实现 EuroSAT RGB 数据读取、划分和增强。

验收标准：

- 能打印类别名称和样本数量。
- 能取出一个 batch。
- 能显示若干样本图像。
- 能通过 `python -m src.eurosat_landuse.data_smoke --config configs/default.yaml --check-only` 检查配置和依赖状态。
- 安装依赖后，能通过 `python -m src.eurosat_landuse.data_smoke --config configs/default.yaml --download --batch-size 8` 读取一个训练 batch。

## 阶段 4：EfficientFormerV2 基线

目标：实现基线训练与评估流程。

验收标准：

- 能通过 `python -m src.eurosat_landuse.train --config configs/default.yaml --smoke-test --download --batch-size 4` 完成单 batch forward/backward。
- 能完成小轮数训练。
- 能保存本地 checkpoint。
- 能输出验证集准确率和损失曲线。

## 阶段 5：增强训练策略

目标：加入增强训练策略并完成对比。

验收标准：

- 基线与增强模型配置可区分。
- 记录准确率、损失和混淆矩阵。
- 能说明增强策略带来的影响。

## 阶段 6：Web 演示界面

目标：实现 Streamlit 上传图像预测界面。

验收标准：

- 能上传单张遥感图像。
- 能输出预测类别、置信度和 Top-3 结果。
- 本地运行无明显界面错误。

## 阶段 7：报告素材整理

目标：整理可用于结题报告的文字、图表和实验结论。

验收标准：

- 报告章节结构完整。
- 实验图表可追溯到具体配置和日志。
- 结论部分能客观说明完成工作与不足。
