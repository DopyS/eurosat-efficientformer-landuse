# 开发规范

## 目录规范

- `configs/`：保存实验配置。
- `src/`：保存训练、评估、模型和工具代码。
- `app/`：保存 Streamlit 演示界面代码。
- `docs/`：保存项目需求、技术设计、执行步骤和报告素材。
- `dev_logs/`：保存每日开发记录。
- `outputs/`：保存本地实验结果，不提交到 GitHub。

## 代码规范

- 代码优先保持简单、可读、可解释。
- 函数命名使用小写加下划线。
- 配置参数尽量放入 `configs/default.yaml`。
- 关键训练参数必须能从配置文件追溯。
- 复杂逻辑需要写简短注释，避免无意义注释。

## 实验记录规范

每次实验至少记录：

- 日期
- Git commit
- 配置文件
- 模型名称
- 数据划分
- 训练轮数
- 主要增强策略
- 验证集准确率
- 测试集准确率
- 重要问题或异常

## Git 规范

Public 仓库不得提交：

- 课程原始材料
- 数据集
- 模型权重
- 训练输出
- 本地环境文件
- 密钥或账号信息

建议提交信息格式：

- `chore: initialize project structure and documentation`
- `docs: update development plan`
- `feat: add eurosat dataset loader`
- `feat: add efficientformer baseline`
- `feat: add enhanced training strategy`
- `feat: add streamlit demo`
- `fix: correct metric calculation`

## 开发日志规范

每次开发结束更新当天日志：

- 今日完成
- 关键决策
- 问题与风险
- 下一步待办
- 验证记录

