# E1：Head-only baseline

## 实验设置

冻结 `dmis-lab/biobert-base-cased-v1.1` 的全部 backbone，仅训练 9 类 typed-BIO 分类头的 6,921 个参数。
使用完整官方 Training（593 篇）和 Validation（100 篇），seed=42，Metal（MPS）。
Batch size=8，AdamW，weight decay=0.01，gradient clipping=1.0；无学习率调度器。
每个学习率独立初始化，最多 30 epochs，验证集 typed exact-span micro F1 连续 3 轮未改善即早停。
512-token 窗口、64-token 重叠；训练时重叠 token 只监督一次。
推理时平均同一全文 token 的窗口 logits，再全局解码；非法 I 转移按 B 修复并统计。

## 验证集选择

| Head learning rate | 实际轮数 | 最佳轮数 | Validation micro F1 | 训练及验证秒数 |
|---|---:|---:|---:|---:|
| 1e-4 | 4 | 1 | 0.00% | 66.1 |
| 5e-4 | 30 | 29 | 39.71% | 500.0 |
| 1e-3 | 24 | 21 | 41.84% | 392.2 |

按 Validation F1 选择 **1e-3**，使用第 **21** 轮 checkpoint。
选择记录先保存到 `results/E1_search/selection.json`，随后才对完整 Test（100 篇）做正式评估。
测试分数没有用于超参数或 checkpoint 选择。此前的 2 篇文档 smoke test 仅用于校验读写及模型重载路径，不参与选型。

## 最终指标

所有指标按严格 typed exact-span entity-level 计算；span-only 列忽略类型。

| Split | Precision | Recall | Micro F1 | Macro F1 | Span-only F1 |
|---|---:|---:|---:|---:|---:|
| Validation | 35.49% | 50.95% | 41.84% | 33.08% | 44.24% |
| Test | 35.96% | 54.69% | 43.39% | 29.54% | 44.55% |

| 类型 | Validation F1 | Test F1 |
|---|---:|---:|
| SpecificDisease | 48.99% | 50.75% |
| DiseaseClass | 23.69% | 18.94% |
| CompositeMention | 17.98% | 2.94% |
| Modifier | 41.67% | 45.51% |

Validation 非法 BIO 转移：443；窗口重叠 token 预测冲突：9。
Test 非法 BIO 转移：616；窗口重叠 token 预测冲突：3。
完整 per-type Precision/Recall/F1、support 和类型混淆矩阵保存在 `results.json`。

## 写入 notebook 时使用

- `experiments/E1/results.json`：最终指标、候选配置、选择依据和源码哈希。
- `experiments/E1/learning_curves.csv`：各学习率的逐轮 loss 和验证指标，可直接画曲线。
- `results/E1_search/1e-3/best_checkpoint/`：最终模型（本地保存，不加入 Git）。
- `results/E1_search/1e-3/validation_predictions.json`：最佳验证逐文档预测。
- `results/E1_selected_test/test_predictions.json`：正式测试逐文档预测。

当前结果仅为 seed=42 的一次运行，不能报告 mean ± std；30 轮上限和早停规则也不保证充分收敛。
`1e-4` 在第 4 轮因连续零 F1 早停，其结果只表示本次预算和早停规则下的表现。
MPS 峰值显存无可用统计，保存为 null。
Notebook 第 5 节已包含可独立复现的 E1 代码，无需外部 `.py` 文件；此报告记录前期脚本实验，原始完整运行文件保留在被 Git 忽略的 `results/` 下。
