# COMP9444 Biomedical Disease NER

当前已完成数据管道、统一评估，以及 Notebook 中的 **E1 Head-only baseline** 和 **E2 Partial fine-tuning**。E3–E6 待完成。
最终仅提交 `MedNLP_research.ipynb`。运行方式见[第 14 节](#14-notebook-运行与提交)。

## 1. 项目目标

本项目使用 **BioBERT** 在 **NCBI Disease Corpus** 上完成生物医学疾病命名实体识别（Named Entity Recognition, NER）。模型的输入是 PubMed 标题和摘要，输出是原文中每个疾病 mention 的精确字符范围和类型。

```text
Input:
The patient was diagnosed with hereditary hemochromatosis.

Output:
hereditary hemochromatosis -> [start, end), SpecificDisease
```

项目不再进行 BioBERT 与 PubMedBERT 的跨模型对比。所有正式实验使用同一个 BioBERT backbone，研究两个问题：

1. 不同参数更新范围如何影响 NER 性能、成本和过拟合？
2. 使用不同 MLM masking 策略进行任务自适应预训练后，是否能改善 NER 性能？

---

## 2. 数据集与标签

NCBI Disease Corpus 包含 793 篇 PubMed 文档，使用官方划分：

| Split | 文档数 | 实体数 | WordPiece token 数 | 需多窗口的文档数 |
|---|---:|---:|---:|---:|
| Training | 593 | 5,145 | 184,552 | 24 |
| Validation | 100 | 787 | 32,132 | 4 |
| Testing | 100 | 960 | 33,092 | 3 |

Token 数不含 `[CLS]` / `[SEP]`；512-token 窗口最多容纳 510 个正文 token。

| Mention type | Training | Validation | Testing |
|---|---:|---:|---:|
| SpecificDisease | 2,972 | 412 | 555 |
| DiseaseClass | 769 | 126 | 121 |
| CompositeMention | 115 | 35 | 20 |
| Modifier | 1,289 | 214 | 264 |

原文统一按以下方式构造：

```python
text = title + " " + abstract
```

实体字符区间使用零起点的半开区间 `[start, end)`。任何清洗、分词和滑动窗口都不能改变这套全局字符坐标。

### 2.1 Typed-BIO 标签

当前实现保留数据集中的四种 mention type，使用 9 类 typed-BIO 标签：

```text
O
B-SpecificDisease / I-SpecificDisease
B-DiseaseClass / I-DiseaseClass
B-CompositeMention / I-CompositeMention
B-Modifier / I-Modifier
```

`[CLS]`、`[SEP]`、padding 和不参与监督的 token 标签必须为 `-100`，不能当作 `O` 参与 loss。

### 2.2 数据目录

```text
Dataset/
├── RawData/
├── ProcessedData/
│   ├── NCBI_training.jsonl
│   ├── NCBI_validation.jsonl
│   └── NCBI_testing.jsonl
└── TokenizedData/
    └── BioBERT/
        ├── NCBI_training.jsonl
        ├── NCBI_validation.jsonl
        └── NCBI_testing.jsonl
```

正式 NER 实验使用 `Dataset/TokenizedData/BioBERT/`。原文、token、`input_ids`、`attention_mask` 和 `offset_mapping` 必须按下标对齐；Training 的 `bio_tags` 同样逐 token 对齐。Validation/Testing 的分词缓存不保存 gold 标签，评估时从 `ProcessedData` 读取标注。

MLM 阶段只能使用 Training 文本。Standard MLM 和 Whole-word MLM 不使用实体标签；Entity-aware MLM 可使用 Training 中的实体 offset。

---

## 3. 总体实验设计

项目共包含 **6 组核心实验**：

| ID | 实验 | MLM 适应 | NER 参数更新范围 |
|---|---|---|---|
| E1 | Head-only baseline | 无 | 只训练 NER classification head |
| E2 | Partial fine-tuning | 无 | 最后 4 个 Transformer 层 + head |
| E3 | Full fine-tuning | 无 | 全部 BioBERT 参数 + head |
| E4 | Standard MLM + Full NER | Random-token TAPT | 全部 BioBERT 参数 + head |
| E5 | Whole-word MLM + Full NER | Whole-word TAPT | 全部 BioBERT 参数 + head |
| E6 | Entity-aware MLM + Full NER | Entity-span label-aware TAPT | 全部 BioBERT 参数 + head |

E1–E3 回答“训练多少层”的问题；E3–E6 回答“MLM 适应和 masking 策略是否有效”的问题。E3 是两类比较共享的 no-adaptation 对照组，因此总数是 6 组而不是 7 组。

---

## 4. 六组实验的训练流程

### E1. Head-only baseline

```text
原始 BioBERT
    ↓
冻结 embedding 和全部 Transformer 层
    ↓
添加 typed-BIO token classification head
    ↓
只训练 classification head
```

```text
Embedding:                frozen
Transformer Layer 0–11:   frozen
NER classification head:  trainable
```

原始 BioBERT 没有针对本项目训练好的 typed-BIO 输出层，因此“不训练 baseline”不能字面理解为所有参数都不更新。E1 不更新 BioBERT backbone，但必须训练新增的分类头。这种设置也称为 linear probing。

### E2. Partial fine-tuning

```text
原始 BioBERT
    ↓
冻结 embedding 和 Layer 0–7
    ↓
解冻 Layer 8–11
    ↓
训练最后 4 层和 classification head
```

```text
Embedding:                frozen
Transformer Layer 0–7:    frozen
Transformer Layer 8–11:   trainable
NER classification head:  trainable
```

核心实验固定解冻最后 4 层。如果尝试其他层数，只能作为额外消融实验，不能在看过 Test 结果后更改。

### E3. Full fine-tuning

```text
原始 BioBERT
    ↓
添加 typed-BIO token classification head
    ↓
训练 embedding、Layer 0–11 和 classification head
```

```text
Embedding:                trainable
Transformer Layer 0–11:   trainable
NER classification head:  trainable
```

E3 是标准 full fine-tuning，同时是 E4–E6 的 no-adaptation 对照组。

### E4. Standard Random-Token MLM + Full NER

```text
原始 BioBERT
    ↓
NCBI Training 文本
    ↓
随机选择约 15% subword token 进行 MLM
    ↓
保存 Standard-MLM adapted checkpoint
    ↓
替换 MLM head 为 typed-BIO NER head
    ↓
Full NER fine-tuning
```

E4 不使用实体标签。MLM 阶段的目标是恢复被 mask 的 token，NER 阶段才学习 typed-BIO 标签。

### E5. Whole-Word MLM + Full NER

```text
原始 BioBERT
    ↓
NCBI Training 文本
    ↓
随机选择约 15% word，屏蔽每个 word 的全部 WordPiece
    ↓
保存 Whole-Word-MLM adapted checkpoint
    ↓
替换 MLM head 为 typed-BIO NER head
    ↓
Full NER fine-tuning
```

例如：

```text
hemo  ##ch  ##rom  ##atosis
```

Whole-word masking 会同时屏蔽所有 subword，而不是只屏蔽其中一个。E5 不使用实体标签。

### E6. Entity-Aware Span MLM + Full NER

```text
原始 BioBERT
    ↓
NCBI Training 文本 + Training entity offsets
    ↓
优先屏蔽完整疾病实体 span，其余 mask 位置随机选择
    ↓
保存 Entity-Aware-MLM adapted checkpoint
    ↓
替换 MLM head 为 typed-BIO NER head
    ↓
Full NER fine-tuning
```

推荐保持总 mask budget 约为 15%：

```text
约 50% mask budget 来自完整疾病 span
约 50% mask budget 来自随机非实体 word
```

没有实体的文本退化为 Whole-word MLM。E6 使用了 Training 实体 offset，因此应称为 **label-aware task adaptation**，不能宣称它是完全无监督的领域适应。

---

## 5. 公平比较规则

### 5.1 E1–E3：参数更新范围

E1–E3 可以针对各自的参数规模，在 Validation 上选择合理 learning rate。

| 实验 | Encoder learning rate | Head learning rate |
|---|---:|---:|
| E1 Head-only | — | `5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3` |
| E2 Partial | `1e-5, 2e-5, 3e-5` | `5e-5, 1e-4, 2e-4` |
| E3 Full | `1e-5, 2e-5, 3e-5` | `1e-4` |

调参只能使用 Validation，不能使用 Test。

### 5.2 E3–E6：MLM masking 策略

E3–E6 的 NER 阶段必须使用完全相同的：

- full fine-tuning 方式；
- NER learning rate 和 head learning rate；
- batch size 与 gradient accumulation；
- epoch 上限和 early stopping；
- max length 与 sliding-window stride；
- random seeds；
- checkpoint 选择规则；
- typed-BIO 解码和评估函数。

E4–E6 的 MLM 阶段也应使用相同的训练预算。唯一有意变量是 masking strategy。

推荐初始配置：

```yaml
mlm_probability: 0.15
block_size: 256
learning_rate: 5e-5
effective_batch_size: 32
epochs: 10
weight_decay: 0.01
warmup_ratio: 0.1
```

每个 MLM 实验都必须从同一个原始 BioBERT checkpoint 独立开始。禁止用以下累积流程作为三者对比：

```text
Standard MLM -> Whole-word MLM -> Entity-aware MLM
```

否则后一种方法会继承前面的训练，无法归因。

### 5.3 MLM 内部划分

MLM 只能在 NCBI Training 内部按文档划分：

```text
90% Training documents -> MLM train
10% Training documents -> MLM validation
```

Training 中重复的 PMID `8528200` 必须去重，或保证两条记录在同一个 MLM split，不能分别进入 MLM train 和 MLM validation。

MLM 训练完成后，NER 阶段仍可以使用完整的官方 593 条 Training 记录。

### 5.4 数据泄漏规则

```text
Training   -> 参数训练、MLM 适应、数据统计
Validation -> 超参数、early stopping、checkpoint 选择
Testing    -> 配置完全冻结后的最终评估
```

不能使用 Validation/Testing 文本进行 MLM，即使 MLM 没有使用实体标签也不允许。

---

## 6. 长文本与滑动窗口

BioBERT 最长通常处理 512 个 token。长文档不能直接永久截断，需要使用重叠滑动窗口。

```yaml
max_length: 512
stride: 64
```

规则：

- 窗口内 offset 仍然使用全文全局字符坐标；
- 重叠窗口先按全局 token 位置聚合 logits；
- 聚合完成后再进行一次全局 typed-BIO 解码；
- 禁止各窗口独立解码后直接拼接实体；
- 必须记录窗口边界冲突和非法 BIO 转移。

---

## 7. 评估协议

正式主指标是严格的 **typed exact-span entity-level Micro Precision, Recall 和 F1**。预测只有在 `start`、`end` 和 `type` 都与 gold 相同时才记为 True Positive。边界正确但类型错误时，记为一个 False Positive 和一个 False Negative。

正式结果必须报告：

- typed exact-span Micro Precision/Recall/F1；
- typed Macro F1；
- 四种 mention type 各自的 Precision/Recall/F1；
- 不考虑 type 的 exact-span F1；
- type confusion matrix；
- 非法 BIO 转移数；
- 训练时间、可训练参数数量和最高显存占用。

Token accuracy 只能作为诊断指标，不能作为主要结论或 checkpoint 选择依据。

### 7.1 统一评估入口

`MedNLP_research.ipynb` 的 `4. Standardized Testing` 章节提供统一入口：

```python
metrics = evaluate_model_predictions(
    prediction_records,
    "validation",  # 或 "test"
)
```

每篇文档的预测格式：

```python
{
    "record_id": "NCBI_testing-0000",
    "pmid": "9949209",
    "entities": [
        {"start": 346, "end": 360, "type": "SpecificDisease"},
        {"start": 512, "end": 527, "type": "DiseaseClass"},
    ],
}
```

每篇 Validation/Testing 文档必须恰好出现一次；没有预测到实体时使用 `"entities": []`。

---

## 8. MLM 结果的解读

Standard MLM、Whole-word MLM 和 Entity-aware MLM 的预测难度不同，因此 MLM validation loss 不宜直接横向比较。`Entity-aware MLM loss > Standard MLM loss` 不能直接证明 Entity-aware MLM 更差。

MLM loss 只用于：

- 检查同一 masking 方法是否收敛；
- 在同一方法内选择 checkpoint；
- 识别过拟合。

三种 masking 策略的最终优劣必须通过后续 NER 的统一实体级指标判断。

---

## 9. 最终结果表

| ID | Method | Encoder trainable scope | Adaptation | Precision (%) | Recall (%) | Micro F1 (%) | Macro F1 (%) | Train (s) |
|---|---|---|---|---:|---:|---:|---:|---:|
| E1 | Head-only | 0 Transformer layers | None | 36.27 ± 1.22 | 54.65 ± 1.46 | 43.60 ± 1.32 | 29.41 ± 1.32 | 21.19 ± 0.40 |
| E2 | Partial | Last 4 layers | None | 66.95 ± 2.96 | 78.72 ± 2.32 | 72.35 ± 2.60 | 59.96 ± 2.68 | 18.24 ± 2.66 |
| E3 | Full | All layers | None | — | — | — | — | — |
| E4 | Standard MLM + Full | All layers | Random-token TAPT | — | — | — | — | — |
| E5 | WWM + Full | All layers | Whole-word TAPT | — | — | — | — | — |
| E6 | Entity MLM + Full | All layers | Entity-aware TAPT | — | — | — | — | — |

E1 和 E2 来自 2026-11-11 在 WSL（RTX 5090、CUDA）上完成的同一组 Notebook 运行。表中为种子 42、9444、2026 的均值 ± 样本标准差。Precision、Recall、Micro F1 和 Macro F1 均为 Testing 的严格 typed exact-span 指标。Train 仅统计最终三次种子训练的训练时间，单位为秒，不含调参、验证和测试。

### 9.1 实验配置与 Validation 选型

两组使用 BioBERT、AdamW、cross-entropy、恒定学习率、batch size 8、weight decay 0.01 和 dropout 0.1。最多训练 30 轮，早停 patience 为 3。滑窗长度 512，重叠参数 stride 为 64。

| 实验 | 训练范围 | 最终 Encoder LR | 最终 Head LR | Validation Micro F1 (%) | 运行记录 |
|---|---|---:|---:|---:|---|
| E1 | 仅分类头 | — | 7.5e-4 | 41.44 ± 0.48 | [配置](results/completed/5.%20fine-tune/5.1%20E1/run_20261111_162541_367808/selected_config.json) · [最终指标](results/completed/5.%20fine-tune/5.1%20E1/run_20261111_162541_367808/5.1.5%20Seeds%20and%20final%20results/final_results.csv) |
| E2 | Transformer 层 8–11 + 分类头 | 3e-05 | 5e-05 | 71.72 ± 0.41 | [配置](results/completed/5.%20fine-tune/5.2%20E2/run_20261111_163330_353457/selected_config.json) · [最终指标](results/completed/5.%20fine-tune/5.2%20E2/run_20261111_163330_353457/5.2.3%20Seeds%20and%20final%20results/final_results.csv) |

E1 在 Validation 上依次选择 head learning rate、dropout、loss 和 scheduler，并根据 batch size 搜索结果决定是否继续搜索 weight decay。E2 使用本次 E1 选出的 head learning rate 搜索 encoder learning rate，再固定 encoder learning rate 搜索 head learning rate。两组均在锁定配置并选出三个种子的 checkpoint 后运行 Testing。

### 9.2 各随机种子的最终指标

| Seed | E1 Validation Micro F1 (%) | E1 Testing Micro F1 (%) | E2 Validation Micro F1 (%) | E2 Testing Micro F1 (%) |
|---:|---:|---:|---:|---:|
| 42 | 41.17 | 42.29 | 71.36 | 69.52 |
| 9444 | 42.00 | 44.93 | 72.16 | 74.64 |
| 2026 | 41.16 | 43.57 | 71.63 | 72.89 |

本轮 E2 的 Testing Micro F1 均值比 E1 高 **28.75 个百分点**。分类型结果见 [E1](results/completed/5.%20fine-tune/5.1%20E1/run_20261111_162541_367808/5.1.5%20Seeds%20and%20final%20results/per_type_results.csv) 和 [E2](results/completed/5.%20fine-tune/5.2%20E2/run_20261111_163330_353457/5.2.3%20Seeds%20and%20final%20results/per_type_results.csv)。

---

## 10. 每次实验必须保存

- Experiment ID；
- BioBERT checkpoint 名称；
- Git commit 或代码版本；
- 完整超参数；
- 参数冻结/解冻配置；
- masking strategy（E4–E6）；
- 随机种子；
- 训练设备、训练时间和显存使用；
- Validation 全部指标；
- 最佳 checkpoint 和选择原因；
- 配置冻结后的 Testing 全部指标；
- 逐文档预测，用于错误分析；
- 失败、OOM 或中断原因会在终端或 Notebook 中显示。

E1、E2 和 Notebook 只保留完成训练、验证及最终 Testing 的正式运行结果。异常或 Ctrl+C 会清理本次输出，smoke test 和仅验证的运行也不会保留结果。运行期间的文件由 Git 忽略，进程被强制终止后残留的文件仍会被忽略。正式报告可从 `results/completed/` 加入 Git。模型 checkpoint 仅保存在本地。

Notebook 的 E1 和 E2 运行目录分别位于 `results/completed/5. fine-tune/5.1 E1/` 和 `results/completed/5. fine-tune/5.2 E2/`。每次运行包含协议、选定配置、各搜索阶段及最终种子结果。最终结果目录包含 `summary.json`、`final_results.csv`、分类型结果和逐种子的 Testing 指标及预测。


---

## 11. 实施顺序

```text
1. 冻结数据、typed-BIO label map 和统一评估函数
2. 实现共享 BioBERT NER 训练器
3. 完成 E1 Head-only
4. 完成 E2 Partial
5. 完成 E3 Full
6. 实现共享 MLM 训练器和三种 masking collator
7. 独立生成 E4、E5、E6 的 adapted checkpoints
8. 使用与 E3 相同的 NER 配置完成 E4、E5、E6
9. 各方法配置冻结后运行 Testing（E1、E2 已完成），不得基于 Test 回头调参
10. 汇总性能、训练成本、稳定性和错误分析
```

---

## 12. 最终完成标准

- [x] 可重复运行的 BioBERT NER 数据管道；
- [x] 固定的 9 类 typed-BIO label map；
- [x] 滑动窗口、重叠 logits 聚合和全局 BIO 解码；
- [x] E1 Head-only baseline；
- [x] E2 Partial fine-tuning；
- [ ] E3 Full fine-tuning；
- [ ] E4 Standard Random-Token MLM + Full NER；
- [ ] E5 Whole-Word MLM + Full NER；
- [ ] E6 Entity-Aware Span MLM + Full NER；
- [x] 严格 typed exact-span Precision、Recall 和 F1；
- [ ] 公平对比的超参数、训练预算和随机种子；
- [ ] 训练时间、可训练参数和显存成本对比；
- [ ] 错误分析和实验限制；
- [ ] 所有配置、checkpoint、seed 和指标可追踪。

---

## 13. 总结

> 本项目在单一 BioBERT backbone 上，先比较 Head-only、Partial 和 Full fine-tuning 三种参数更新范围，再在固定 Full NER 配置下比较 Standard Random-Token、Whole-Word 和 Entity-Aware Span 三种 MLM 任务适应策略，最终使用统一的严格 typed exact-span 实体级指标评估性能、稳定性和计算成本。

## 14. Notebook 运行与提交

1. 打开 `MedNLP_research.ipynb`，选择项目的 Python 内核。本次 WSL 运行使用 Python 3.12、PyTorch 2.9.0+cu128 和 Transformers 5.3.0。
2. 从上到下运行单元格。首次运行需要联网安装缺失依赖和获取模型或数据。
3. 正式运行保留 `SMOKE_TEST = False`，依次完成 E1 和 E2。E2 从同一 Notebook 会话中的 E1 选型结果读取配置。
4. 完成后保存 Notebook。正式结果可随 Notebook 提交，结果路径见第 10 节。

Notebook 内联实现 E1 和 E2。较早的独立 E1 运行见 [E1 实验报告](experiments/E1/README.md)。
