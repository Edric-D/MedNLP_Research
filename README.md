# COMP9444 Biomedical Disease NER

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

| Split | 文档数 | 实体数 |
|---|---:|---:|
| Training | 593 | 5,145 |
| Validation | 100 | 787 |
| Testing | 100 | 960 |

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

正式 NER 实验使用 `Dataset/TokenizedData/BioBERT/`。原文、token、`input_ids`、`attention_mask`、`offset_mapping` 和 `bio_tags` 必须按下标对齐。

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
| E1 Head-only | — | `1e-4, 5e-4, 1e-3` |
| E2 Partial | `1e-5, 2e-5, 3e-5` | `1e-4` |
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

| ID | Method | Encoder trainable scope | Adaptation | Precision | Recall | Micro F1 | Macro F1 | Time |
|---|---|---|---|---:|---:|---:|---:|---:|
| E1 | Head-only | 0 Transformer layers | None |  |  |  |  |  |
| E2 | Partial | Last 4 layers | None |  |  |  |  |  |
| E3 | Full | All layers | None |  |  |  |  |  |
| E4 | Standard MLM + Full | All layers | Random-token TAPT |  |  |  |  |  |
| E5 | WWM + Full | All layers | Whole-word TAPT |  |  |  |  |  |
| E6 | Entity MLM + Full | All layers | Entity-aware TAPT |  |  |  |  |  |

如果计算资源允许，重要结果使用至少 3 个随机种子，例如 `42, 43, 44`，并报告 `mean F1 ± standard deviation`。

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
- 失败、OOM、中断或其他异常说明。

建议结果文件命名：

```text
results/E1_head_only_metrics.json
results/E2_partial_metrics.json
results/E3_full_metrics.json
results/E4_standard_mlm_metrics.json
results/E5_whole_word_mlm_metrics.json
results/E6_entity_aware_mlm_metrics.json
```

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
9. 冻结全部方法选择后，统一运行 Testing
10. 汇总性能、训练成本、稳定性和错误分析
```

---

## 12. 最终完成标准

- [ ] 可重复运行的 BioBERT NER 数据管道；
- [ ] 固定的 9 类 typed-BIO label map；
- [ ] 滑动窗口、重叠 logits 聚合和全局 BIO 解码；
- [ ] E1 Head-only baseline；
- [ ] E2 Partial fine-tuning；
- [ ] E3 Full fine-tuning；
- [ ] E4 Standard Random-Token MLM + Full NER；
- [ ] E5 Whole-Word MLM + Full NER；
- [ ] E6 Entity-Aware Span MLM + Full NER；
- [ ] 严格 typed exact-span Precision、Recall 和 F1；
- [ ] 公平对比的超参数、训练预算和随机种子；
- [ ] 训练时间、可训练参数和显存成本对比；
- [ ] 错误分析和实验限制；
- [ ] 所有配置、checkpoint、seed 和指标可追踪。

---

## 13. 总结

> 本项目在单一 BioBERT backbone 上，先比较 Head-only、Partial 和 Full fine-tuning 三种参数更新范围，再在固定 Full NER 配置下比较 Standard Random-Token、Whole-Word 和 Entity-Aware Span 三种 MLM 任务适应策略，最终使用统一的严格 typed exact-span 实体级指标评估性能、稳定性和计算成本。

## 独立运行 E1

`e1_head_only.py` 冻结 BioBERT 全部 backbone 参数，只训练 9 类 typed-BIO 分类头。
脚本读取 notebook 生成的 `Dataset/TokenizedData/BioBERT/NCBI_*.jsonl`，
验证与测试 gold 标注单独从 `Dataset/ProcessedData/` 读取。
使用 512-token 窗口、64-token 重叠；训练时重叠 token 只监督一次，
推理时先平均全文同位置的 logits，再统一解码。非法 `I` 转移按 `B` 修复并计数。

使用已有 Conda 环境运行：

```bash
conda run -n cs9517 python e1_head_only.py
```

默认最多 30 epochs、batch size 8、head learning rate `5e-4`、seed 42，
连续 3 epochs 验证集 typed exact-span micro F1 未改善则停止。
设备自动按 Metal（MPS）→ CUDA → CPU 选择。
默认只读 Training 和 Validation，最佳 checkpoint 和指标保存在
`results/E1_head_only/`，包含配置、Git 版本、训练历史、验证预测和运行摘要。
每次运行需要使用空的输出目录，避免覆盖已有实验。
MPS 没有与 CUDA 对等的峰值显存统计，此项保存为 `null`。

先做小规模检查（结果不能作为正式 E1 指标）：

```bash
conda run -n cs9517 python e1_head_only.py --epochs 1 --limit-docs 2 --output-dir results/E1_smoke
```

配置冻结后，可在正式运行中显式加入 `--test`，对验证集选出的最佳模型进行测试评估。
学习率搜索使用 `--learning-rate 1e-4`、`5e-4`、`1e-3`，
各自指定不同 `--output-dir`，仅根据 Validation 结果选择。

```bash
conda run -n cs9517 python -m unittest test_e1 -v
```

最终选好学习率后，无需重新训练即可加载最佳模型进行一次正式测试：

```bash
conda run -n cs9517 python e1_head_only.py \
  --evaluate-checkpoint results/E1_search/1e-3/best_checkpoint \
  --test --output-dir results/E1_selected_test
```

上面的 checkpoint 路径应替换为验证集 F1 最高的那一组。测试结果不参与配置选择。

本次 E1（seed=42）的完整选型结果和 notebook 写入材料见 [experiments/E1/README.md](experiments/E1/README.md)。
