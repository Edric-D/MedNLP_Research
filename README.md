# COMP9444 医学文本疾病命名实体识别项目方案

## 1 项目结论

本项目使用 NCBI Disease Corpus 训练疾病命名实体识别模型。目标是让模型从未见过的 PubMed 摘要中，识别疾病名称的完整文本范围。

团队以 BioBERT 和 PubMedBERT 为两条主线：先在统一条件下完成基础微调，再对两种模型进行公平的超参数和正则化比较，最后分别探索一个独立拓展方向。

四人分工总体合理：

- A 负责数据处理、统一评估和调整后模型的总结性分析。
- B 负责两种模型的 baseline、baseline 对比，以及全部实验的最终比较。
- C 负责 BioBERT 的调参、正则化和一个独立拓展。
- D 负责 PubMedBERT 的调参、正则化和一个独立拓展。

这个分工成立的前提是：所有人必须使用相同的数据版本、标签转换、评估函数、随机种子规范、搜索预算和结果格式。

---

## 2 项目到底要做什么

项目不是搜索所有“医学单词”，而是识别文本中属于疾病的连续词组，并确定准确的开始和结束位置。

例如输入：

```text
The patient was diagnosed with hereditary hemochromatosis and developed liver cirrhosis.
```

期望模型识别：

```text
hereditary hemochromatosis  ->  DISEASE
liver cirrhosis             ->  DISEASE
```

训练时通常将实体转换为 BIO 标签：

```text
hereditary       B-DISEASE
hemochromatosis  I-DISEASE
and              O
developed        O
liver            B-DISEASE
cirrhosis        I-DISEASE
```

模型需要解决的不只是关键词匹配，还包括：

- 识别多词疾病名称；
- 判断疾病名称的准确边界；
- 识别缩写和复合表达；
- 区分疾病、基因、药物和普通医学词汇；
- 泛化到训练集中没有出现过的疾病表达。

建议将总研究问题写成：

> 在统一的数据、训练预算和评估协议下，BioBERT 与 PubMedBERT 在 NCBI Disease Corpus 上的疾病实体识别性能有何差异？超参数调整、正则化和独立拓展方法能否改善模型的实体级泛化能力？

---

## 3 项目基础流程

完整流程如下：

```text
读取 NCBI Disease Corpus
        ↓
解析文本和实体字符区间
        ↓
转换为 BIO 标签
        ↓
与 tokenizer 的 subword 对齐
        ↓
训练 Vanilla BioBERT 和 Vanilla PubMedBERT
        ↓
在统一协议下调参和正则化
        ↓
分别研究独立拓展方法
        ↓
使用统一的实体级指标评估
        ↓
误差分析和最终比较
```

当前数据集包含 793 篇 PubMed 摘要，使用现有官方划分：

| Split | 文档数量 |
|---|---:|
| Train | 593 |
| Validation | 100 |
| Test | 100 |

主要指标是严格实体级：

- Precision
- Recall
- F1-score

Token accuracy 只能作为辅助指标。模型只预测出疾病名称的一部分时，即使多数 token 正确，严格实体级评估仍应判定为错误。

---

## 4 总体实验设计

实验分为三个阶段。

### 4.1 原始状态和 baseline

B 使用固定、合理且不进行复杂搜索的配置，分别正常微调 BioBERT 和 PubMedBERT。

Baseline 不是完全不训练的预训练模型，而是标准 token classification fine-tuning。

建议初始配置：

| 配置项 | 初始约定 |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epoch | 3 |
| Max length | 256，由 A 的长度和截断分析确认 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Seed | 至少固定并记录 42 |
| Checkpoint selection | 仅依据 validation entity F1 |

需要得到：

```text
B1  Vanilla BioBERT
B2  Vanilla PubMedBERT
```

B 首先比较 B1 和 B2，回答在相同基础条件下哪一种预训练模型更适合当前任务。

### 4.2 公平的调参和正则化比较

C 和 D 分别优化对应模型，但必须使用相同类型的搜索空间和相近的计算预算。

```text
T1  Tuned BioBERT
T2  Tuned PubMedBERT
```

推荐搜索范围：

| 参数 | 候选范围 |
|---|---|
| Learning rate | 1e-5, 2e-5, 3e-5, 5e-5 |
| Batch size | 8, 16, 32 |
| Epoch | 3, 5, 8 |
| Weight decay | 0, 0.01, 0.1 |
| Dropout | 0.1, 0.2, 0.3 |
| Warmup ratio | 0, 0.1 |
| Max length | 128, 256, 512 |

推荐使用 Random Search 或规模受控的 Grid Search。不能为了追求结果无限增加某一方的试验次数。

适合 Transformer NER 的正则化方法包括：

- Dropout；
- Weight decay；
- Early stopping；
- Gradient clipping；
- 可选的 label smoothing。

不建议优先加入 BatchNorm。BERT 系列内部使用 LayerNorm，而且 NER 的 batch 通常较小，额外 BatchNorm 会改变预训练结构，也不利于公平比较。

A 最后对 T1 和 T2 做总结性比较，包括 Precision、Recall、F1、随机波动、训练成本和错误类型。

### 4.3 独立改进

公平比较完成后，C 和 D 各自选择一个拓展方向：

```text
X1  Tuned BioBERT + extension
X2  Tuned PubMedBERT + extension
```

可选方向包括：

| 方向 | 研究内容 |
|---|---|
| CRF classification head | 建模相邻 BIO 标签约束，减少非法标签序列 |
| Continued pretraining | 继续进行 Masked Language Modelling，再执行 NER fine-tuning |
| Layer-wise learning rate | 底层使用较小学习率，顶部和分类头使用较大学习率 |
| Layer freezing | 比较冻结、逐步解冻和完全微调 |
| Focal loss 或 class weights | 研究大量 O 标签带来的不平衡问题 |
| 安全数据增强 | 替换完整疾病实体或增强非实体上下文，同时保持标签对齐 |

每个拓展都必须有消融对照。例如：

```text
Tuned BioBERT
vs
Tuned BioBERT + CRF
```

不要一次同时加入多个新组件，否则无法判断提升来自哪一项修改。

C 和 D 可以研究不同拓展，但 X1 和 X2 的差异不能直接解释为 BioBERT 与 PubMedBERT 的模型差异，因为拓展方法本身不同。

独立改进开始前，负责人至少阅读：

1. 对应模型的原始论文；
2. 一篇与拓展方法直接相关的论文；
3. 一篇在 NER 或 biomedical NLP 中使用该方法的实验论文。

负责人需要说明：方法假设、实现成本、预期收益、对照实验和失败条件。

---

## 5 四人具体分工

### A 数据处理和总结性分析

A 负责所有模型共同依赖的数据和评估基础。

主要任务：

- 解析 NCBI Disease Corpus；
- 冻结 train、validation、test 划分；
- 检查重复 PMID、字符 offset 和异常标注；
- 将字符级实体转换为 BIO 标签；
- 完成 tokenizer subword 与 BIO 标签对齐；
- 处理特殊 token、padding 和截断实体；
- 统计文本长度、实体数量、实体长度和 mention type；
- 统计被 max length 截断的样本和实体比例；
- 提供团队唯一的实体级评估函数；
- 验证预测 BIO 标签能否正确恢复成字符 span；
- 对 T1 与 T2 进行总结性比较；
- 分析边界错误、漏检、误检、缩写、复合实体和未见实体。

A 的主要交付：

```text
统一数据文件
统一 preprocessing/tokenization 模块
统一 evaluation 模块
数据分析图表
T1 与 T2 的公平比较
错误分析
```

### B Baseline 和最终比较

B 负责两种模型的 vanilla baseline，并在项目末期整合全部结果。

主要任务：

- 使用同一训练脚本分别运行 Vanilla BioBERT 和 Vanilla PubMedBERT；
- 固定 baseline 配置和随机种子；
- 仅使用 validation set 选择最佳 checkpoint；
- 保存 B1、B2 的训练日志、配置和逐样本预测；
- 比较两个 baseline 的 Precision、Recall、F1、时间和成本；
- 最终整合 B1、B2、T1、T2、X1、X2；
- 制作最终总表；
- 给出项目的最终模型选择和研究结论。

B 的最终比较必须区分：

1. 基础模型差异；
2. 标准调参带来的增益；
3. 独立拓展带来的额外增益；
4. 性能提升与额外复杂度是否值得。

### C BioBERT 优化和拓展

主要任务：

- 基于 B1 的共享代码进行 BioBERT 调参；
- 使用团队统一的搜索空间和预算；
- 研究 dropout、weight decay、early stopping 等正则化；
- 仅根据 validation entity F1 选择 T1；
- 选择一个独立拓展方向并完成 X1；
- 提交配置、日志、checkpoint、预测和消融结果。

### D PubMedBERT 优化和拓展

主要任务：

- 基于 B2 的共享代码进行 PubMedBERT 调参；
- 使用团队统一的搜索空间和预算；
- 研究 dropout、weight decay、early stopping 等正则化；
- 仅根据 validation entity F1 选择 T2；
- 选择一个独立拓展方向并完成 X2；
- 提交配置、日志、checkpoint、预测和消融结果。

---

## 6 工作交接顺序

| 里程碑 | 负责人 | 交付内容 | 接收方 |
|---|---|---|---|
| M1 | A | 冻结数据、BIO 标签和评估函数 | B、C、D |
| M2 | B | B1、B2 baseline 和共享训练代码 | A、C、D |
| M3 | C、D | T1、T2 公平调参结果 | A、B |
| M4 | A | T1、T2 总结性比较和误差分析 | 全组 |
| M5 | C、D | X1、X2 独立拓展和消融 | A、B |
| M6 | B | 全部实验的最终比较和主要结论 | 全组 |
| M7 | 全组 | 合并 notebook、报告和演示材料 | 最终提交 |

**课程最终只有一个 notebook**，但是在开发阶段可以使用模块化的多个ipynb文件，提交前再将必要流程整合进主 notebook。

---

## 7 公平比较的硬性规则

### 7.1 Test set 不能用于调参

正确流程：

```text
Train       -> 训练模型
Validation  -> 选择超参数和 checkpoint
Test        -> 配置冻结后进行最终评估
```

不能根据 test F1 改学习率、epoch、max length 或拓展方法。

### 7.2 必须共用评估函数

所有成员必须调用 A 提供的同一套实体级评估代码。不能由每个人自行实现不同版本的 F1。

### 7.3 搜索预算需要接近

C 和 D 应使用相同或相近的：

- 搜索次数；
- 训练 epoch 上限；
- GPU 时间；
- 随机种子数量；
- checkpoint 选择规则。

如果预算不一致，必须在报告中披露。

### 7.4 重要结果需要考虑随机波动

如果资源允许，重要配置建议运行多个 seed，并报告：

```text
F1 mean ± standard deviation
```

如果只能运行一次，需要在 limitations 中说明。

### 7.5 所有方法必须直接对同一测试文件输出预测

所有正式方法，包括 B1、B2、T1、T2、X1 和 X2，都必须读取同一个测试集：

```text
Dataset/ProcessedData/test.jsonl
```

每个方法必须对其中全部 100 条记录逐条预测，并生成独立的 JSONL 文件。禁止覆盖或修改原始 `test.jsonl`。

建议统一命名：

```text
predictions/B1_test_predictions.jsonl
predictions/B2_test_predictions.jsonl
predictions/T1_test_predictions.jsonl
predictions/T2_test_predictions.jsonl
predictions/X1_test_predictions.jsonl
predictions/X2_test_predictions.jsonl
```

预测文件应保持与 `test.jsonl` 相似的逐行 JSON 结构，并至少包含：

```json
{
  "record_id": "test-0000",
  "pmid": "9949209",
  "text": "...",
  "entities": [
    {
      "start": 346,
      "end": 360,
      "text": "Wilson disease",
      "label": "DISEASE",
      "score": 0.987
    }
  ],
  "experiment_id": "B1",
  "model": "BioBERT"
}
```

字段约束：

- `record_id` 必须与 `test.jsonl` 完全一致，并作为 gold 与 prediction 的连接键；
- 每条测试记录必须恰好出现一次，包括预测不到任何实体的记录；
- `start` 和 `end` 必须使用与原数据相同的字符 offset 约定，其中 `end` 为不包含的结束位置；
- 必须满足 `text[start:end] == entity["text"]`；
- `entities` 按 `start`、`end` 排序，避免重复实体；
- `score` 可选，但如果输出就必须明确其含义；
- 当前主任务只预测 `DISEASE`，不要求模型预测 `mention_type` 或 `concept_id`；
- `annotated_text`、`mention_type`、`concept_id` 和 `text_matches_annotation` 属于 gold 标注或分析信息，不能从 gold 文件复制到预测结果中冒充模型输出。

统一评估程序按照 `record_id` 读取两个文件，并严格比较实体字符区间：

```python
gold_spans = {(e["start"], e["end"]) for e in gold_record["entities"]}
pred_spans = {(e["start"], e["end"]) for e in pred_record["entities"]}
```

只有 `start` 和 `end` 都完全相同才算 True Positive。部分重叠或只预测出疾病名称的一部分，都不能算正确实体。

每个预测文件还应配套保存指标文件：

```text
results/B1_test_metrics.json
results/B2_test_metrics.json
results/T1_test_metrics.json
results/T2_test_metrics.json
results/X1_test_metrics.json
results/X2_test_metrics.json
```

指标文件至少包含 `precision`、`recall`、`f1`、`true_positives`、`false_positives`、`false_negatives`、测试记录数、预测实体数、模型配置和随机种子。

---

## 8 需要特别注意的问题

### 数据和标签

- 字符 offset 必须与原文一致；
- subword 对齐时，特殊 token 和 padding 通常使用 `-100`；
- 需要明确后续 subword 是继续使用 I 标签，还是只监督第一个 subword；
- 记录因截断丢失的实体；
- max length 不足时考虑 sliding window；
- 不要改变官方 train、validation、test 划分。

### 模型训练

- 两个 baseline 应共享同一训练代码；
- 记录随机种子、依赖版本、模型 checkpoint 和设备；
- 梯度累积后需要记录等效 batch size；
- validation checkpoint 选择规则必须统一；
- 不要只保存最终指标，还要保存逐样本预测。

### 数据增强

- 任何增强必须保持实体边界和标签对齐；
- 不要随意替换实体内部单词；
- 回译可能改变实体文本和位置，需要重新标注；
- 安全做法是替换完整疾病实体或只修改非实体上下文。

### 结果解释

- T1 与 T2 可以作为公平优化后的模型比较；
- X1 与 X2 如果使用不同拓展方法，不能直接归因于基础模型差异；
- 不能用很高的 token accuracy 掩盖实体边界错误；
- 需要同时讨论 Precision、Recall、F1、训练成本和稳定性。

---

## 9 最终比较框架

| 比较层次 | 实验 | 应回答的问题 |
|---|---|---|
| 基础模型比较 | B1 vs B2 | 相同基础条件下哪个模型更合适 |
| 模型内优化 | B1 vs T1 | BioBERT 的标准调参带来多少提升 |
| 模型内优化 | B2 vs T2 | PubMedBERT 的标准调参带来多少提升 |
| 公平优化比较 | T1 vs T2 | 统一优化协议后哪个模型更强 |
| 拓展消融 | T1 vs X1 | BioBERT 独立拓展是否有效 |
| 拓展消融 | T2 vs X2 | PubMedBERT 独立拓展是否有效 |
| 最终系统比较 | 全部实验 | 性能、稳定性、成本和复杂度的综合结论 |

最终结果表建议包含：

| Experiment | Model | Method | Precision | Recall | F1 | Seed | Time |
|---|---|---|---:|---:|---:|---:|---:|
| B1 | BioBERT | Vanilla |  |  |  |  |  |
| B2 | PubMedBERT | Vanilla |  |  |  |  |  |
| T1 | BioBERT | Tuned |  |  |  |  |  |
| T2 | PubMedBERT | Tuned |  |  |  |  |  |
| X1 | BioBERT | Tuned + extension |  |  |  |  |  |
| X2 | PubMedBERT | Tuned + extension |  |  |  |  |  |

---

## 10 每次实验必须保存的内容

- Experiment ID；
- 模型 checkpoint 名称；
- Git commit 或代码版本；
- 完整超参数；
- 随机种子；
- 运行设备和训练时间；
- Validation Precision、Recall、F1；
- Test Precision、Recall、F1；
- 最佳 checkpoint 的选择依据；
- 逐样本预测文件；
- 与 `Dataset/ProcessedData/test.jsonl` 一一对应的标准测试预测 JSONL；
- 独立的测试指标 JSON 文件；
- 训练日志；
- 失败或异常说明。

逐样本预测至少包含：

```json
{
  "record_id": "test-0001",
  "gold_entities": ["hereditary hemochromatosis"],
  "predicted_entities": ["hemochromatosis"]
}
```

---

## 11 推荐阅读

### 基础模型

- [BioBERT: a pre-trained biomedical language representation model for biomedical text mining](https://academic.oup.com/bioinformatics/article/36/4/1234/5566506)
- [Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing](https://doi.org/10.1145/3458754)

### 领域适配

- [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://aclanthology.org/2020.acl-main.740/)

### CRF

- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://www.cs.columbia.edu/~jebara/6772/papers/crf.pdf)

### 数据集

- [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)

---

## 12 最终完成标准

- [ ] 一个可以按顺序从头运行的主 notebook；
- [ ] 固定的数据版本和官方划分；
- [ ] 正确的 BIO 标签和 subword 对齐；
- [ ] Vanilla BioBERT baseline；
- [ ] Vanilla PubMedBERT baseline；
- [ ] Tuned BioBERT；
- [ ] Tuned PubMedBERT；
- [ ] BioBERT 独立拓展和消融；
- [ ] PubMedBERT 独立拓展和消融；
- [ ] 严格实体级 Precision、Recall、F1；
- [ ] 逐样本预测文件；
- [ ] B1、B2、T1、T2、X1、X2 均输出与 `test.jsonl` 一一对应的标准预测 JSONL；
- [ ] 所有预测都通过 `record_id` 和严格字符 span 与 gold test set 比较；
- [ ] 数据分析和误差分析；
- [ ] 公平比较、最终结论和限制说明；
- [ ] 全部配置、seed 和运行环境可追踪。

