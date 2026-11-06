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

### 7.5 所有方法必须使用统一的模型输入和标准输出

数据与分词负责人 A 已经完成原始语料解析、标题与摘要拼接、BioBERT/PubMedBERT 分词和字符 offset 映射。所有正式方法，包括 B1、B2、T1、T2、X1 和 X2，都必须直接使用 `Dataset/TokenizedData` 中对应模型的数据作为训练、验证和测试输入，不得自行重新分词或生成另一套字符坐标。

```text
Dataset/TokenizedData/
├── BioBERT/
│   ├── NCBI_training.jsonl
│   ├── NCBI_validation.jsonl
│   └── NCBI_testing.jsonl
└── PubMedBERT/
    ├── NCBI_training.jsonl
    ├── NCBI_validation.jsonl
    └── NCBI_testing.jsonl
```

任务与输入目录的对应关系为：

| 实验 | 必须使用的输入目录 |
|---|---|
| B1、T1、X1（BioBERT） | `Dataset/TokenizedData/BioBERT/` |
| B2、T2、X2（PubMedBERT） | `Dataset/TokenizedData/PubMedBERT/` |

Validation 用于选择超参数和 checkpoint；Test 只能在配置冻结后用于最终评估。禁止覆盖或修改 A 提供的 TokenizedData 文件。

#### 7.5.1 统一模型输入

每行 JSON 表示一篇已经由 A 完成分词的文档。BioBERT 和 PubMedBERT 的 token 内容与数量可以不同，但字段结构完全相同：

```json
{
  "record_id": "NCBI_training-0000",
  "pmid": "10192393",
  "model": "dmis-lab/biobert-base-cased-v1.1",
  "tokenizer": "google-bert/bert-base-cased",
  "text": "A common human skin tumour ...",
  "entities": [
    {
      "start": 15,
      "end": 26,
      "type": "SpecificDisease"
    }
  ],
  "tokens": ["A", "common", "human", "skin", "t", "##umour"],
  "input_ids": [138, 1887, 1769, 2241, 189, 27226],
  "attention_mask": [1, 1, 1, 1, 1, 1],
  "offset_mapping": [
    [0, 1],
    [2, 8],
    [9, 14],
    [15, 19],
    [20, 21],
    [21, 26]
  ],
  "bio_tags": [
    "O",
    "O",
    "O",
    "B-SpecificDisease",
    "I-SpecificDisease",
    "I-SpecificDisease"
  ]
}
```

输入约束：

- `text` 已由 A 严格按照 `title + " " + abstract` 构造，不得再次清洗、规范化或修改；
- `record_id` 是连接 tokenized input、gold 和 prediction 的唯一键，不能只使用 PMID；
- `tokens`、`input_ids`、`attention_mask` 和 `offset_mapping` 必须按相同下标一一对应，长度必须相等；
- `offset_mapping` 的每一项是 `[start, end]`，表示相对于完整 `text` 的全局半开字符区间 `[start, end)`；
- Training 的 `entities` 包含 gold `start`、`end` 和 `type`，`bio_tags` 与 WordPiece 一一对应；
- Validation/Test 保留 `entities` 和 `bio_tags` 字段，但内容固定为 `[]`；
- 各实验必须使用对应目录中已经生成的 `input_ids` 和 token offset，不得再次调用 tokenizer 重新分词；
- 模型负责人可以根据训练需要添加特殊 token、padding 或滑动窗口，但必须以 A 提供的 token 序列、typed-BIO 标签和全局 offset 为唯一基础，且不能改变原文坐标；
- `attention_mask` 的 `1` 表示有效 token；后续 padding 位置使用 `0`，不能把 `attention_mask` 改作 BIO 标签或预测结果字段。

#### 7.5.2 模型负责人必须完成的处理

分工边界如下：

```text
A 已完成：
RawData
    -> title + " " + abstract
    -> BioBERT/PubMedBERT tokenizer
    -> typed-BIO 标签与 WordPiece 对齐
    -> tokens、input_ids、attention_mask、offset_mapping、bio_tags
    -> Validation/Test gold 字段遮掩
    -> Dataset/TokenizedData 对应目录

各模型负责人完成：
读取对应 TokenizedData
    -> 将 Training bio_tags 映射为 label ID 并训练模型
    -> 在 Validation/Test 上推理
    -> token/window logits 聚合
    -> typed-BIO 约束解码
    -> 原文字符区间 [start, end) 与实体 type
    -> 去重和排序
    -> 标准预测 list[dict]
```

每个模型负责人必须直接读取对应 TokenizedData 完成训练和测试，并在调用评估函数前完成：

```text
模型 logits
    -> 滑动窗口重叠 logits 聚合
    -> typed-BIO 与 WordPiece 合法性约束
    -> typed-BIO 序列解码
    -> 原文字符区间 [start, end) 与实体 type
    -> 去重和排序
```

统一评估程序不负责重新分词、token 聚合、滑动窗口合并或 BIO 解码，只接收已经恢复到原文全局字符坐标的实体区间。模型负责人交付给 A 的结果必须符合下一节的标准预测输出格式。

##### 7.5.2.1 typed-BIO 训练与推理约束

本项目统一使用 `O` 加四种实体类型各自的 `B-`、`I-` 标签，共 `4 * 2 + 1 = 9` 类：

```text
O
B-SpecificDisease / I-SpecificDisease
B-DiseaseClass / I-DiseaseClass
B-CompositeMention / I-CompositeMention
B-Modifier / I-Modifier
```

训练数据约束：

- 所有模型必须使用相同的 `LABEL2ID` 和 `ID2LABEL`，不得自行调整标签顺序；
- Training 的 `bio_tags` 必须与 `tokens`、`input_ids`、`attention_mask` 和 `offset_mapping` 按下标一一对应；
- `[CLS]`、`[SEP]`、padding 和其他不参与监督的位置统一使用 `-100`，不得当作 `O` 参与 loss；
- Validation/Test 的 `entities=[]`、`bio_tags=[]` 是主动遮掩。不得从 RawData、ProcessedData 或其他 gold 文件恢复标签，也不得将其用于 loss、输入特征、数据增强或阈值拟合；
- baseline 先使用普通 cross-entropy。加权 cross-entropy、focal loss 或重采样只能作为独立实验，且类别权重只能由 Training 统计得到；
- token accuracy 只能作为诊断指标，不能替代 typed entity-level F1，也不能单独作为 checkpoint 选择依据；

滑动窗口约束：

- 窗口内的 `offset_mapping` 必须继续使用全文全局字符坐标，不得转换成窗口局部坐标；
- `[CLS]`、`[SEP]` 和 padding 的标签使用 `-100`；正文 token 保留原始 typed-BIO 标签；
- 重叠窗口必须先按照全局 token 位置聚合 logits，再进行一次全局 typed-BIO 解码；禁止每个窗口独立解码后直接拼接实体；
- 必须记录被窗口边界截断的实体数量；任何方法都不能静默丢弃或缩短这些实体；

typed-BIO 合法转移约束：

- 序列开头或 `O` 后只能出现 `O` 或 `B-X`，不能直接出现 `I-X`；
- `B-X` 或 `I-X` 后若继续使用 `I`，类型必须仍为 `X`；`I-Y`（`X != Y`）属于非法转移；
- 相邻但类型不同的实体必须由新的 `B-Y` 开始；
- 若逐 token argmax 产生非法 `I-X`，统一将其按 `B-X` 处理；正式实验优先使用带合法转移限制的 Viterbi 解码；
- 对以 `##` 开头的 WordPiece，不允许在同一单词内部新建实体或切换类型：前一 WordPiece 为 `O` 时保持 `O`，前一 WordPiece 属于类型 `X` 时只能继续为 `I-X`；
- 不得用多数投票强行合并连续但类型冲突的预测，这会掩盖模型真实的类型错误；

解码与评估约束：

- 每个预测实体必须通过首尾 WordPiece 的全局 offset 恢复为 `start`、`end` 和 `type`；
- 类型切换必须形成新的实体边界，最终预测按 `(start, end, type)` 去重和排序；
- 严格评估只有在 `start`、`end` 和 `type` 三项均正确时才记为 True Positive；边界正确但类型错误时记为一个 False Positive 和一个 False Negative；
- 除 Micro F1 外，还必须报告 Macro F1、每种实体类型的 Precision/Recall/F1、类型 confusion matrix、非法 BIO 转移数和 WordPiece 内部类型冲突数。

#### 7.5.3 标准预测输出

每种方法必须在内存中生成一个 Python `list[dict]`，然后直接传给统一评估函数，不需要先将预测结果保存为 JSONL。列表中的每个字典对应一篇文档，并且只能包含文档标识和模型识别出的实体字符区间：

```python
prediction_records = [
    {
        "record_id": "NCBI_testing-0000",
        "pmid": "9949209",
        "entities": [
            {"start": 346, "end": 360, "type": "SpecificDisease"},
            {"start": 512, "end": 527, "type": "DiseaseClass"},
        ],
    },
    {
        "record_id": "NCBI_testing-0001",
        "pmid": "1234567",
        "entities": [],
    },
]
```

输出约束：

- 每篇 Validation/Test 文档必须恰好出现一次，包括预测不到任何实体的文档；
- `record_id` 和 `pmid` 必须与对应输入记录一致；
- `start` 和 `end` 必须是相对于完整 `text` 的全局字符 offset；
- 字符区间统一使用 `[start, end)`，其中 `start` 包含、`end` 不包含；
- 每个预测必须满足 `0 <= start < end <= len(text)`；
- 每个实体的 `type` 必须属于 `SpecificDisease`、`DiseaseClass`、`CompositeMention` 或 `Modifier`；
- `entities` 必须按 `start`、`end`、`type` 升序排列，并删除重复的 `(start, end, type)`；
- 标准输出实体只包含 `start`、`end` 和 `type`，不包含 token、BIO 标签、实体文本、score、logits、attention mask 或窗口编号；
- 模型配置、checkpoint、随机种子和实验名称保存在独立配置或指标文件中；
- 禁止从 gold 文件复制 `entities` 作为模型预测。

#### 7.5.4 统一实体级评估

统一评估程序按照 `record_id` 连接 gold 与 prediction，并严格比较实体字符区间和实体类型：

```python
gold_entities = {
    (e["start"], e["end"], e["type"])
    for e in gold_record["entities"]
}
pred_entities = {
    (e["start"], e["end"], e["type"])
    for e in pred_record["entities"]
}

true_positives = len(gold_entities & pred_entities)
false_positives = len(pred_entities - gold_entities)
false_negatives = len(gold_entities - pred_entities)
```

只有 `start`、`end` 和 `type` 都完全相同才算 True Positive。边界正确但类型错误时计为一个 False Positive 和一个 False Negative。主指标为严格 typed exact-span Micro Precision、Recall 和 F1，并同时报告 Macro F1 与每种类型的 Precision、Recall 和 F1。

除主指标外，统一评估函数还应报告忽略类型的 Exact-span F1、边界完全正确条件下的 type accuracy、类型 confusion matrix，以及 Start-boundary、End-boundary 和 Character-overlap 指标。这些指标只用于诊断，不能替代 typed exact-span F1。

##### 评估函数入口

评估函数位于主 notebook 的 `4. Standardized Testing` 章节。运行该代码单元后，只使用下面这个统一入口：

```python
metrics = evaluate_model_predictions(prediction_records, split)
```

函数恰好接收两个参数：

| 参数 | 类型 | 说明 |
|---|---|---|
| `prediction_records` | `list[dict]` | 模型在内存中生成的标准预测列表；列表内容必须符合 7.5.3 的格式 |
| `split` | `str` | 只能是 `"validation"` 或 `"test"`；大小写和首尾空格会被标准化 |

`split` 会自动选择 `Dataset/ProcessedData` 中的金标准，不需要也不允许调用者另行传入 gold 路径：

| `split` | 自动使用的 gold 文件 |
|---|---|
| `"validation"` | `Dataset/ProcessedData/NCBI_validation.jsonl` |
| `"test"` | `Dataset/ProcessedData/NCBI_testing.jsonl` |

Validation 调用示例：

```python
metrics = evaluate_model_predictions(
    validation_prediction_records,
    "validation",
)

display(metrics_to_table(metrics).round(4))
```

配置冻结后的 Test 调用示例：

```python
metrics = evaluate_model_predictions(
    test_prediction_records,
    "test",
)

display(metrics_to_table(metrics).round(4))
save_metrics(
    metrics,
    project_dir / "results" / "B1_test_metrics.json",
)
```

函数返回一个 Python 字典，而不是单独一个 F1。输出结构如下：

```python
{
    "averaging": "micro",
    "document_count": ...,
    "gold_entity_count": ...,
    "predicted_entity_count": ...,
    "exact_span": {
        "true_positives": ...,
        "false_positives": ...,
        "false_negatives": ...,
        "precision": ...,
        "recall": ...,
        "f1": ...,
    },
    "start_boundary": { ... },
    "end_boundary": { ... },
    "character_overlap": { ... },
}
```

四个指标项具有相同的内部字段。`exact_span` 是正式主指标；其余三项用于边界和部分匹配分析。`metrics_to_table(metrics)` 可以将四组结果转换成表格，`save_metrics(metrics, output_path)` 可以将完整字典保存为 JSON。

评估前，函数还会检查以下内容：输入是否为字典列表、预测列表是否覆盖所选划分的全部文档且没有额外文档、`record_id` 是否唯一、`pmid` 是否匹配、实体区间是否合法，以及 `entities` 是否已经排序和去重。格式不合格时函数会抛出 `TypeError` 或 `ValueError`，不会静默跳过错误记录。

每次正式测试评估还应保存对应的指标文件：

```text
results/B1_test_metrics.json
results/B2_test_metrics.json
results/T1_test_metrics.json
results/T2_test_metrics.json
results/X1_test_metrics.json
results/X2_test_metrics.json
```

指标文件至少保存评估函数返回的完整字典。模型配置、checkpoint 和随机种子应保存在配套实验配置中，或在保存指标时作为额外元数据一同记录。

---

## 8 需要特别注意的问题

### 数据和标签

- 字符 offset 必须与原文一致；
- typed-BIO 必须使用统一的 9 类标签表，不能退回无类型的 `B-DISEASE/I-DISEASE`；
- 当前 TokenizedData 已为正文 WordPiece 生成 typed-BIO；特殊 token 和 padding 使用 `-100`；
- Validation/Test 中 `entities=[]`、`bio_tags=[]` 是主动遮掩，不是数据缺失；
- 记录因截断丢失的实体；
- max length 不足时考虑 sliding window；
- 不要改变官方 train、validation、test 划分。

### 模型训练

- 两个 baseline 应共享同一训练代码；
- 记录随机种子、依赖版本、模型 checkpoint 和设备；
- 梯度累积后需要记录等效 batch size；
- validation checkpoint 选择规则必须统一；
- 记录非法 BIO 转移数、WordPiece 内部类型冲突数和各类型实体级 F1；
- 类别加权或 focal loss 只能使用 Training 标签统计，并与未加权 baseline 分开报告；
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
- 独立的测试指标 JSON 文件；
- 训练日志；
- 失败或异常说明。

统一评估直接接收内存中的标准预测列表，因此不要求额外保存预测 JSONL。若某项实验需要进行逐样本错误分析，可以自行将内存结果序列化，但序列化格式不属于评估函数接口。

用于错误分析的逐样本记录可以包含：

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
- [ ] B1、B2、T1、T2、X1、X2 均生成与所选划分一一对应的标准内存预测列表；
- [ ] 所有预测都通过 `record_id` 和严格字符 span 与 gold test set 比较；
- [ ] 数据分析和误差分析；
- [ ] 公平比较、最终结论和限制说明；
- [ ] 全部配置、seed 和运行环境可追踪。

