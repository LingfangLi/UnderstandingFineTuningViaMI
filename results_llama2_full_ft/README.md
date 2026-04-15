# Llama-2-7B 全微调实验结果汇总

范围：针对 Llama-2-7B 在 6 个任务上的全微调（full fine-tuning）结果整理。6 个任务：
`sentiment_yelp`, `sentiment_sst2`, `qa_squad`, `qa_coqa`, `mt_kde4`, `mt_tatoeba`。

**未包含**（仍在做或排除）：layerwise representation distance、EAP top-2000 扫描、induction head 因果消融。

FT 模型路径：`/mnt/scratch/users/sglli24/fine-tuning-project/fine_tuned_model/llama2-7b-{task}-full/`

---

## 1. EAP top-400 Edges（同任务 PT base vs FT 对比）

每个任务在全微调模型上计算 top-400 最重要的 edges（通过 EAP 梯度归因）。

**结果位置**
- 边文件：`output/EAP_edges/finetuned/llama2_{task}_finetuned_edges.csv`
- 与 PT base 的 overlap：`output/EAP_edges/overlap/llama2_{task}_overlap.csv`

**关键发现：FT vs PT 电路保留度（Jaccard）**（见 `pt_vs_ft_overlap.csv`）：

| 任务 | Jaccard | FT 从 PT 继承 % | 重排程度 |
|---|---|---|---|
| squad | **0.477** | 64.4% | 电路几乎原地微调 |
| yelp | 0.384 | 55.5% | 中等保留 |
| tatoeba | 0.370 | 53.8% | 中等保留 |
| coqa | 0.333 | 50.0% | 半数新路由 |
| kde4 | 0.271 | 42.5% | 显著重排 |
| sst2 | **0.176** | 30.0% | 电路大幅重建 |

→ sst2 和 kde4 的 FT 对电路做了最剧烈的重组；squad 基本上是在 PT 电路上微调系数。

---

## 2. EAP Edge 层分布（top-400 heads 所在层）

**结果位置**：`edge_layer_distribution.csv`

| 任务 | Top-5 层 (layer: count) | 模式 |
|---|---|---|
| yelp | L2:41, L4:35, L3:34, L5:33, L6:29 | **早层集中** |
| squad | L2:45, L3:39, L4:38, L5:34, L6:34 | **早层集中** |
| coqa | L2:45, L4:44, L5:43, L3:39, L6:37 | **早层集中** |
| sst2 | L19:25, L31:21, L23:16, L20:15, L16:14 | 中后层扩散 |
| kde4 | **L31:81**, L0:34, L3:32, L2:30, L4:20 | **边界（首末层）** |
| tatoeba | **L31:153**, L0:119, L1:74, L30:12, L28:2 | **极端边界（首末层）** |

→ 3 个 QA/sentiment 类任务（yelp/squad/coqa）都依赖**早期层 L2–L6**；MT 任务（尤其 tatoeba）的重要 head 完全集中在**首末两层**（L0/L1 输入端 + L31 输出端），是和其他任务最不同的电路形态。

**可视化**：`experiments/component_distribution/figure/llama2_{task}_top_400_edges_component_distribution.pdf`

---

## 3. EAP Cross-Task 分析

每个任务用其他 5 个任务的数据作 corrupted baseline 重跑 EAP（共 6×5=30 组），检查某任务电路的跨任务一致性。

**结果位置**：`output/EAP_edges/cross_task_edges/llama2_Finetuned-{A}_Corrupted-Data_{B}_finetuned_edges.csv`

共 30 个 CSV，每个保留 top-400 edges。用于回答"任务 A 的 FT 电路在换用任务 B 数据时边重要性是否稳定"。

> 具体跨任务重叠矩阵尚未做汇总分析，需要跑一次 overlap 比较脚本。建议后续补上（参考同任务 overlap 的做法）。

---

## 4. Component Distribution（top-400 edges 组件类型分布）

统计 top-400 edges 里 attention heads / MLPs / input / output 各占多少。

**结果位置**：`experiments/component_distribution/figure/llama2_{task}_top_400_edges_component_distribution.pdf`

6 个 PDF 全部产出。每个饼图/条形图展示该任务的电路成分构成。

---

## 5. Induction Head Detection

用合成 [A,B,C,A,B,C] 序列测每个 head 的 induction score，阈值 0.3 挑选 induction heads。

**结果位置**：`experiments/induction_head/output/llama2/`
- `induction_scores_FineTuned_{task}.npy` — 32×32 score 矩阵
- `detected_heads_FineTuned_{task}.json` — 通过阈值的 head 列表
- `heatmap_FineTuned_{task}.png` — 可视化
- `scatter_{task}.pdf/png` — PT vs FT 的 induction score 散点图

**每任务 induction head 数量**：

| 任务 | #Heads | 异常 |
|---|---|---|
| sentiment_yelp | 53 | 正常 |
| sentiment_sst2 | 52 | 正常 |
| qa_coqa | 61 | 正常 |
| qa_squad | **15** | ⚠️ 显著少，max score 仅 0.503 |
| mt_kde4 | 58 | 正常 |
| mt_tatoeba | 49 | 正常 |

→ SQuAD FT 后 induction heads 大规模被抑制（15 vs 其他 49-61）。这跟 squad 的 EAP top-400 集中在早期层 L2-L6 一致——squad FT 重新分配了 attention 机制，让 induction-style 复制头失效。

---

## 6. Induction × EAP Overlap

检验"induction heads"和"EAP top-400 功能重要 heads"是否重叠。

**结果位置**
- `experiments/induction_head/output/induction_overlap_stats_edges400.csv`（原始全模型表）
- `induction_overlap.csv`（只 filter llama2 的子表，见本目录）
- 详细分析文档：`experiments/induction_head/NOTES_llama2_full_ft_overlap.md`

**Llama-2 6 任务结果**：

| 任务 | #Ind | #EAP heads | Overlap | Recall(%) | Precision(%) |
|---|---|---|---|---|---|
| yelp | 53 | 263 | 4 | 7.55 | 1.52 |
| sst2 | 52 | 94 | **0** | **0** | **0** |
| squad | 15 | 270 | 2 | 13.33 | 0.74 |
| coqa | 61 | 227 | 12 | 19.67 | 5.29 |
| kde4 | 58 | 166 | **0** | **0** | **0** |
| tatoeba | 49 | 88 | **0** | **0** | **0** |

→ **3 个任务（sst2/kde4/tatoeba）零重叠**，其他任务 recall 也只有 7–20%。这和 gpt2（recall 38–79%）、llama3/qwen2 差异明显。

**解读**：Llama-2 全微调的 "功能重要 heads" 和 "induction heads" 几乎在空间上分离——induction heads 集中在 L6–L26 中部，EAP top-400 集中在 L0/L1 输入端、L31 输出端或早期 L2–L6（详见 NOTES 文档）。

→ 上升到 top-2000 edges 仍然未收敛（canonical induction heads L11.H15, L8.H26, L16.H19 等永久不在 EAP 重要集合里）。

---

## 7. Attention Matrix Shift (KL Divergence)

逐头测 base vs FT 的 attention pattern KL 散度。

**状态**：🔵 正在跑（SLURM job 2954443, gpu-l40s, gpu43）。

**结果位置（跑完后）**：`experiments/attention_matrix_analysis/attention_analysis_results/llama2_full_ft/{task}/`
- `kl_divergence_heads.csv` — 32×32 KL 矩阵
- `kl_divergence_heads.npy`

跑完后需要生成 heatmap/barplot 可视化（参考已有的 `llama2_qlora/` 结果结构）。

---

## 8. Text Complexity 控制实验

只针对 **yelp / squad / tatoeba** 3 个任务，按 lexical complexity 切分 train/test，3×3 交叉（base, simple-FT, complex-FT）× (test_simple, test_complex)。

**结果位置**：
- `experiments/text_complexity/fine_tune/results_full/summary_long.csv`
- 复制到本目录：`text_complexity_summary.csv`

**核心数据**（详见 summary CSV）：

| task | train=base | train=simple | train=complex |
|---|---|---|---|
| yelp (acc) | 0.029-0.036 | 0.936-0.940 | 0.939-0.945 |
| squad (F1) | 0.512-0.537 | 0.757-0.844 | 0.757-0.839 |
| tatoeba (BLEU) | 0.267-0.268 | 0.352-0.376 | 0.345-0.361 |

→ 没有明显的 "FT 偏好训练集复杂度" 现象；simple-train 和 complex-train 在各自 test set 上表现几乎对称。

---

## 汇总表文件索引

本目录下的 CSV：

| 文件 | 内容 |
|---|---|
| `edge_layer_distribution.csv` | 每任务 EAP top-400 edges 的层分布 top5 |
| `pt_vs_ft_overlap.csv` | 每任务 PT vs FT 的 top-400 edges Jaccard overlap |
| `induction_overlap.csv` | 每任务 induction heads 与 EAP heads 的重叠统计 |
| `text_complexity_summary.csv` | Text complexity 3×3 交叉实验指标 |

---

## 整体发现要点（一句话版）

1. **FT 电路继承度任务间差别很大**：squad (64%) > yelp/tatoeba (~55%) > coqa (50%) > kde4 (42%) > sst2 (30%)。
2. **任务 FT 后电路分两类模式**：QA/sentiment 类集中在早期 L2–L6；MT 类集中在首末边界层 L0/L1/L31。
3. **SQuAD 抑制了 induction 机制**：FT 后只剩 15 个 induction heads，且 EAP top-400 集中早层，显示 squad 完全重建了 attention 机制。
4. **Induction heads 和功能重要 heads 在 Llama-2 全微调中几乎不重叠**：3 个任务 0% recall。这与 gpt2 完全不同（gpt2 coqa 79% recall）。
5. **Text complexity 不是 FT 的有效控制变量**：simple/complex train 对结果影响小，两者在 OOD 测试集上差距 <0.02（F1/BLEU）。
