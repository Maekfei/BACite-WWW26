# BA-Cite: Bias-Aware Citation Prediction

Official code for the paper **"From Newborn to Impact: Bias-Aware Citation Prediction"**,
accepted at **WWW'26** (The Web Conference 2026), Long Paper track.

📄 Paper: https://arxiv.org/abs/2510.19246

BA-Cite predicts the future citation impact of **newly published ("newborn") papers** —
a cold-start setting where a paper has little or no citation history — while explicitly
**correcting for exposure bias**, in particular the tendency of high-prestige venues to
attract citations regardless of a paper's intrinsic merit. On the Aminer and OpenAlex
datasets it reduces point error (MALE / RMSLE) by ~13% and improves ranking quality
(NDCG) by ~5.5% over strong baselines.

---

## Table of contents
- [Motivation](#motivation)
- [Method](#method)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Pretrained model & inference](#pretrained-model--inference)
- [Main results](#main-results)
- [Reproducibility notes](#reproducibility-notes)
- [Citation](#citation)

---

## Motivation

Citation data are highly biased. The paper identifies two gaps:

1. **Insufficient modeling of implicit factors.** Prior models lean on a few strongly
   correlated *explicit* proxies (e.g. venue prestige, author reputation), which act as
   shortcuts and generalize poorly under distribution shift.
2. **No bias-aware learning for lowly cited papers.** Citation counts are heavily
   long-tailed; minimizing average error biases models toward highly cited papers and
   systematically under-predicts newborn / long-tail work.

BA-Cite addresses both with (i) a **multi-agent × graph co-learning** module that mines
fine-grained, interpretable signals, and (ii) a set of **bias-aware training mechanisms**.

---

## Method

### 1. Fine-grained feature extraction (six agents)

Six domain-specific agents derive implicit signals from metadata + external resources and
concatenate them into a per-paper feature vector
**`f_p = [R, Q, C, H, Y, A1, A2, A3]`** (8 dims):

| Symbol | Agent | What it captures | Source / scale |
|---|---|---|---|
| `A1,A2,A3` | Role-aware Author Reputation | first / last / other-author reputation | metadata + Wikipedia/Scholar, 1–5 |
| `V`* | Venue Prestige | venue credibility | CCF / CORE rankings, 1–5 |
| `R` | Reproducibility | code/data availability | GitHub/GitLab link check, 0/1 |
| `C` | Collaboration Network | team size, cross-country breadth | metadata + external, 1–5 |
| `H` | Topic Hotness | trendiness of the paper's keywords | prior-year keyword paper counts |
| `Q` | Text Quality | clarity of title/abstract | LLM-scored, 1–5 |
| `Y` | — | publication year | metadata |

\* `V` (venue) is used only in **Stage A** (see below); it is deliberately withheld from
the final prediction to avoid venue shortcut bias.

These features are injected as paper-node attributes in a heterogeneous academic graph
(`paper / author / venue / topic`; edges `written_by`, `published_in`, `has_topic`,
`cites`, …) and propagated by an HGT encoder to produce paper embeddings `z_p`.

### 2. Two-stage forward process (the de-biasing core)

```
Stage A  — Exposure estimation  (V → E)
   full graph INCLUDING venue
   Ê = g_φ([z_p ; V, R, C, H, Q, Y, A1, A2, A3])     # early-exposure latent
   + auxiliary head predicting the citation quartile

Stage B  — Venue-excluded prediction  (E → Y)
   reduced graph with venue nodes/edges REMOVED
   Ŷ = f_θ([z_p ; f_p^(−V) ; Ê])                      # never sees venue directly
```

Venue influences the outcome **only indirectly** through the exposure latent `Ê`,
enforcing the pathway `V → E → Y`. This separates *intrinsic impact* from *exposure bias*.

### 3. Bias-aware training mechanisms

- **GroupDRO** — partitions training papers into venue-tier environments (low / high,
  threshold τ=0.8) and minimizes the **worst-group** risk via adversarial reweighting,
  countering head-dominated bias.
- **Counterfactual regularization** — what-if interventions on *controllable* factors
  (reproducibility `R`, text quality `Q`): raising a good factor should not decrease the
  prediction (**monotonicity**) and should change it smoothly (**smoothness**).
- **Auxiliary Stage-B losses** — an exposure-calibration loss and an adversarial
  venue-invariance loss (small weights), to scrub residual venue information.
- **Prediction loss** — Log-MSE on `log(1+y)` to tame the heavy tail. (The base model
  `two_stage_hgt.py` also supports a negative-binomial head via `use_negative_binomial`.)

Total objective: `L_total = λ_main·L_pred + λ_groupdro·L_groupdro + λ_reg·L_reg (+ aux)`.

---

## Repository structure

```
feature_extraction/                 # agent-based fine-grained feature extraction + data acquisition
  enhanced_author_agent.py          # role-aware author-reputation agent (CAMEL + Google Scholar)
  download_ai_papers_complete.py    # pull CS/AI papers from the OpenAlex API (no key needed)
  convert_format.py                 # raw -> intermediate format conversion
  ai_concept_ids.json               # OpenAlex AI concept ids used to filter the domain
  Only_AI.ipynb                     # filter to AI-domain papers
  mergecitation.ipynb               # merge citation records

gnn/                                # heterogeneous-graph construction + GNN training + eval
  two_stage_hgt_counterfactual.py   # *** BA-Cite *** : two-stage HGT + GroupDRO + counterfactual (Aminer)
  two_stage_hgt_counterfactual_oa.py# BA-Cite, OpenAlex variant
  two_stage_hgt.py                  # base two-stage HGT (no GroupDRO/counterfactual); also defines
                                    #   ModifiedHeterographDataBuilder (builds the PyG HeteroData)
  two_stage_hgt_oa.py               # base two-stage HGT, OpenAlex variant
  counterfactual_config.py          # BA-Cite hyper-params + controllable variables (Aminer)
  counterfactual_config_oa.py       # BA-Cite config, OpenAlex
  two_stage_config.py               # base-model config (DATA_DIR, architecture, training)
  distribution_aware_sampler.py     # distribution-aware sampling for the long-tailed splits
  evaluation_metrics.py             # metric implementations: RMSE/RMSLE/MALE/MAPE/NDCG@k/Hit@k/...
  calculate_metrics.py              # aggregate metrics across runs
  create_overall_data.py            # build the 5-year "overall" aggregation
  data_processing.ipynb             # build the heterogeneous-graph dataset (features, venue tiers)
  data_sampling.ipynb               # train/eval/test split sampling
  process_citations.py              # citation-relation processing (source of `cites` edges)
  simple_merge.py                   # small data-merge utility
```

End-to-end pipeline:

```
download_ai_papers_complete.py / Only_AI / mergecitation   (collect + filter papers)
        │
        ▼
data_processing.ipynb  ──►  data_sampling.ipynb            (records + 8-d features → splits)
        │
        ▼
ModifiedHeterographDataBuilder (in two_stage_hgt*.py)      (records → PyG HeteroData)
        │
        ▼
two_stage_hgt_counterfactual.py                            (two-stage + bias-aware training)
        │
        ▼
evaluation_metrics.py / calculate_metrics.py              (MALE / RMSLE / NDCG / Hit)
```

---

## Installation

```bash
# 1. Install PyTorch matching your CUDA build first:   https://pytorch.org
# 2. Install PyTorch Geometric:                        https://pytorch-geometric.readthedocs.io
# 3. Then the rest:
pip install -r requirements.txt
```

Main dependencies: `torch`, `torch-geometric`, `numpy`, `scipy`, `scikit-learn`,
`pandas`, `networkx`, `tqdm`, `requests`, `matplotlib`, `seaborn`.
The author-reputation agent additionally uses the `camel-ai` toolkit (optional; only
needed if you re-run feature extraction).

---

## Data preparation

The processed datasets are **not** committed (they are large; `.gitignore` excludes
`*.jsonl`, `*.csv`, model weights, etc.). Two public CS-domain datasets are used —
**Aminer** and **OpenAlex** — both heterogeneous (paper/author/venue/topic), covering
2010–2025.

To build them:

1. **Collect papers** — `feature_extraction/download_ai_papers_complete.py` pulls CS/AI
   papers from the OpenAlex API (no API key required), filtered by
   `feature_extraction/ai_concept_ids.json`. Use `Only_AI.ipynb` / `mergecitation.ipynb`
   to filter and merge citation records.
2. **Build the graph dataset** — run `gnn/data_processing.ipynb` (computes the 8-dim
   features and CCF/CORE venue tiers), then `gnn/data_sampling.ipynb` to create splits.
3. **Place the splits** — write `train_processsed.jsonl`, `eval_processsed.jsonl`,
   `test_processsed.jsonl` under `maindata/test1/` (or point `DATA_DIR` at them).

Data paths resolve relative to the **`DATA_DIR`** environment variable (default `.`):

```bash
export DATA_DIR=/path/to/your/data    # code reads $DATA_DIR/maindata/test1/*.jsonl
```

**Split protocol** (as in the paper): train on papers from **2010–2018**, validate on
**2019**, test on **2020**; the target is the citation count over the **next 5 years**
(years 1–5 from the second year after publication). Each split samples 10,000 train /
1,000 val / 1,000 test papers; sampling is repeated 3 times with 5 seeds each, and
results are averaged.

---

## Training

```bash
# *** BA-Cite *** — full bias-aware model (two-stage HGT + GroupDRO + counterfactual)
python gnn/two_stage_hgt_counterfactual.py            # Aminer
python gnn/two_stage_hgt_counterfactual_oa.py         # OpenAlex

# Base two-stage HGT (no GroupDRO / counterfactual)
python gnn/two_stage_hgt.py                           # Aminer
python gnn/two_stage_hgt_oa.py                        # OpenAlex
```

Model / training defaults live in the `gnn/*_config.py` files (architecture, GroupDRO,
counterfactual variables, optimizer, schedule).

---

## Pretrained model & inference

A trained checkpoint is provided so you can run BA-Cite **without retraining**.

**Download** (from the [v1.0 release](https://github.com/Maekfei/BACite-WWW26/releases/tag/v1.0)):

```bash
wget https://github.com/Maekfei/BACite-WWW26/releases/download/v1.0/groupdro_counterfactual_two_stage_model.pth
```

`gnn/predict.py` loads it regardless of where you put it (pass `--checkpoint`); the script
accepts either a raw `state_dict` or a `{'model_state_dict': ...}` checkpoint.

### Input format
BA-Cite is a graph model: it consumes the **whole heterogeneous graph** and predicts for
the papers in it.

- A PyG `HeteroData` built by `ModifiedHeterographDataBuilder` (in `gnn/two_stage_hgt.py`)
  from the data splits — node types `paper / author / venue / topic`; edges `written_by`,
  `published_in`, `has_topic`, `cites`.
- Each **paper** node carries an 8-dim feature vector `[R, Q, C, H, Y, A1, A2, A3]`:

  | dim | name | meaning | range |
  |---|---|---|---|
  | `R` | reproducibility | code/data available | 0 / 1 |
  | `Q` | text quality | title/abstract clarity | 1–5 |
  | `C` | collaboration | team breadth | 1–5 |
  | `H` | topic hotness | keyword trendiness | continuous |
  | `Y` | year | publication year | e.g. 2020 |
  | `A1`/`A2`/`A3` | author reputation | first / last / other | 1–5 |

  The graph is built from `$DATA_DIR/maindata/test1/{train,eval,test}_processsed.jsonl`
  (train+eval are needed to form the graph; predictions are produced for the `test` papers).

### Output format
`model(x_dict, edge_index_dict, paper_features)` returns a dict; the prediction is in
`output['citation']`:

- **Log-MSE head (BA-Cite default):** `output['citation']['log_citation']` = predicted
  `log(1 + y)`. The predicted **5-year citation count** is `expm1(log_citation)`.
- **Negative-binomial head (if enabled):** `output['citation']['mu']` = predicted mean count.

The dict also returns `early_exposure` (the latent Ê) and node `embeddings`.

### Run inference
```bash
python gnn/predict.py --checkpoint path/to/bacite_aminer.pth --out predictions.json
```
This loads the checkpoint, rebuilds the graph from your `DATA_DIR`, runs the model, and
writes per-test-paper predicted 5-year citation counts to `predictions.json`.

### Evaluation metrics
`gnn/evaluation_metrics.py` implements **MALE** / **RMSLE** (point accuracy, on `log(1+y)`,
lower is better) and **NDCG@10 / NDCG@20 / Hit@k** (ranking, higher is better); aggregate
across runs with `gnn/calculate_metrics.py`.

---

## Main results

Overall results (mean ± std), reproduced from the paper (Table 1). Lower is better for
MALE / RMSLE; higher is better for NDCG.

| Dataset | MALE ↓ | RMSLE ↓ | NDCG@10 ↑ | NDCG@20 ↑ |
|---|---|---|---|---|
| **Aminer**   | 0.71 ± 0.02 | 0.88 ± 0.03 | 0.34 ± 0.12 | 0.37 ± 0.13 |
| **OpenAlex** | 0.67 ± 0.01 | 0.92 ± 0.04 | 0.51 ± 0.04 | 0.47 ± 0.01 |

BA-Cite outperforms GNN, sequence, LLM, and metadata baselines (GAT, HINTS, DyGFormer,
BiLSTM, DeepCas, SI-HDGNN, GPT-4o, Llama-3.1-405B, NAIP, BP-NN) across all metrics; see
the paper for the full comparison and ablations.

---

## Reproducibility notes

- Randomness is seeded (`seed=42` by default), but exact numbers can vary slightly with
  hardware / library versions; the paper reports mean ± std over multiple runs.

---

## Citation

```bibtex
@inproceedings{lu2026newborn,
  title={From newborn to impact: Bias-aware citation prediction},
  author={Lu, Mingfei and Wu, Mengjia and Xu, Jiawei and Li, Weikai and Liu, Feng and Ding, Ying and Sun, Yizhou and Lu, Jie and Zhang, Yi},
  booktitle={Proceedings of the ACM Web Conference 2026},
  pages={7600--7609},
  year={2026}
}
```

## License

<!-- TODO: add a license (e.g. MIT) before making the repository public. -->
