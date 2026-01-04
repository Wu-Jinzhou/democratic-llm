# democratic-llm

Democratic alignment pipeline built on PRISM preference data and Sortition Foundation panel selection. It trains and evaluates models under hard/soft panel objectives and a constitutional evaluation.

## Repository layout

- `configs/panel_config.yaml`: demographic quotas, tolerance, locale filtering, and slack categories.
- `scripts/prepare_data.py`: builds DPO datasets (hard/soft/us_rep/full).
- `scripts/sortition.py`: LEGACY/LEXIMIN panel selection + weight estimation.
- `scripts/train_dpo.py`: DPO fine-tuning for `meta-llama/Llama-3.1-8B`.
- `generate_questions.py`: generate clause questions (OpenAI API).
- `scripts/evaluate_constitution.py`: compare two models + judge.
- `scripts/data_explore.py`: quick data inspection utilities.
- `third_party/`: Sortition Foundation LEXIMIN implementation (GPLv3).

## Setup

```bash
pip install -r requirements.txt
```

Auth tokens:
- OpenAI: `export OPENAI_API_KEY=...`
- Hugging Face (Meta license accepted): `export HF_TOKEN=...`

Weights & Biases (optional):
- `pip install wandb` if not already installed.
- `export WANDB_API_KEY=...`

Optional for LEXIMIN:
- Install `gurobipy` + `python-mip` and a Gurobi license.
- See `third_party/stratification.py` (GPLv3).

## Data

Place the PRISM dataset under `prism-alignment/` (already in `.gitignore`).
You can clone it directly from Hugging Face:
```bash
git clone https://huggingface.co/datasets/HannahRoseKirk/prism-alignment
```

The folder should contain:
- `survey.jsonl` (demographics)
- `utterances.jsonl` (preference comparisons)
- `conversations.jsonl` (multi-turn context; not required for DPO prep)

## Step-by-step guide

### 1) Inspect PRISM categories
```bash
python scripts/data_explore.py \
  --survey prism-alignment/survey.jsonl \
  --columns study_locale,gender,age,education,ethnicity,religion,marital_status,employment_status
```

### 2) Configure quotas (sortition)
Edit `configs/panel_config.yaml`:
- `panel_size`: number of raters in the panel.
- `tolerance`: relative slack around target proportions.
- `locale_filter`: set `"us"` or `null` to include all locales.
- `attributes`: list of demographic attributes with `population_proportions`.
- `slack_categories`: categories that impose no bounds (useful for `Other` / `Prefer not to say`).

Example structure:
```yaml
panel_size: 300
tolerance: 0.05
locale_filter: "us"
attributes:
  - name: gender
    column: gender
    population_proportions:
      Female: 0.51
      Male: 0.48
      "Non-binary / third gender": 0.005
      "Prefer not to say": 0.005
    slack_categories:
      - "Prefer not to say"
```

### 3) Build DPO datasets
All datasets output JSONL with fields `prompt`, `chosen`, `rejected`, `user_id`, `interaction_id`, `weight`.
By default the dataset uses TRL's conversational format where `prompt`, `chosen`, and `rejected` are lists of
`{"role","content"}` messages so the chat template is applied during training. Use `--dataset-format raw` if you
want plain strings instead.

Optional flags for `scripts/prepare_data.py`:
- `--survey` (default: `prism-alignment/survey.jsonl`)
- `--utterances` (default: `prism-alignment/utterances.jsonl`)
- `--panel-config` (default: `configs/panel_config.yaml`)
- `--panel-algorithm` (choices: `legacy`, `leximin`, `random`)
- `--panel-seed` (default: `42`)
- `--num-panel-samples` (default: `2000`)
- `--num-workers` (default: `1`)
- `--dataset-format` (default: `chat`, use `raw` for plain strings)
- `--system-prompt` (optional, adds a system message for chat format)

Hard panel (single LEGACY/LEXIMIN panel):
```bash
python scripts/prepare_data.py \
  --mode hard \
  --panel-algorithm legacy \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --panel-seed 42 \
  --output artifacts/data/hard_panel.jsonl
```

Soft panel (weights = selection probabilities):
```bash
python scripts/prepare_data.py \
  --mode soft \
  --panel-algorithm legacy \
  --num-panel-samples 2000 \
  --num-workers 8 \
  --panel-seed 42 \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --output artifacts/data/soft_panel.jsonl
```

US-representative subset (`included_in_US_REP=true`):
```bash
python scripts/prepare_data.py \
  --mode us_rep \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --output artifacts/data/us_rep.jsonl
```

Full dataset (no filtering):
```bash
python scripts/prepare_data.py \
  --mode full \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --output artifacts/data/full.jsonl
```

Panel algorithms:
- `legacy`: Sortition Foundation LEGACY algorithm (default; randomized).
- `leximin`: Sortition Foundation LEXIMIN algorithm (exact probabilities; requires Gurobi).
- `random`: naive rejection sampling (debug only).

LEXIMIN setup (Gurobi):
```bash
pip install gurobipy mip
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

Use LEXIMIN:
```bash
python scripts/prepare_data.py \
  --mode soft \
  --panel-algorithm leximin \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --output artifacts/data/soft_panel.jsonl
```

### 4) Train with DPO (Llama 3.1 8B)
Optional flags for `scripts/train_dpo.py`:
- `--model-id` (default: `meta-llama/Llama-3.1-8B`)
- `--device-map` (default: `auto`, use `none` for distributed/FSDP)
- `--attn-implementation` (e.g. `flash_attention_2`)
- `--output-dir` (default: `checkpoints/llama3.1-8b-dpo`)
- `--hf-token` (default: `HF_TOKEN`)
- `--per-device-train-batch-size` (default: `1`)
- `--gradient-accumulation-steps` (default: `8`)
- `--learning-rate` (default: `5e-6`)
- `--num-train-epochs` (default: `2`)
- `--beta` (default: `0.1`)
- `--weight-decay` (default: `0.0`)
- `--eval-ratio` (default: `0.02`)
- `--eval-strategy` (default: `steps`, choices: `no`, `steps`, `epoch`)
- `--eval-steps` (default: `100`)
- `--save-strategy` (default: `no`, choices: `no`, `steps`, `epoch`)
- `--save-steps` (default: `500`, only used with `--save-strategy steps`)
- `--save-total-limit` (optional, max checkpoints to keep)
- `--seed` (default: `42`)
- `--report-to` (default: `wandb`, use `none` to disable)
- `--logging-dir` (default: `logs`)
- `--run-name` (optional W&B run name)
- `--wandb-project` (optional)
- `--wandb-entity` (optional)
- `--wandb-group` (optional)
- `--dataloader-num-workers` (default: `0`)
- `--dataloader-prefetch-factor` (optional; set >0 when num-workers > 0)
- `--fsdp` (e.g. `full_shard auto_wrap`, enable FSDP)
- `--fsdp-min-num-params` (optional)
- `--fsdp-transformer-layer-cls-to-wrap` (e.g. `LlamaDecoderLayer`)
- `--fsdp-use-orig-params` (optional, sets `use_orig_params=True`)
- `--fsdp-config` (optional JSON file with extra FSDP settings)

```bash
python scripts/train_dpo.py \
  --dataset artifacts/data/hard_panel.jsonl \
  --model-id meta-llama/Llama-3.1-8B \
  --output-dir checkpoints/llama3.1-8b-hard \
  --hf-token $HF_TOKEN \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

Switch `--dataset` to `soft_panel.jsonl`, `us_rep.jsonl`, or `full.jsonl` as needed.
If the tokenizer does not include a chat template, `scripts/train_dpo.py` falls back to a simple
`User:` / `Assistant:` template so conversational datasets still work.
For multi-GPU runs (accelerate/torchrun), `device_map=auto` is automatically disabled so FSDP/DDP can shard.

Multi-GPU (Accelerate):
```bash
accelerate config
```

Then launch:
```bash
accelerate launch scripts/train_dpo.py \
  --dataset artifacts/data/soft_panel.jsonl \
  --model-id meta-llama/Llama-3.1-8B \
  --output-dir checkpoints/llama3.1-8b-soft \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --fsdp "full_shard auto_wrap" \
  --fsdp-transformer-layer-cls-to-wrap LlamaDecoderLayer
```

### 5) Generate constitution questions
Optional flags for `generate_questions.py`:
- `--output-dir` (default: `artifacts/questions`)
- `--question-model` (default: `gpt-5.2`)
- `--api-key` (default: `OPENAI_API_KEY`)
- `--base-url` (optional OpenAI-compatible endpoint)
- `--n-questions` (default: `40`)
- `--num-workers` (default: `1`)
- `--max-retries` (default: `3`)
- `--retry-backoff` (default: `2.0`)
- `--max-output-tokens` (default: `3000`)
- `--start-clause` / `--end-clause` (optional clause range)
- `--temperature` (default: `0.7`)

```bash
python generate_questions.py \
  --constitution-path constitution.txt \
  --output-dir artifacts/questions \
  --question-model gpt-5.2 \
  --num-workers 8
```

### 6) Evaluate models with a judge
Optional flags for `scripts/evaluate_constitution.py`:
- `--questions-dir` (default: `artifacts/questions`)
- `--mode` (default: `pairwise`, choices: `pairwise`, `listwise`)
- `--models` (candidate models; listwise ranks all models, pairwise compares all pairs)
- `--hf-token` (default: `HF_TOKEN`)
- `--judge-model` (default: `gpt-5.2`)
- `--use-hf-judge` (use HF judge instead of OpenAI)
- `--output` (default: `artifacts/evaluations/pairwise.jsonl` or `listwise.jsonl` based on mode)
- `--preferences-output` (default: `artifacts/evaluations/preferences.jsonl`)
- `--responses-dir` (default: `artifacts/evaluations/responses`)
- `--overwrite-responses` (regenerate cached responses)
- `--max-questions` (optional global cap)
- `--questions-per-clause` (sample N questions per clause)
- `--num-judges` (default: `1`)
- `--max-new-tokens` (default: `256`)
- `--judge-max-output-tokens` (default: `400`)
- `--judge-retries` (default: `3`)
- `--retry-backoff` (default: `2.0`)
- `--system-prompt` (optional)
- `--batch-size` (default: `1`, local model generation batching)
- `--seed` (default: `42`)

```bash
python scripts/evaluate_constitution.py \
  --questions-dir artifacts/questions \
  --mode listwise \
  --models checkpoints/llama3.1-8b-hard checkpoints/llama3.1-8b-soft \
  --questions-per-clause 10 \
  --num-judges 1 \
  --judge-model gpt-5.2 \
  --output artifacts/evaluations/listwise.jsonl \
  --preferences-output artifacts/evaluations/preferences.jsonl
```

Add `--use-hf-judge` to judge with a Hugging Face model instead of OpenAI.

Judgement count formulas:
- Let `C` = number of clauses, `Q` = questions per clause, `J` = judges per question, `M` = number of models.
- Total questions: `C * Q`
- Listwise judgements (one ranking per judge): `C * Q * J`
- Pairwise judgements (one comparison per judge per model pair): `C * Q * J * (M * (M - 1) / 2)`
- Pairwise comparisons implied by listwise rankings: `C * Q * J * (M * (M - 1) / 2)`

Fit Bradley-Terry scores from preferences:
```bash
python scripts/fit_bradley_terry.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --output artifacts/evaluations/bradley_terry_scores.json
```

Bootstrap confidence intervals:
```bash
python scripts/fit_bradley_terry.py \
  --preferences artifacts/evaluations/preferences.jsonl \
  --output artifacts/evaluations/bradley_terry_scores.json \
  --bootstrap-samples 500
```

Score models from listwise rankings (choose one method at a time):
```bash
python scripts/score_rankings.py \
  --listwise artifacts/evaluations/listwise.jsonl \
  --output artifacts/evaluations/ranking_scores.json \
  --method plackett-luce \
  --bootstrap-samples 500
```

Methods: `plackett-luce`, `borda`, `copeland`, `kemeny`.

For Kemeny ILP, install `mip` (optional; otherwise uses a heuristic for larger model counts):
```bash
pip install mip
```

### Notebook: compare checkpoint responses
Use `notebooks/compare_checkpoints.ipynb` to compare multiple models side by side on a single prompt
with chat-style formatting.

## Outputs

- DPO dataset JSONL: `artifacts/data/*.jsonl`
  - fields: `prompt`, `chosen`, `rejected`, `user_id`, `interaction_id`, `weight`
  - `prompt`/`chosen`/`rejected` are chat messages when `--dataset-format chat`
- Question files: `artifacts/questions/clause_XX.json`
  - fields: `clause_id`, `clause`, `questions`, `question_model`
- Evaluation JSONL (pairwise): `artifacts/evaluations/pairwise.jsonl`
  - fields include `question`, `responses`, `wins_i`, `wins_j`, `majority_winner`, `judge_raw`
- Evaluation JSONL (listwise): `artifacts/evaluations/listwise.jsonl`
  - fields include `question`, `responses`, `rankings`, `judge_raw`
- Preferences JSONL: `artifacts/evaluations/preferences.jsonl`
  - fields include `model_i`, `model_j`, `wins_i`, `wins_j`, `majority_winner`
- Bradley-Terry scores: `artifacts/evaluations/bradley_terry_scores.json`
- Ranking scores: `artifacts/evaluations/ranking_scores.json`

## Troubleshooting

- **No feasible panels**: loosen `tolerance`, reduce `panel_size`, or mark `Other`/`Prefer not to say` as slack. Verify category labels with `scripts/data_explore.py`.
- **LEXIMIN errors**: install `gurobipy` + `python-mip` and a valid Gurobi license.
- **Token errors**: set `OPENAI_API_KEY` for OpenAI calls, `HF_TOKEN` for Llama access.

## Notes

- The PRISM data are conversational, but DPO prep uses per-interaction `user_prompt` with preferred vs. rejected responses.
- Soft panel weights are exact under `leximin`, approximate under `legacy` (Monte Carlo).
