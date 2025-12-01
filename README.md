# democratic-llm

End-to-end pipeline for democratic alignment experiments:
- Generate constitution-based evaluation questions
- Build sortition panels from PRISM demographics
- Prepare preference data for hard/soft panel DPO
- Fine-tune Llama 3.1 8B (base) with DPO
- Evaluate tuned models with constitutional judging

## Dependencies

```bash
pip install -r requirements.txt  # or: pip install pandas pyyaml datasets transformers trl torch openai
```

Auth tokens:
- OpenAI: `export OPENAI_API_KEY=...`
- Hugging Face (Meta license accepted): `export HF_TOKEN=...`

The PRISM dataset should live at `prism-alignment/` (already in `.gitignore`), containing `survey.jsonl`, `utterances.jsonl`, and `conversations.jsonl`.

## Data exploration (PRISM)

Peek at unique values before choosing quotas:

```bash
python scripts/data_explore.py \
  --survey prism-alignment/survey.jsonl \
  --columns study_locale,gender,age,education,ethnicity
```

## Sortition configuration

Quotas + attributes live in `configs/panel_config.yaml`. Defaults enforce US locale and quotas for ethnicity (simplified), gender, age, and education with ±5% tolerance. Edit proportions / attributes there to change sortition and panel size.

## Build preference datasets

Creates JSONL with `prompt`, `chosen`, `rejected`, `user_id`, and optional `weight`.

Hard panel (US locale + quotas; filter raters to one sampled panel):
```bash
python scripts/prepare_data.py \
  --mode hard \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --panel-seed 0 \
  --output artifacts/data/hard_panel.jsonl
```

Soft panel (weights = selection probabilities from Monte Carlo sortition):
```bash
python scripts/prepare_data.py \
  --mode soft \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --panel-config configs/panel_config.yaml \
  --num-panel-samples 2000 \
  --panel-seed 0 \
  --output artifacts/data/soft_panel.jsonl
```

Full US-representative subset (no sortition; `included_in_US_REP=true`):
```bash
python scripts/prepare_data.py \
  --mode us_rep \
  --survey prism-alignment/survey.jsonl \
  --utterances prism-alignment/utterances.jsonl \
  --output artifacts/data/us_rep.jsonl
```

## Train Llama 3.1 8B with DPO

Uses TRL’s `DPOTrainer`; weights are honored when present (soft panel).

```bash
python scripts/train_dpo.py \
  --dataset artifacts/data/hard_panel.jsonl \
  --model-id meta-llama/Llama-3.1-8B \
  --output-dir checkpoints/llama3.1-8b-hard \
  --hf-token $HF_TOKEN
```

Switch `--dataset` to `soft_panel.jsonl` or `us_rep.jsonl` for the other variants. Adjust batch size / grad accumulation as needed for your hardware.

## Constitution question generation

Generates 10 questions per clause (written incrementally per clause).

```bash
python generate_questions.py \
  --constitution-path constitution.txt \
  --output-dir artifacts/questions \
  --question-model gpt-4o-mini
```

## Constitutional evaluation (model vs. model)

Compare two models on generated questions and have a judge pick winners.

```bash
python scripts/evaluate_constitution.py \
  --questions-dir artifacts/questions \
  --model-a checkpoints/llama3.1-8b-hard \
  --model-b checkpoints/llama3.1-8b-soft \
  --judge-model gpt-4o-mini \
  --output artifacts/evaluations/compare.jsonl
```

Use `--use-hf-judge` to judge with a Hugging Face model instead of OpenAI. The judge sees anonymized answers (order is shuffled per question).
