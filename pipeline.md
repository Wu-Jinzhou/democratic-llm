Here’s a minimal full-parameter DPO fine-tuning script for `meta-llama/Meta-Llama-3-8B` using 🤗 Transformers + TRL’s `DPOTrainer`. This assumes you already have:

* Access to the Llama-3 model on Hugging Face (you must accept Meta’s license & AUP). ([Hugging Face][1])
* A preference dataset with **`prompt`, `chosen`, `rejected`** text fields, as expected by TRL. ([Hugging Face][2])

```python
# train_llama3_dpo.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# 1. Model + tokenizer ---------------------------------------------------------
model_name = "meta-llama/Meta-Llama-3-8B"  # or "meta-llama/Meta-Llama-3-8B-Instruct" if you prefer

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Full-parameter models (no LoRA / PEFT here)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # requires accelerate; spreads across GPUs
)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Preference dataset --------------------------------------------------------
# Dataset must have: "prompt", "chosen", "rejected" text columns
# Example: load from hub or local
dataset = load_dataset("your_org/your_dpo_dataset")  # replace with your dataset
train_dataset = dataset["train"]
eval_dataset = dataset.get("validation", None)

# 3. DPO training config (full finetune) --------------------------------------
training_args = DPOConfig(
    output_dir="./llama3-8b-dpo",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    warmup_ratio=0.1,
    bf16=True,                      # use bf16 if your GPUs support it
    gradient_checkpointing=True,
    max_length=2048,                # passed through to trainer
    max_prompt_length=1024,
    push_to_hub=False,
)

# 4. DPO trainer --------------------------------------------------------------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,            # frozen reference model for implicit rewards :contentReference[oaicite:2]{index=2}
    args=training_args,
    beta=0.1,                       # typical DPO beta in ~[0.1, 0.5] :contentReference[oaicite:3]{index=3}
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,            # TRL will handle tokenization internally :contentReference[oaicite:4]{index=4}
)

# 5. Train + save -------------------------------------------------------------
dpo_trainer.train()
dpo_trainer.save_model("./llama3-8b-dpo")
tokenizer.save_pretrained("./llama3-8b-dpo")
```

Run it with `accelerate` (recommended by TRL docs for DPO): ([Hugging Face][3])

```bash
accelerate launch train_llama3_dpo.py
```

### Notes you’ll probably care about

* **“Full fine tune”** here means all model weights are trainable; we’re *not* using LoRA/PEFT.
* You’ll need **serious GPU memory** for full DPO on Llama-3-8B (multi-GPU or high-RAM A100/4090-class).
* Your dataset rows should look like:

  ```python
  {
      "prompt":   "User: ...\nAssistant:",
      "chosen":   "Better response…",
      "rejected": "Worse response…",
  }
  ```

If you tell me what hardware you’re on and how big your dataset is, I can sketch more realistic hyperparams (batch size, grad accumulation, etc.) tuned to that setup.

[1]: https://huggingface.co/meta-llama/Meta-Llama-3-8B "meta-llama/Meta-Llama-3-8B · Hugging Face"
[2]: https://huggingface.co/docs/trl/v0.9.6/en/dpo_trainer "DPO Trainer"
[3]: https://huggingface.co/docs/trl/en/dpo_trainer "DPO Trainer"
