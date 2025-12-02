import os
import math
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from bert_score import score as bertscore_score
import torch.nn.functional as F

SFT_MODEL_PATH = "./sft_model"

CSV_PATH = "qqp_subset_1000.csv"

# Output directory for this specific experiment
OUTPUT_DIR = "./output_rl_bertscore/final"

BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
LEARNING_RATE = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
hf_logging.set_verbosity_error()

# ============================================================
# 1. Data Loading
# ============================================================
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)

EXAMPLES = [
    {"instruction": row["instruction"], "reference": row["reference"]}
    for _, row in df.iterrows()
]

class InstructionDataset(Dataset):
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex["instruction"], ex["reference"]


dataset = InstructionDataset(EXAMPLES)

# ============================================================
# 2. Model Loading (From SFT Base)
# ============================================================
print(f"Loading SFT model from: {SFT_MODEL_PATH} ...")

tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Policy Model (The one we train)
model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(device)
model.resize_token_embeddings(len(tokenizer))

# Reference Model (Frozen, for KL penalty)
ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Scaler


# ============================================================
# 3. Collate Function
# ============================================================
def collate_batch(batch, max_prompt_len: int = 128):
    instructions, references = zip(*batch)
    enc = tokenizer(
        list(instructions),
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "instructions": list(instructions),
        "references": list(references),
    }


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# 4. Penalty & Diversity Functions
# ============================================================
def repetition_penalty_score(text: str) -> float:
    toks = text.strip().split()
    if len(toks) <= 1: return 0.0
    return len(set(toks)) / len(toks)


def apply_length_repetition_penalty(texts, base_rewards, beta_len=0.2, beta_rep=0.2):
    penalties = []
    for t in texts:
        toks = t.strip().split()
        length = len(toks)
        # Penalty for extremely short sentences (< 5 words)
        len_pen = -beta_len if length < 5 else 0.0
        # Penalty for low diversity (repetition)
        div_score = repetition_penalty_score(t)
        rep_pen = -beta_rep * (1.0 - div_score)
        penalties.append(len_pen + rep_pen)

    penalties = torch.tensor(penalties, dtype=torch.float32)
    return torch.clamp(base_rewards + penalties, 0.0, 1.0)


def repetition_diversity_scores(texts):
    return torch.tensor(
        [repetition_penalty_score(t) for t in texts],
        dtype=torch.float32
    )


# ============================================================
# 5. BERTScore Reward Calculation
# ============================================================
def compute_bertscore_rewards(hyps, refs):
    # Calculate BERTScore (semantic similarity)
    P, R, F1 = bertscore_score(hyps, refs, lang="en", verbose=False, device=device)

    # Detach to ensure no gradient flow back into BERTScore model
    base = F1.detach().float().cpu()  # Move to CPU for arithmetic with penalties

    # Apply penalties (Length & Repetition)
    shaped = apply_length_repetition_penalty(hyps, base)

    return shaped, base


# ============================================================
# 6. Generation (Sampling)
# ============================================================
def generate_responses(input_ids, attention_mask, max_new_tokens=64):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    model.eval()
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=5,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    outputs = []
    for i in range(gen_ids.size(0)):
        prompt_len = input_ids.size(1)
        new_tokens = gen_ids[i, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if "Response:" in text:
            text = text.split("Response:")[-1].strip()
        outputs.append(text)

    return outputs, gen_ids


# ============================================================
# 7. Log Probabilities & KL
# ============================================================
def sequence_log_probs(gen_ids):
    input_ids = gen_ids[:, :-1].to(device)
    target_ids = gen_ids[:, 1:].to(device)
    attention_mask = (target_ids != tokenizer.pad_token_id).long()

    model.train()
    logits = model(input_ids=input_ids).logits

    log_probs = torch.log_softmax(logits, dim=-1)
    gather = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    gather = gather * attention_mask
    return gather.sum(dim=1)


def kl_penalty(gen_ids, coef=0.05):
    with torch.no_grad():
        ref_logits = ref_model(gen_ids[:, :-1]).logits
    cur_logits = model(gen_ids[:, :-1]).logits

    kl = F.kl_div(
        F.log_softmax(cur_logits, dim=-1),
        F.softmax(ref_logits, dim=-1),
        reduction="batchmean",
    )
    return coef * kl


# ============================================================
# 8. Main Training Loop
# ============================================================
@dataclass
class RLConfig:
    num_epochs: int = 1
    max_new_tokens: int = 40
    beta_div: float = 0.05


def train_bertscore_baseline(dataloader, config: RLConfig):
    model.train()
    print(f"Start BERTScore-Only RL Training... (Batch={BATCH_SIZE}, Accum={GRAD_ACCUMULATION}, FP16=On)")

    for epoch in range(config.num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{config.num_epochs} =====")
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            references = batch["references"]

            # (1) Generate Responses
            hyps, gen_ids = generate_responses(
                input_ids, attention_mask, max_new_tokens=config.max_new_tokens
            )

            # (2) Calculate Reward = BERTScore (shaped) + diversity bonus
            bert_rewards_shaped, bert_raw = compute_bertscore_rewards(hyps, references)
            diversity_scores = repetition_diversity_scores(hyps)

            total_rewards = bert_rewards_shaped + config.beta_div * diversity_scores
            total_rewards = total_rewards.to(device)

            # (3) Advantage Normalization
            mean_reward = total_rewards.mean()
            advantages = total_rewards - mean_reward

            # (4) Forward Pass & Loss Calculation (Mixed Precision)
            with torch.cuda.amp.autocast():
                # Log Probabilities
                log_probs = sequence_log_probs(gen_ids)

                # KL Penalty (prevents drifting too far from SFT)
                kl_loss = kl_penalty(gen_ids, coef=0.05)

                # PPO/Policy Gradient Loss
                loss = -(log_probs * advantages.detach()).mean() + kl_loss

                # Normalize loss for gradient accumulation
                loss = loss / GRAD_ACCUMULATION

            # (5) Backward Pass
            scaler.scale(loss).backward()

            # (6) Optimizer Step (Only every Accumulation steps)
            if (step + 1) % GRAD_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                print(
                    f"[Ep {epoch + 1} Step {step}] "
                    f"Loss: {loss.item() * GRAD_ACCUMULATION:.4f} | "
                    f"BERT (Raw): {bert_raw.mean():.4f} | "
                    f"R_Total: {mean_reward.item():.4f} | "
                    f"KL: {kl_loss.item():.4f}"
                )

    # Save Final Model
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete & Saved!")


if __name__ == "__main__":
    rl_config = RLConfig(
        num_epochs=1,
        max_new_tokens=40,
        beta_div=0.05
    )
    train_bertscore_baseline(dataloader, rl_config)