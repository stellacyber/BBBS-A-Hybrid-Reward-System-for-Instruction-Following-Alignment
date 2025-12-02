import os
import math
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bertscore_score
import torch.nn.functional as F

SFT_MODEL_PATH = "./sft_model"
OUTPUT_DIR = "./output_rl_model/final"

BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
RL_LEARNING_RATE = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
hf_logging.set_verbosity_error()

# ============================================================
# 1. Data Loading
# ============================================================
CSV_PATH = "qqp_subset_1000.csv"
df = pd.read_csv(CSV_PATH)
print("Raw dataset shape:", df.shape)

EXAMPLES = [
    {
        "instruction": row["instruction"],
        "reference": row["reference"],
    }
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
print(f"Loading SFT model from: {SFT_MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(device)
model.resize_token_embeddings(len(tokenizer))

ref_model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH).to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()


# ============================================================
# 2. Collate Function
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

smooth = SmoothingFunction().method3

def simple_tokenize(text: str):
    return text.lower().strip().split()


# ============================================================
# 3. Penalty Functions
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
        len_pen = -beta_len if length < 8 else 0.0
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
# 4. Reward Functions
# ============================================================
def compute_bleu_raw(hyps: List[str], refs: List[str]) -> torch.Tensor:
    scores = []
    for hyp, ref in zip(hyps, refs):
        hyp_tokens = simple_tokenize(hyp)
        ref_tokens = simple_tokenize(ref)
        if len(hyp_tokens) == 0:
            scores.append(0.0)
            continue
        bleu = sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth,
        )
        scores.append(float(bleu))
    return torch.tensor(scores, dtype=torch.float32)


def compute_bertscore_raw(hyps: List[str], refs: List[str]) -> torch.Tensor:
    P, R, F1 = bertscore_score(hyps, refs, lang="en", verbose=False, device=device)
    return F1.detach().float()


def compute_hybrid_rewards(hyps, refs, w_bleu=0.3, w_bert=0.7, beta_div=0.1):
    bleu_raw = compute_bleu_raw(hyps, refs)
    bert_raw = compute_bertscore_raw(hyps, refs)
    base = w_bleu * bleu_raw.cpu() + w_bert * bert_raw.cpu()
    shaped = apply_length_repetition_penalty(hyps, base)
    diversity = repetition_diversity_scores(hyps)

    total = shaped + beta_div * diversity
    return total, bleu_raw, bert_raw, diversity


# ============================================================
# 5. Generation & Probabilities
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
# 6. Training Loop
# ============================================================
@dataclass
class HybridConfig:
    num_epochs: int = 3
    max_new_tokens: int = 40
    w_bleu: float = 0.4
    w_bert: float = 0.6
    beta_div: float = 0.05


def train_hybrid_baseline(dataloader, config: HybridConfig):
    model.train()
    print(f"Start RL Training... (Batch={BATCH_SIZE}, Accum={GRAD_ACCUMULATION}, FP16=On)")

    for epoch in range(config.num_epochs):
        print(f"\n===== [Hybrid] Epoch {epoch + 1}/{config.num_epochs} =====")
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            references = batch["references"]

            hyps, gen_ids = generate_responses(
                input_ids,
                attention_mask,
                max_new_tokens=config.max_new_tokens,
            )

            total_rewards, bleu_raw, bert_raw, div_scores = compute_hybrid_rewards(
                hyps,
                references,
                w_bleu=config.w_bleu,
                w_bert=config.w_bert,
                beta_div=config.beta_div,
            )
            total_rewards = total_rewards.to(device)

            mean_reward = total_rewards.mean()
            advantages = total_rewards - mean_reward

            with torch.cuda.amp.autocast():
                # log p(y|x)
                log_probs = sequence_log_probs(gen_ids)
                # KL penalty
                kl_loss = kl_penalty(gen_ids, coef=0.05)
                # PPO Loss
                loss = -(log_probs * advantages.detach()).mean() + kl_loss

                # gradient accumulate：Loss needs to remove steps
                loss = loss / GRAD_ACCUMULATION

            # back propagation(Scaler)
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                print(
                    f"[Ep {epoch + 1} Step {step}] "
                    f"Loss: {loss.item() * GRAD_ACCUMULATION:.4f} | "
                    f"R_Total: {mean_reward.item():.4f} | "
                    f"BLEU: {bleu_raw.mean().item():.4f} | "
                    f"BERT: {bert_raw.mean().item():.4f} | "
                    f"KL: {kl_loss.item():.4f}"
                )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Saving final model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Done! model saved..")


if __name__ == "__main__":
    hybrid_config = HybridConfig(
        num_epochs=1,
        max_new_tokens=40,
        w_bleu=0.4,
        w_bert=0.6,
        beta_div=0.05,
    )
    train_hybrid_baseline(dataloader, hybrid_config)