import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os

# ===========================================
BATCH_SIZE = 8
LEARNING_RATE = 2e-5 
EPOCHS = 8
MODEL_NAME = "gpt2-medium"
OUTPUT_DIR = "./sft_model"
# ===========================================

if not torch.cuda.is_available():
    print("Warning: not detected GPU")
    device = "cpu"
else:
    print(f"Connected GPU: {torch.cuda.get_device_name(0)}")
    device = "cuda"

df = pd.read_csv("qqp_subset_1000.csv")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

class SFTDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.inputs = []
        for _, row in df.iterrows():
            text = f"{row['instruction']}\nResponse: {row['reference']}{tokenizer.eos_token}"
            self.inputs.append(text)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

def collate_fn(batch):
    return tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")

dataset = SFTDataset(df, tokenizer)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler() 

model.train()
print(f"Start SFT Training (HPC Mode | Batch={BATCH_SIZE})...")

for epoch in range(EPOCHS):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if step % 10 == 0:
            print(f"Epoch {epoch+1} Step {step} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"SFT Model saved at {OUTPUT_DIR}")




