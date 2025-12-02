import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_SFT_PATH = "./sft_model"
RL_BERT_PATH = "./output_rl_bertscore/final"
RL_BLEU_PATH = "./output_rl_bleu/final"
RL_BBBS_PATH = "./output_rl_model/final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_CASES = [
    "What is the best way to learn Python?",
    "How do I prepare for interviews?",
    "I am in the second year of my CSE and I want to crack GATE 2017. How do I start my preparation? What topics should I be more concentrated on?",
    "Why should we learn photography?"
]

def load_model(path, name):
    print(f"Loading {name} from {path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None


print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(BASE_SFT_PATH)
tokenizer.pad_token = tokenizer.eos_token

model_sft = load_model(BASE_SFT_PATH, "SFT Baseline")
model_bert = load_model(RL_BERT_PATH, "RL (BERTScore Only)")
model_bleu = load_model(RL_BLEU_PATH, "RL (BLEU Only)")
model_bbbs = load_model(RL_BBBS_PATH, "RL (BBBS Hybrid)")


def generate(model, sentence):
    if model is None: return "N/A"
    prompt = f"Paraphrase this question: {sentence}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=40, do_sample=True, top_p=0.9, temperature=0.6,
            pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Response:")[-1].strip() if "Response:" in text else text


print("\n" + "=" * 100)
print(f"{'Input Question':<40} | {'SFT':<25} | {'BERTScore Only':<25} | {'BLEU Only':<25} | {'BBBS (Hybrid)':<25} ")
print("=" * 100)

for q in TEST_CASES:
    out_sft = generate(model_sft, q)
    out_bert = generate(model_bert, q)
    out_bleu = generate(model_bleu, q)
    out_bbbs = generate(model_bbbs, q)

    print(f"Q: {q}")
    print(f"SFT : {out_sft}")
    print(f"BERT: {out_bert}")
    print(f"BLEU: {out_bleu}")
    print(f"BBBS: {out_bbbs}")
    print("-" * 100)