# BBBS: A Hybrid Reward System for Instruction-Following Alignment

## Installation & Setup
Follow these steps to set up the environment on an HPC cluster (Linux/Slurm).

1. Create Conda Environment
It is recommended to use Python 3.10 or higher.

```commandline
# Create a new environment named 'bbbs'
conda create -n bbbs python=3.10 -y

# Activate the environment
conda activate bbbs
```


2. Install PyTorch (GPU Support)
CRITICAL STEP: Do not install PyTorch via the requirements.txt file initially. 
Install it directly via Conda to ensure the correct CUDA drivers are linked for your HPC GPUs (e.g., V100/A100).

```commandline
# For CUDA 12.1 (Recommended for newer HPCs)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# OR

# For CUDA 11.8 (If your HPC is older)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

3. Install Python Dependencies
Create a file named requirements.txt with the following content:

```text
pandas
transformers
bert-score
nltk
datasets
accelerate
scikit-learn
```

Then, install them using pip:

```commandline
pip install -r requirements.txt
```


## Running the Training

### Phase 1: Supervised Fine-Tuning (SFT)
Train the base model to learn the paraphrasing format using the QQP dataset.

```commandline
python train_sft.py
```

### Phase 2: RLHF Training
Train the each of those 3 models by specific command

**Option A: BERTScore model**

```commandline
python Bertscore_only_reward.py
```

**Option B: BLEU model**

```commandline
python BLEU_only_reward.py
```
Option C: BBBS(Bertscore + BLEU) reward model 

```commandline
python BBBS_reward.py
```

### Phase 3: Evaluation
Compare the SFT baseline against the RL-tuned models.

````commandline
python test_model.py
````


## Project Structure
* `train_sft.py`: Standard Supervised Fine-Tuning script.

* `Bertscore_only_reward.py`: Main RL training script for Bert score only model

* `BBBS_reward.py`: Main RL training script for BBBS hybrid reward model.

* `BLEU_only_reward.py`: RL training script using only BLEU reward.

* `test_model.py`: Script to generate side-by-side comparisons of model outputs.

* `qqp_subset_1000.csv`: The training dataset.