from transformers import AutoTokenizer
from dataset import Prot2TextInstructDataset
import os

SPLIT = "train"  # run script for "eval" and "test" as well
CSV_DIR = "data/"
DATA_ROOT_DIR = "data/"
# LLAMA_DIR = "meta-llama/Meta-Llama-3.1-8B-Instruct-hf"
LLAMA_DIR = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ESM_DIR = "facebook/esm2_t36_3B_UR50D"

split_dataset = Prot2TextInstructDataset(
    alphafold_version="6",
    root_dir=os.path.join(DATA_ROOT_DIR, SPLIT),
    csv_path=os.path.join(CSV_DIR, f"{SPLIT}.csv"),
    sequence_tokenizer=AutoTokenizer.from_pretrained(ESM_DIR),
    description_tokenizer=AutoTokenizer.from_pretrained(LLAMA_DIR, pad_token='<|reserved_special_token_0|>'),
    skip_download=False,
    skip_reload=False, 
    sample=10,
)
split_dataset.process_text()