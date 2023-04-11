import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
import argparse
import torch
from utils import load_data_from_file

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a finetuned Hugging Face model for spelling correction.')
    parser.add_argument('--test-data', type=str,  default='./test_data.json',   help='Full path for training data')
    parser.add_argument('--ckpt-path', type=str,  default='output/checkpoint-5000/pytorch_model.bin',   help='Full path for the model checkpoint')    
    parser.add_argument('--tokenizer', type=str,  default='facebook/bart-base',   help='The name of the tokenizer')    
    parser.add_argument('--model', type=str,  default='facebook/bart-base',   help='The name of the model')
    parser.add_argument('--max-length', type=int,  default=16,   help='Maximum context length')

    # Parse the arguments
    args = parser.parse_args()

    return args

def evaluate_model(args):
    # Load test_data

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    print(f"Number of parameters: {model.num_parameters():,}")

    # Load the checkpoint
    model.load_state_dict(torch.load(args.ckpt_path))
    print(f'Checkpoint loaded successfully: {args.ckpt_path}')

    spellchecker = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_length=args.max_length)

    while True:    
        query = input("Search query:")
        result = spellchecker(query)
        print(f'Corrected query: {result[0]["generated_text"]}')


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)