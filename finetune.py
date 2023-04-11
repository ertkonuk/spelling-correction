import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import argparse
import torch
from utils import load_data_from_file, freeze_params
from dataset import SpellingCorrectionDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser(description='Finetune a Hugging Face model for spelling correction.')
    parser.add_argument('--train-data', type=str,  default='./train_data.json',   help='Full path for training data')
    parser.add_argument('--val-data', type=str,  default='./val_data.json',   help='Full path for validation data')
    parser.add_argument('--output-dir', type=str,  default='output',   help='Output directory')
    parser.add_argument('--logging-dir', type=str,  default='logs',   help='Loggind directory')
    parser.add_argument('--tokenizer', type=str,  default='facebook/bart-base',   help='The name of the tokenizer')
    parser.add_argument('--optimizer', type=str,  default='adamw_torch_fused',   help='The name of the optimizer')
    parser.add_argument('--model', type=str,  default='facebook/bart-base',   help='The name of the model')
    parser.add_argument('--lr_scheduler_type', type=str,  default='constant',   help='Learning rate scheduler')
    parser.add_argument('--max-length', type=int,  default=16,   help='Maximum context length')
    parser.add_argument('--num-epochs', type=int,  default=3,   help='Number of epochs')
    parser.add_argument('--lr', type=float,  default=1e-4,   help='Learning rate')
    parser.add_argument('--batch-size', type=int,  default=256,   help='Train batch size')
    parser.add_argument('--eval-batch-size', type=int,  default=256,   help='Validation batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int,  default=2,   help='Number of steps for gradient accumulation.')
    parser.add_argument('--save-steps', type=int,  default=1000,   help='Save model interval')    
    parser.add_argument('--logging-steps', type=int,  default=100,   help='Logging interval')    

    parser.add_argument('--freeze-encoder', help='Freeze the encoder', action='store_true')
    parser.add_argument('--freeze-decoder', help='Freeze the decoder', action='store_true')
    parser.add_argument('--torch-compile', help='Compile the model (only in PyTorch 2.0 and above)', action='store_true')
    parser.add_argument('--bf16', help='Use bf16 training', action='store_true')


    # Parse the arguments
    args = parser.parse_args()

    return args

def train(args):
    # Load train_data and val_data
    train_pairs = load_data_from_file(args.train_data)
    val_pairs   = load_data_from_file(args.val_data)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    train_data = SpellingCorrectionDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_data   = SpellingCorrectionDataset(val_pairs  , tokenizer, max_length=args.max_length)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    print(f"Number of parameters: {model.num_parameters():,}")

    # Data collator for Seq2Seq tasks: shifts the decoder input to the right by one position
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    if args.freeze_encoder:
        freeze_params(model.get_encoder())

    if args.freeze_decoder:
        freeze_params(model.get_decoder())


    # Define training arguments and instantiate the trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        disable_tqdm=False,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        # PyTorch 2.0 specifics
        bf16=args.bf16, # bfloat16 training
    	torch_compile=args.torch_compile, # optimizations
        optim=args.optimizer, # improved optimizer    
    )    

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=seq2seq_data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,   
    )

    # Fine-tune the model
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    train(args)