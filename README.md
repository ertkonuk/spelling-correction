# Spell Checker with Transformers
This repo use HuggingFace to finetune a Seq2SeqLM transformer model for spelling correction. 

This is a work in progress. 

## Usage
First, generate a spelling-correction dataset from a text file that contains samples without spelling errors:

    ```
    python generate_data.py \
            --file-path ./data.json \
            --output-dir ./output \
            --max-length 16 \
            --error-rate 0.98
    ```
    
To finetune a pretrained model:

    ```
    python finetune.py \
            --train-data ./train_data.json \
            --val-data ./val_data.json \
            --output-dir ./output \
            --logging-dir ./logs \
            --tokenizer 'facebook/bart-base' \
            --model 'facebook/bart-base' \
            --optimizer adamw_torch_fused \
            --max-length 16 \
            --lr 1e-4 \
            --batch-size 256 \
            --eval-batch-size 256 \
            --gradient-accumulation-steps 2 \
            --bf16 \
            --torch-compile
    ```
    
  To interact with the finetuned model:

    ```
    python evaluate.py \
            --ckpt-path ./output/checkpoint-<#step>/pytorch_model.bin \
            --tokenizer 'facebook/bart-base' \
            --model 'facebook/bart-base' \
            --max-length 16
    ```
