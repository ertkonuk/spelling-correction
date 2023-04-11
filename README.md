# spelling-correction
Finetune Bart model for spelling correction

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
