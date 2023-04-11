import torch

class SpellingCorrectionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=16):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, target_text = self.data[idx]
        input_encodings = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        #with tokenizer.as_target_tokenizer():
        target_encodings = self.tokenizer(target_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return {"input_ids": input_encodings['input_ids'].squeeze(),
                "attention_mask": input_encodings["attention_mask"].squeeze(),
                "labels": target_encodings['input_ids'].squeeze()}