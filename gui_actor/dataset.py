from torch.utils.data import Dataset
from transformers import LayoutLMv3TokenizerFast
import torch

class GUIDataset(Dataset):
    def __init__(self, data, tokenizer: LayoutLMv3TokenizerFast):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item["words"]
        boxes = item["bboxes"]
        labels = item["patch_labels"]

        encoding = self.tokenizer(
            text=text,             
            boxes=boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=False,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        label_len = encoding["input_ids"].shape[0]
        padded_labels = labels + [0] * (label_len - len(labels))
        encoding["labels"] = torch.tensor(padded_labels[:label_len])

        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                encoding[k] = v.to(torch.long)

        return encoding

