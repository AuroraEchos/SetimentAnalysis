from torch.utils.data import Dataset, DataLoader
import torch

class SentimentDataset(Dataset):
    """Sentiment Dataset"""

    def __init__(self, path):
        self.path = path
        self._labels, self._text_a = [], []
        self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            dataset = f.read()
        lines = dataset.split("\n")
        for line in lines[1:-1]:
            label, text_a = line.split("\t")
            self._labels.append(int(label))
            self._text_a.append(text_a)

    def __getitem__(self, index):
        return self._labels[index], self._text_a[index]

    def __len__(self):
        return len(self._labels)

def process_dataset(source, tokenizer, max_seq_len=64, batch_size=32, shuffle=True):
    def collate_fn(batch):
        labels, texts = zip(*batch)
        encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        labels = torch.tensor(labels, dtype=torch.long)
        return input_ids, attention_mask, labels

    dataset = SentimentDataset(source)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader
