import torch
from transformers import BertTokenizer, BertForSequenceClassification

class SentimentAnalyzer:
    def __init__(self):
        model_path = "saved_models/bert_epoch_0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.label_map = {0: "中性", 1: "高兴", 2: "生气", 3: "伤心", 4: "恐惧", 5: "惊讶"}

    def predict(self, text, label=None):
        self.model.eval()

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predict_label = torch.argmax(logits, dim=-1).item()
            predict_confidence = torch.max(torch.softmax(logits, dim=-1)).item()
        
        
        info = f"inputs: '{text}', predict: '{self.label_map[predict_label]}'"
        if label is not None:
            info += f" , label: '{self.label_map[label]}'"
        print(info)
        return self.label_map[predict_label], predict_confidence