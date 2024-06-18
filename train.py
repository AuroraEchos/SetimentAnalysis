import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

import os
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

from preprocess import process_dataset

class MyCallBack:
    def __init__(self):
        self.loss = 0

    def log_epoch_end(self, epoch, avg_loss):
        logging.info('Avg loss at epoch %d: %.6f', epoch, avg_loss)

model_path = 'bert_model_files'
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


dataset_train = process_dataset("data/web_train.tsv", tokenizer)
dataset_val = process_dataset("data/web_val.tsv", tokenizer)
dataset_test = process_dataset("data/web_test.tsv", tokenizer, shuffle=False)


optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 10
num_training_steps = num_epochs * len(dataset_train)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

metric = accuracy_score
my_callback = MyCallBack()

save_directory = "data/saved_models"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def save_model(model, tokenizer, epoch, path):
    model.to("cpu") 
    save_path = os.path.join(path, f"bert_epoch_{epoch + 1}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    model.to(device)
    print(f"Model and tokenizer saved to {save_path}")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataset_train), total=len(dataset_train), desc=f'Epoch {epoch + 1}/{num_epochs}')
    for step, batch in progress_bar:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        progress_bar.set_postfix({'Avg Loss': total_loss / (step + 1)})
    
    avg_loss = total_loss / len(dataset_train)
    my_callback.log_epoch_end(epoch, avg_loss)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataset_val:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_accuracy = metric(all_labels, all_preds)
    logging.info('Validation accuracy after epoch %d: %.6f', epoch, val_accuracy)

    save_model(model, tokenizer, epoch, "saved_models")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in dataset_test:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_accuracy = metric(all_labels, all_preds)
logging.info('Test accuracy: %.6f', test_accuracy)