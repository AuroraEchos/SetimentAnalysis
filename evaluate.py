import torch
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
import logging
from tqdm import tqdm
from preprocess import process_dataset

logging.basicConfig(level=logging.INFO)

def evaluate(model, dataloader, device, metric):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = metric(all_labels, all_preds)
    return accuracy

if __name__ == "__main__":
    
    model_path = 'saved_models/bert_epoch_0'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset_val = process_dataset("data/web_val.tsv", tokenizer, shuffle=False)
    dataset_test = process_dataset("data/web_test.tsv", tokenizer, shuffle=False)

    val_accuracy = evaluate(model, dataset_val, device, accuracy_score)
    logging.info('Validation accuracy: %.6f', val_accuracy)

    test_accuracy = evaluate(model, dataset_test, device, accuracy_score)
    logging.info('Test accuracy: %.6f', test_accuracy)
