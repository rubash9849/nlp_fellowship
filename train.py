import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer,EarlyStoppingCallback
from sklearn import preprocessing
from transformers  import AutoTokenizer, AutoModelForSequenceClassification
import re
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from sklearn.preprocessing import LabelEncoder


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def clean_text(text):
  contractions_mapping = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "I'm": "I am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is"
    # Add more contractions as needed
    }
  for contraction, expansion in contractions_mapping.items():
        text = text.replace(contraction, expansion)
  cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)  # Replace non-letters with spaces
  cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
  cleaned_text = cleaned_text.strip()  # Remove leading and trailing spaces
  cleaned_text = cleaned_text.lower()  # Convert to lowercase
  return cleaned_text


def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred,average='macro',zero_division=0)
    precision = precision_score(y_true=labels, y_pred=pred,average='macro',zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=pred,average='macro',zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}   

def train_save():
    data = pd.read_csv("dta.csv", encoding="ISO-8859-1")
    data.columns=['sentiment','id','date','query','username','text']
    data['cleaned_text'] = data['text'].apply(clean_text)
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()
    data['sentiment']= label_encoder.fit_transform(data['sentiment'])
    
    tokenizer =  AutoTokenizer.from_pretrained("bert-base-uncased")
    model =  AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",ignore_mismatched_sizes=True,num_labels=2)

    #tokenizer.pad_token = tokenizer.eos_token
    model.to('cuda')
    
    X = list(data["cleaned_text"])
    y = list(data["sentiment"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2,stratify=y)
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.0)
    args = TrainingArguments(
        output_dir="output-llm-t5",
        num_train_epochs=15,
        per_device_train_batch_size=32,
        save_total_limit=2,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,

    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]
        
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model('output-bert')


if __name__=='__main__':
    train_save()




