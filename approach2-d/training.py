from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
from transformers import DataCollatorWithPadding, AutoTokenizer,TrainingArguments, Trainer
from model import RobertaForSpanCategorization
import numpy as np

MAX_LENGTH = 512
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def get_tags():
    list_tags = []
    ds = Dataset.from_json("./training2.jsonlines")
    for row in ds:
        tags = row["tags"]
        for t in tags:
            list_tags.append(t["tag"])
    return list(set(list_tags))
    

def get_token_role_in_span(token_start: int, token_end: int, span_start: int, span_end: int):
    """
    Check if the token is inside a span.
    Args:
      - token_start, token_end: Start and end offset of the token
      - span_start, span_end: Start and end of the span
    Returns:
      - "B" if beginning
      - "I" if inner
      - "O" if outer
      - "N" if not valid token (like <SEP>, <CLS>, <UNK>)
    """
    if token_end <= token_start:
        return "N"
    if token_start < span_start or token_end > span_end:
        return "O"
    if token_start > span_start:
        return "I"
    else:
        return "B"



def tokenize_and_adjust_labels(sample):
    """
    Args:
        - sample (dict): {"id": "...", "text": "...", "tags": [{"start": ..., "end": ..., "tag": ...}, ...]
    Returns:
        - The tokenized version of `sample` and the labels of each token.
    """
    # Tokenize the text, keep the start and end positions of tokens with `return_offsets_mapping` option
    # Use max_length and truncation to ajust the text length
    tokenized = tokenizer(sample["text"], 
                          return_offsets_mapping=True, 
                          padding="max_length", 
                          max_length=MAX_LENGTH,
                          truncation=True)
    
    # We are doing a multilabel classification task at each token, we create a list of size len(label2id)=13 
    # for the 13 labels
    labels = [[0 for _ in label2id.keys()] for _ in range(MAX_LENGTH)]
    
    # Scan all the tokens and spans, assign 1 to the corresponding label if the token lies at the beginning
    # or inside the spans
    for (token_start, token_end), token_labels in zip(tokenized["offset_mapping"], labels):
        for span in sample["tags"]:
            role = get_token_role_in_span(token_start, token_end, span["start"], span["end"])
            if role == "B":
                token_labels[label2id[f"B-{span['tag']}"]] = 1
            elif role == "I":
                token_labels[label2id[f"I-{span['tag']}"]] = 1
    
    return {**tokenized, "labels": labels}

def divide(a: int, b: int):
    return a / b if b > 0 else 0
    
def compute_metrics(p):
    global n_labels
    """
    Customize the `compute_metrics` of `transformers`
    Args:
        - p (tuple):      2 numpy arrays: predictions and true_labels
    Returns:
        - metrics (dict): f1 score on 
    """
    # (1)
    predictions, true_labels = p
    
    # (2)
    predicted_labels = np.where(predictions > 0, np.ones(predictions.shape), np.zeros(predictions.shape))
    metrics = {}
    
    # (3)
    cm = multilabel_confusion_matrix(true_labels.reshape(-1, n_labels), predicted_labels.reshape(-1, n_labels))
    
    # (4) 
    for label_idx, matrix in enumerate(cm):
        if label_idx == 0:
            continue # We don't care about the label "O"
        tp, fp, fn = matrix[1, 1], matrix[0, 1], matrix[1, 0]
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * precision * recall, precision + recall)
        metrics[f"f1_{id2label[label_idx]}"] = f1
        
    # (5)
    macro_f1 = sum(list(metrics.values())) / (n_labels - 1)
    metrics["macro_f1"] = macro_f1
        
    return metrics
    
total_tags = get_tags()
tag2id = {}
for i, t in enumerate(total_tags):
    tag2id[t] = i + 1
    
label2id = {
    'O': 0, 
    **{f'B-{k}': 2*v - 1 for k, v in tag2id.items()},
    **{f'I-{k}': 2*v for k, v in tag2id.items()}
}

id2label = {v:k for k, v in label2id.items()}

fine_dataset = Dataset.from_json("training2.jsonlines")
X, Y = train_test_split(fine_dataset, test_size=0.2, random_state=42)

train_ds = Dataset.from_dict(X)
val_ds = Dataset.from_dict(Y)
tokenized_train_ds = train_ds.map(tokenize_and_adjust_labels, remove_columns=train_ds.column_names)
tokenized_val_ds = val_ds.map(tokenize_and_adjust_labels, remove_columns=val_ds.column_names)


data_collator = DataCollatorWithPadding(tokenizer, padding=True)
n_labels = len(id2label)

training_args = TrainingArguments(
    output_dir="./models/roberta_meddoplace_overlap",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_steps = 100,
    save_strategy='epoch',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    log_level='critical',
    seed=12345
)

def model_init():
    # For reproducibility
    #return TransformersCRF(model.config)
    #return RoBERTaBiLSTMCRF(num_labels=21, hidden_dim=512, lstm_layers=1, dropout_rate=0.1)
    return RobertaForSpanCategorization.from_pretrained("xlm-roberta-base", id2label=id2label, label2id=label2id, num_labels=21)
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_val_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("test_pc")