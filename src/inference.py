import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "models/bert_imdb_model"
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    return label_map[pred], probs[0][pred].item()

if __name__ == "__main__":
    text = "This movie was absolutely fantastic!"
    label, confidence = predict_sentiment(text)
    print(f"Sentiment: {label} ({confidence:.2%})")
