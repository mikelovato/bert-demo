from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    attention_weights = outputs.attentions
    loss = outputs.loss

    return jsonify({"predictions": predictions, "weights":attention_weights, "loss": loss})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6372)