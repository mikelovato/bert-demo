from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import load_dataset
# from torchviz import make_dot

# Load a dataset
dataset = load_dataset("imdb")  # IMDB sentiment classification dataset
train_data = dataset["train"].select(range(512))
test_data = dataset["test"].select(range(512))

# check the size of the dataset

# map them
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=128)
train_data = train_data.map(preprocess, batched=True)
test_data = test_data.map(preprocess, batched=True)

# set format to torch
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# train
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.08,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)
print("trainer start running...")
trainer.train()

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print(model)
# dummy_input = {"input_ids": torch.randint(0, 1000, (1, 128)), "attention_mask": torch.ones(1, 128)}
# output = model(**dummy_input)
# make_dot(output.logits, params=dict(model.named_parameters())).render("model_structure", format="png")

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained('./fine_tuned_model')
tokenizer = BertTokenizer.from_pretrained('./fine_tuned_model')

app = Flask(__name__)

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
