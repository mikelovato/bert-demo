import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. Load Pretrained Model and Tokenizer
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config=BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_8bit=True,  # Load in 8-bit for memory efficiency
    #quantization_config=quantization_config,
    device_map="auto",
)

# 2. Prepare Model for LoRA Training
model = prepare_model_for_kbit_training(model)

# 3. Define LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target modules for LoRA
    lora_dropout=0.05,  # Dropout rate for LoRA
    bias="none",  # Bias setting
    task_type="CAUSAL_LM"  # Task type (causal language modeling)
)

# Wrap model with LoRA
model = get_peft_model(model, lora_config)

# 4. Load and Preprocess Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Example dataset
tokenized_dataset = dataset.map(
    lambda samples: tokenizer(samples["text"], truncation=True, padding=True, max_length=512),
    batched=True
)

# 5. Define Data Collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM does not use masked language modeling
)

# 6. Define Training Arguments
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./llama2-lora-finetuned",  # Directory to save the model
    per_device_train_batch_size=4,  # Batch size per device
    gradient_accumulation_steps=8,  # Gradient accumulation
    num_train_epochs=3,  # Number of epochs
    logging_steps=10,  # Logging frequency
    save_steps=100,  # Model saving frequency
    evaluation_strategy="steps",  # Evaluation strategy
    eval_steps=50,  # Evaluation frequency
    save_total_limit=3,  # Max number of saved checkpoints
    learning_rate=2e-4,  # Learning rate
    fp16=True,  # Mixed precision training
    report_to="none"  # Disable reporting to tracking systems
)

# 7. Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)
print("trainer start running...")
# 8. Train the Model
trainer.train()

# 9. Save the Fine-Tuned Model
model.save_pretrained("./llama2-lora-finetuned")
tokenizer.save_pretrained("./llama2-lora-finetuned")
