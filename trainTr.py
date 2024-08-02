from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer

# Load the converted model and tokenizer
model_name = "./trained_model/Saligned-mawps-single"  
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer from the same directory


model = AutoModelForMaskedLM.from_pretrained(model_name)

# Load and tokenize the dataset
dataset = load_dataset("afkfatih/turkishdataset")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# Set up data collator and training arguments
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

# Initialize the Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./lang/saligned_pretrained")
tokenizer.save_pretrained("./lang/saligned_pretrained")
