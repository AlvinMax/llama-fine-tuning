import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Load the 7b mistral model
model_id = "mistralai/Mistral-7B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# Set it to a new token to correctly attend to EOS tokens.
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config)

train_dataset = load_dataset("stingning/ultrachat", split="train[:1%]")
# noinspection PyTypeChecker
training_arguments = TrainingArguments(
    output_dir="/tmp/llama-7b-qlora-ultrachat",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=10,
    logging_steps=1,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    max_steps=30,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    gradient_checkpointing=True,
    report_to="none",  # report_to="wandb"
)


def formatting_func(example):
    text = f"### USER: {example['data'][0]}\n### ASSISTANT: {example['data'][1]}"
    return text


# noinspection PyTypeChecker
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    packing=True,
    dataset_text_field="id",
    tokenizer=tokenizer,
    max_seq_length=1024,
    formatting_func=formatting_func,
)
trainer.train()
