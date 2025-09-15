import json
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TextStreamer
from unsloth.trainer import UnslothDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import is_bf16_supported
from transformers import TrainingArguments

# 1. Load dữ liệu JSON Moozi
with open("data.json", "r", encoding="utf-8") as f:
    items = json.load(f)["items"]

def format_item(item):
    return {
        "prompt": f"Hãy trả lời dựa trên knowledge base.\nCâu hỏi: {item['title']}\n",
        "response": item["text"]
    }

dataset = Dataset.from_list([format_item(it) for it in items])

# 2. Load base model Llama-3.2-3B-Instruct (4-bit nếu muốn tiết kiệm tài nguyên)
model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    max_seq_length=2048,
    dtype=None,
    device_map="auto",
)

# 3. (Optional) Sử dụng LoRA để fine-tune hiệu quả hơn — tiết kiệm VRAM, nhanh hơn
from unsloth import FastLanguageModel as FLM
model = FLM.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
)

# 4. Thiết lập huấn luyện
training_args = TrainingArguments(
    output_dir="moozi_llama32_3b_finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="response",
    max_seq_length=1024,
    args=training_args,
)

trainer.train()
trainer.save_model("moozi_llama32_3b_finetune")
tokenizer.save_pretrained("moozi_llama32_3b_finetune")

print("✅ Fine-tune xong! Lưu model tại moozi_llama32_3b_finetune/")