# LoRA 微调 Qwen1.5-1.8B 模型：自建 QA 数据集
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.integrations import TensorBoardCallback
from transformers import DataCollatorWithPadding
import os, json

# 设置基础路径与环境
model_name = "Qwen/Qwen1.5-1.8B"
os.environ["WANDB_DISABLED"] = "true"  # 可选，防止自动调用 wandb

with open("../data/train.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

dataset = Dataset.from_list(qa_data)

# Tokenizer 加载 + 构建 tokenized 数据集
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
data_collator = DataCollatorWithPadding(tokenizer)
tokenizer.pad_token = tokenizer.eos_token  # 必须设置 pad_token

def preprocess(example):
    full_input = f"用户: {example['prompt']}\n助手: {example['answer']}"
    tokenized = tokenizer(
        full_input,
        truncation=True,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


tokenized_dataset = dataset.map(preprocess)

# 加载 4bit Qwen 模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 启用 4bit 加载
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# LoRA 参数配置
with open("../configs/lora_config.json", 'r') as f:
    config_dict = json.load(f)

peft_config = LoraConfig(**config_dict)

# LoRA 模型构建（注入 adapter）
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
model.print_trainable_parameters()

# TrainingArguments 设置
training_args = TrainingArguments(
    output_dir="../results/checkpoints",
    per_device_train_batch_size=1,  # 显存限制，保持不变
    gradient_accumulation_steps=8,  # 提高等效batch size，提升稳定性
    num_train_epochs=15,  # 数据不多，10轮足够，100轮会过拟合且耗时
    learning_rate=1e-4,  # 更稳健的学习率
    save_strategy="epoch",
    save_total_limit=3,  # 最多只保存1个检查点（即最后一个或最佳）
    metric_for_best_model="loss",  # 按 loss 选择最优
    greater_is_better=False,
    logging_steps=5,
    logging_dir="../results/logs",
    fp16=True,
    report_to="none"
)

#  构建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[TensorBoardCallback()],
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存 Adapter（PEFT 格式）
model.save_pretrained("../results/adapter")
tokenizer.save_pretrained("../results/adapter")

print("训练完成，模型已保存")