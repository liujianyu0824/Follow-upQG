import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import json

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 加载预训练的 BART 模型和分词器
model_name = "/embedding/bart/large"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 准备数据集
with open("/data/train.json", "r") as f:
    train_data = json.load(f)

with open("/data/valid.json", "r") as f:
    dev_data = json.load(f)


train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)

# 定义数据预处理函数
def preprocess_function(examples):
    inputs = [f"{q} <SEP> {a}" for q, a in zip(examples["question"], examples["answer"])]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["follow-up"], max_length=128, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 对数据集应用预处理函数
train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="/baselines/results/bart",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
)

# 定义训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# 开始训练
trainer.train()

# 评估模型
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")

# 保存模型
trainer.save_model("/baselines/model/bart-follow-up-generator")
tokenizer.save_pretrained("/baselines/model/bart-follow-up-generator")

