import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ['SAFETENSORS_DISABLE'] = '1'


# 初始化tokenizer和模型
tokenizer = T5Tokenizer.from_pretrained('/embedding/t5-base')
model = T5ForConditionalGeneration.from_pretrained('/embedding/t5-base')

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
    output_dir="/baselines/results/t5",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=3,
    load_best_model_at_end=True,
)

# 定义训练器
class CustomTrainer(Trainer):
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        # 确保模型的所有权重都是连续的
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        # 调用父类的保存方法
        super().save_model(output_dir, _internal_call)

# 使用自定义的Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# 开始训练
trainer.train()



# 保存模型
trainer.save_model("/home/liujianyu/llmgraph/baselines/model/t5-follow-up-generator")
tokenizer.save_pretrained("/home/liujianyu/llmgraph/baselines/model/t5-follow-up-generator")

