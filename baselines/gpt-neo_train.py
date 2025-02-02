from transformers import GPTNeoForCausalLM, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import os
import json

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 定义特殊标记
SEP_TOKEN = "<SEP>"
QUS_TOKEN = "<QUS>"

# 加载预训练的GPT-Neo模型和分词器
model = GPTNeoForCausalLM.from_pretrained("/embedding/gpt-neo/125M")
tokenizer = GPT2Tokenizer.from_pretrained("/embedding/gpt-neo/125M")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# 添加特殊标记到分词器的词汇表中
special_tokens_dict = {"sep_token": SEP_TOKEN, "additional_special_tokens": [QUS_TOKEN]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# 准备训练数据
def prepare_data(text1, text2, text3):
    return f"{text1}{SEP_TOKEN}{text2}{QUS_TOKEN}{text3}"

# 读取训练和验证数据
with open("/data/train.json", "r") as f:
    train_data = json.load(f)

with open("/data/valid.json", "r") as f:
    dev_data = json.load(f)

# 准备训练和验证文本
train_texts = []
dev_texts = []

for data in train_data:
    text1 = data["question"]
    text2 = data["answer"]
    text3 = data["follow-up"]
    train_texts.append(prepare_data(text1, text2, text3))

for data in dev_data:
    text1 = data["question"]
    text2 = data["answer"]
    text3 = data["follow-up"]
    dev_texts.append(prepare_data(text1, text2, text3))

# 将文本转换为token
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=512)

# 创建自定义数据集
train_dataset = Dataset.from_dict(train_encodings)
dev_dataset = Dataset.from_dict(dev_encodings)

# 创建数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)


training_args = TrainingArguments(
    output_dir="/baselines/results/gpt-neo",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=10,
    save_total_limit=3,
    load_best_model_at_end=True,
)


# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

# 开始训练
trainer.train()

# def make_contiguous(model):
#     for param in model.parameters():
#         if not param.is_contiguous():
#             param.data = param.contiguous()

# # 修改保存模型的部分
# make_contiguous(model)  # 保存前确保所有张量是连续的


# 保存模型和分词器
trainer.save_model("/baselines/model/gpt-neo-follow-up-generator")
tokenizer.save_pretrained("/baselines/model/gpt-neo-follow-up-generator")
