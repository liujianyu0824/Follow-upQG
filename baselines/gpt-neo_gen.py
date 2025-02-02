from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import os 
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# # 加载模型
model = GPTNeoForCausalLM.from_pretrained("/baselines/model/gpt-neo")
tokenizer = GPT2Tokenizer.from_pretrained("/baselines/model/gpt-neo")

# 输入推理
# Copy test.json and rename it "gptneo_followupQ.json"
with open("/result/gptneo_followupQ.json", "r") as f:
        original_data = json.load(f)

# 设置生成参数
max_new_tokens = 16  # 生成文本的最大长度
num_return_sequences = 1  # 生成的文本数量

i = 0
# 输入文本
for data in original_data[:]:
    question = data['question']
    answer = data['answer']

    # 将问题和答案串联在一起
    input_text = f"{question}<SEP>{answer}<QUS>"

    # 对提示进行编码
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # 生成文本
    output = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences
    )

    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)


    original_data[original_data.index(data)]['generated_follow-up'] = [generated_text]

    with open("/result/gptneo_followupQ.json", "w") as f:
        json.dump(original_data, f, indent=4)

    i += 1
    print('i=', i)






