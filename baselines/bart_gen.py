from transformers import BartTokenizer, BartForConditionalGeneration
import os 
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def generate_text(model, tokenizer, input_text, max_length=100):
    input_encoding = tokenizer(
        input_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = input_encoding['input_ids']
    attention_mask = input_encoding['attention_mask']

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    # 指定训练后的模型保存路径
    model_path = '/baselines/model/bart-follow-up-generator'

    # 加载训练后的模型和tokenizer
    tokenizer = BartTokenizer.from_pretrained('/baselines/model/bart-follow-up-generator')
    model = BartForConditionalGeneration.from_pretrained(model_path)

    # Copy test.json and rename it "bart_followupQ.json"
    with open("/result/bart_followupQ.json", "r") as f:
        original_data = json.load(f)

    i = 0
    # 输入文本
    for data in original_data[:]:
        question = data['question']
        answer = data['answer']
    
        # 将问题和答案串联在一起
        input_text = f"{question} <SEP> {answer}"

        # 生成文本
        followup_question = generate_text(model, tokenizer, input_text)

        original_data[original_data.index(data)]['generated_follow-up'] = [followup_question]

        with open("/result/bart_followupQ.json", "w") as f:
            json.dump(original_data, f, indent=4)

        i += 1
        # print('i=', i)

if __name__ == '__main__':
    main()