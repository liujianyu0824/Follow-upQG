import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the pre-trained T5 model and tokenizer
model_name = '/embedding/t5-large'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

model = model.cuda()

# Function to calculate the log-likelihood of a question given a passage
def calculate_log_likelihood(question, passage):
    input_text = f"{passage} Please write a question based on this passage."
    input_encoding = tokenizer( input_text,
                                padding='longest',
                                max_length=512,
                                pad_to_multiple_of=8,  
                                truncation=True,
                                return_tensors='pt')
    context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
    # input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    labels = tokenizer(question, 
                       max_length=128, 
                       truncation=True, 
                       return_tensors='pt').input_ids

    with torch.no_grad():
        logits = model(input_ids=context_tensor, attention_mask=attention_mask, labels=labels)
        # log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        log_likelihood = -logits.loss.item()

    return log_likelihood


def inference(question, passages,batch_size=32):
    all_ids = []
    for passage in passages:
        all_ids.append(f"Passage: {passage}. Please write a question based on this passage.")

    input_encoding = tokenizer(all_ids,
                                padding='longest',
                                max_length=512,
                                pad_to_multiple_of=8,
                                truncation=True,
                                return_tensors='pt')
    
    context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
    
    context_tensor = context_tensor.cuda()
    attention_mask = attention_mask.cuda()


    decoder_question = [f"Question: {question}"]
    target_encoding = tokenizer(decoder_question,
                                max_length=128,
                                truncation=True,
                                return_tensors='pt')
    decoder_prefix_tensor = target_encoding.input_ids.cuda()

    decoder_prefix_tensor = torch.repeat_interleave(decoder_prefix_tensor,
                                                    len(context_tensor),
                                                    dim=0)

    sharded_nll_list = []
    # print(context_tensor.device, attention_mask.device, decoder_prefix_tensor.device)
    # print(len(context_tensor))
    for i in range(0, len(context_tensor), batch_size):
        # 检索文本的编码向量,每一批次处理32个
        encoder_tensor_view = context_tensor[i: i + batch_size].cuda()
        attention_mask_view = attention_mask[i: i + batch_size].cuda()
        decoder_tensor_view = decoder_prefix_tensor[i: i + batch_size].cuda()

        # print(encoder_tensor_view.device, attention_mask_view.device, decoder_tensor_view.device)

        # 计算logP(q|zi)
        with torch.no_grad():
            logits = model(input_ids=encoder_tensor_view,
                                attention_mask=attention_mask_view,
                                labels=decoder_tensor_view).logits

        log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
        nll = - \
            log_softmax.gather(
                2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

        avg_nll = torch.sum(nll, dim=1)
        sharded_nll_list.append(avg_nll)

    # 重排检索的context，保存对应的scores及id
    topk_scores, indexes = torch.topk(-torch.cat(
        sharded_nll_list), k=len(context_tensor))
    topk_scores = topk_scores.tolist()
    indexes = indexes.tolist()

    return topk_scores, indexes

# Example passages and a question
# passages = [
#     "How to make scrambled eggs with tomatoes.", "Ming went to school today." , "The colours of the rainbow are red, orange, yellow, green, cyan, blue and purple.", "The sun rises in the east.", "The moon shines in the west.", "The stars are beautiful."
# ]
# question = "How many colours are there in the rainbow"

# # Calculate log-likelihoods for each passage

# topk_scores, indexes = inference(question, passages, batch_size=6)

# print("Ranked Passages:")
# for index in indexes:
#     print(passages[index])