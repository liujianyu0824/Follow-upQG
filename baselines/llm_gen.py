import json
import re
import httpx
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8002/v1"

# Copy test.json and rename it "glm4_followupQ.json"
with open("/result/glm4_followupQ.json", "r") as f:
    original_data = json.load(f)

def extract_json(passage):
    pattern = re.compile(r'({.*?(\n}))', re.DOTALL)
    match = pattern.search(passage)

    extracted_content = match.group(1)
    passage = json.loads(extracted_content)
    return passage


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


i = 0
for data in original_data[:]:
# LLM-Research/Meta-Llama-3-8B-Instruct
    followup_question = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 'I will provide you with a Question-Answer pair. The content is as follows:\n"""\n' + '"Question":' + data["question"] + '\n"Answer": ' + data["answer"]  + '''\n"""\nBased on these information, please raise a follow-up question which related to the given "Related Knowledge".Please return the result in json:
                {
                    "followup_question":[]
                }
                You only need to return json data, no extra content is needed.'''},
        ]
    )
    # print(followup_question.choices[0].message.content)

    sign = False
    while sign == False:
        followup_question = client.chat.completions.create(
            model="glm-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": 'I will provide you with a Question-Answer pair. The content is as follows:\n"""\n' + '"Question":' + data["question"] + '\n"Answer": ' + data["answer"]  + '''\n"""\nBased on these information, please raise a follow-up question which related to the given "Related Knowledge".Please return the result in json:
                    {
                        "followup_question":[]
                    }
                    You only need to return json data, no extra content is needed.'''},
            ]
        )
        try:
            followup_question = extract_json(followup_question.choices[0].message.content)
            sign = True
        except:
            continue



    while len(followup_question['followup_question']) == 0:
        followup_question = client.chat.completions.create(
            model="glm-4",
            messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": 'I will provide you with a Question-Answer pair. The content is as follows:\n"""\n' + '"Question":' + data["question"] + '\n"Answer": ' + data["answer"]  + '''\n"""\nBased on these information, please raise a follow-up question which related to the given "Related Knowledge".Please return the result in json:
                            {
                                "followup_question":[]
                            }
                            You only need to return json data, no extra content is needed.'''},
                    ]
                )
        followup_question = extract_json(followup_question.choices[0].message.content)

    # print(followup_question)

    original_data[original_data.index(data)]['generated_follow-up'] = followup_question['followup_question']

    with open("/result/glm4_followupQ.json", "w") as f:
        json.dump(original_data, f, indent=4)

    i += 1
    print('i=', i)