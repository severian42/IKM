from openai import OpenAI
from params import OPENAI_MODEL, OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def send_to_chatgpt(msg_list):
    completion = client.chat.completions.create(model=OPENAI_MODEL, temperature=0.5, messages=msg_list)
    # Assuming the correct way to access the message content is through attribute access
    chatgpt_response = completion.choices[0].message.content  # Changed from ["content"] to .content
    chatgpt_usage = completion.usage
    return chatgpt_response, chatgpt_usage

def send_to_llm(provider, msg_list):
    if provider == "openai":
        response, usage = send_to_chatgpt(msg_list)
