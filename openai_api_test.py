from openai import OpenAI
from typing import List, Dict

# Initialize OpenAI API with custom base URL and key
openai_api_key = "tubitak"
openai_api_base = "http://10.15.52.20:1234/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_llm_response(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model="atahanuz/dpo",
        stream=False,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract and return only the content from the first choice
    return response.choices[0].message.content
messages = [
    {"role": "system", "content": "Sen kullanıcının talimatlarını kısaca yerine getiren bir yapay zeka asistanısın. Yalnızca talimati yerine getir fazladan yazma."},
    {"role": "user", "content": "merhaba"}
]

# Call the function and print the content of the response
print(get_llm_response(messages))
