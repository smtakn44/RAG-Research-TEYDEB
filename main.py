import re
from openai import OpenAI
import json
from typing import List, Dict
#  from multi_web_search_agent import web_search_agent
from database_agent import database_agent

openai_api_key = "tubitak"
openai_api_base = "http://10.15.52.20:1234/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_llm_response(messages: List[Dict[str, str]]) -> str:
    """Get LLM response based on provided messages."""
    response = client.chat.completions.create(
        messages=messages,
        model="atahanuz/dpo",
        stream=False,
        temperature=0.6,
        top_p=0.9,
    )
    return response.choices[0].message.content

def extract_json_from_response(response: str) -> str:
    """Extracts the JSON part from the response text."""
    try:
        # Use regex to find the JSON block within the response
        json_str = re.search(r'(\{.*\})', response, re.DOTALL).group(1)
        return json_str
    except AttributeError:
        raise ValueError("No valid JSON found in the response")

def react_agent(main_question: str, max_iterations: int = 0) -> str:
    """Core agent logic that handles querying database or web agents and returns the final answer."""
    context = ""          
    result = database_agent(main_question)
    context += f"First database search results from main question:\n{result}"
    print("context= \n", context)
    print("Context updated with first database search")
    
    for i in range(max_iterations):
        # Get LLM's thoughts and next action
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant. Analyze the question, think about the next step, decide on an agent and generate a sub question for this agent."
            },
            {
                "role": "user", 
                "content": f"Main Question: {main_question}\nContext so far: {context}\n\nThink about the next step and decide on an agent. Valid agents are: 'database_search', 'finish'. \
                            Respond in JSON format with 'thought', 'agent', and 'sub_question' keys. For 'database_search', produce a proper sub_question according to context. \
                            Only choose 'finish' agent if enough context is available.  Don't write anything other than JSON format."
            }
        ]

        response = get_llm_response(messages)
        
        try:
            # Extract JSON from the response
            json_str = extract_json_from_response(response)
            reaction = json.loads(json_str)
            thought, agent, sub_question = reaction['thought'], reaction['agent'], reaction['sub_question']
        except (json.JSONDecodeError, ValueError, KeyError):
            print(f"Error: Invalid LLM response format. Full response: {response}")
            continue 

        print(f"Iteration {i+1}:")
        print(f"Thought: {thought}")
        print(f"Agent: {agent}")
        print(f"Sub question: {sub_question}")
        print("\n")

        if agent == 'database_search':           
            result = database_agent(sub_question)
            context += f"Iteration {i+1}\nThought: {thought}\nAgent: {agent}\nSub question: {sub_question}\nDatabase search results: {result}"
            print("Context updated with database search")
        elif agent == 'finish':
            final_answer = get_llm_response([
                {
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Provide a final answer to only Main Question based on the given context in Turkish."
                },
                {
                    "role": "user", 
                    "content": f"Main Question: {main_question}\nContext: {context}\n\nProvide a final answer to the question in Turkish. \
                                 If the context doesn't provide enough information, say 'Soruyu cevaplamak için yeterli bilgi yok.'"
                }
            ])
            return final_answer
        else:
            return f"Error: Invalid agent '{agent}'"
        # print("context= ", context)
    
    print("Maximum iterations reached, providing final answer.")
    final_answer = get_llm_response([
        {
            "role": "system", 
            "content": "You are a helpful AI assistant. Provide a final answer to only Main Question based on the given context in Turkish."
        },
        {
            "role": "user", 
            "content": f"Main Question: {main_question}\nContext: {context}\n\nProvide a final answer to the question in Turkish. \
                         If the context doesn't provide enough information, say 'Soruyu cevaplamak için yeterli bilgi yok.'"
        }
    ])
    return final_answer 



if __name__ == "__main__":
    user_question = input("Enter your question: ")
    final_answer = react_agent(user_question)
    print("\nFinal Answer:")
    print(final_answer)
