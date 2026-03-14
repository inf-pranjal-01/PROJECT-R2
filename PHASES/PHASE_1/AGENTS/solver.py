import os
import json

# for saving me from relative path hell during running of scripts from different locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_SOLVER, API_URL


#read solver prompt
with open(os.path.join(BASE_DIR, "PROMPTS/solver.txt"), "r", encoding="utf-8") as f:
    SOLVER_PROMPT = f.read()                                  


import requests



def load_session_memory():
    solver_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_solver_response.json")))["response"]
    critic_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_critic_response.json")))["response"]
    judge_last  = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_judge_response.json")))["response"]

    return solver_last, critic_last, judge_last




def solver_run(question, user_preference):
    
    solver_last, critic_last, judge_last = load_session_memory()

    
    
    #prepare payload for next solver run
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"{SOLVER_PROMPT}\n\n Context: {user_preference}"},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": solver_last}, # model recognizes this as its own voice
            {"role": "user",      "content": f"""
                            Critic feedback: {critic_last}
                            Judge verdict:   {judge_last}
                            Now improve your answer.
                            """}



        ],
        "max_tokens": 800,
        "temperature": 0.7
    }


    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY_SOLVER}"},
        json=payload
    )
    
    result = response.json()

    content = result["choices"][0]["message"]["content"]
    usage   = result["usage"]

    prompt_tokens     = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    total_tokens      = result["usage"]["total_tokens"]
    
    json.dump(
        {"response": content},
        open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_solver_response.json"), "w"))
    

    
    return content, prompt_tokens, completion_tokens
