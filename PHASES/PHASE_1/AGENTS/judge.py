import os
import json

# for saving me from relative path hell during running of scripts from different locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_JUDGE, API_URL


#read judge prompt
with open(os.path.join(BASE_DIR, "../PROMPTS/judge.txt"), "r", encoding="utf-8") as f:
    JUDGE_PROMPT = f.read()                                  


import requests



def load_session_memory():
    solver_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_solver_response.json")))["response"]
    critic_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_critic_response.json")))["response"]
    judge_last  = open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_judge_response.json")).read()

    return solver_last, critic_last, judge_last




def judge_run(question, user_preference):
    
    solver_last, critic_last, judge_last = load_session_memory()

    try:
        judge_data = json.loads(judge_last)
        score      = judge_data["SCORE"]
        verdict    = judge_data["VERDICT"]
        reason     = judge_data["REASON"]
    
    except:
     score   =  0
     verdict = "NOT APPLICABLE, THIS IS FIRST PASS."
     reason  = "NOT APPLICABLE, THIS IS FIRST PASS."
    
    
    #prepare payload for next judge run
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"{JUDGE_PROMPT}\n\n Context: {user_preference}"},
            {"role": "user",      "content": question},
            
            {"role": "user",      "content": f"""
                            Solver response: {solver_last}
                            Critic feedback: {critic_last}
                            Last verdict (PASS/RETRY): {verdict}
                            Last score:     {score}
                            Reason for last verdict : {reason}
                            Now PASS VERDICT ON SOLVER RESPONSE AND SCORE IT.
                            """}



        ],
        "max_tokens": 150,
        "temperature": 0.2
    }


    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY_JUDGE}"},
        json=payload
    )
    
    result = response.json()

    content = result["choices"][0]["message"]["content"]
    usage   = result["usage"]

    prompt_tokens     = result["usage"]["prompt_tokens"]
    completion_tokens = result["usage"]["completion_tokens"]
    total_tokens      = result["usage"]["total_tokens"]
    
    with open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_judge_response.json"), "w") as f:
     f.write(content)

    
    return content, prompt_tokens, completion_tokens
