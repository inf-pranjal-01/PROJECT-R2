import os
import json

# for saving me from relative path hell during running of scripts from different locations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_CRITIC, API_URL


#read critic prompt
with open(os.path.join(BASE_DIR, "../PROMPTS/critic.txt"), "r", encoding="utf-8") as f:
    CRITIC_PROMPT = f.read()                                  


import requests



def load_session_memory():
    solver_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_solver_response.json")))["response"]
    critic_last = json.load(open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_critic_response.json")))["response"]
    judge_last  = open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_judge_response.json")).read()

    return solver_last, critic_last, judge_last



def critic_run(question, user_preference):
    
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
    
    
    #prepare payload for next critic run
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": f"{CRITIC_PROMPT}\n\n Context: {user_preference}"},
            {"role": "user",      "content": question},
            
            {"role": "user",      "content": f"""
                            Solver response: {solver_last}
                            Last Judge verdict (PASS/RETRY): {verdict}
                            Last Judge score:     {score}
                            Reason for last Judge verdict : {reason}
                            Now FIND FLAWS, TRY YOU BEST TO BREAK IT.
                            """}



        ],
        "max_tokens": 400,
        "temperature": 0.4
    }


    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY_CRITIC}"},
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
        open(os.path.join(BASE_DIR, "../SESSION_MEMORY/latest_critic_response.json"), "w"))
    

    
    return content, prompt_tokens, completion_tokens
