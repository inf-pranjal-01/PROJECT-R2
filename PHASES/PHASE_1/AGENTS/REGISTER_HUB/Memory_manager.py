import os
import json
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_MEMORY_MANAGER, GROQ_API_URL, MODEL_GROQ_REASONING

# paths
LAST_5_PATH           = os.path.join(BASE_DIR,"..","..","SESSION_MEMORY","MEMORY_DB","Last_5.json")
REGISTER_PATH         = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "MEMORY_DB", "register.json")
CONVERSATION_DB_PATH  = os.path.join(BASE_DIR,"..","..", "SESSION_MEMORY", "CONVERSATION_DB.json")
TRACKER_STATE_PATH    = os.path.join(BASE_DIR,"..", "..", "SESSION_MEMORY","MEMORY_DB","User_Tracker_State.json")
CONTEXT_PATH          = os.path.join(BASE_DIR,"..","..","SESSION_MEMORY","MEMORY_DB","context.json")
EXTRACT_PROMPT_PATH   = os.path.join(BASE_DIR,"..", "..", "PROMPTS", "memory_manager.txt")
DEEP_PROMPT_PATH      = os.path.join(BASE_DIR,"..","..", "PROMPTS", "memory_manager_DEEP.txt")


with open(EXTRACT_PROMPT_PATH,"r", encoding="utf-8") as f:
    EXTRACT_PROMPT =f.read()

with open(DEEP_PROMPT_PATH,"r",encoding="utf-8") as f:
    DEEP_PROMPT = f.read()

def load_register():
    return json.load(open(REGISTER_PATH))

def load_last_5():
    return json.load(open(LAST_5_PATH))

def load_tracker_state():
    return json.load(open(TRACKER_STATE_PATH))

def load_conversation_db():
    return json.load(open(CONVERSATION_DB_PATH))

def load_context():
    return json.load(open(CONTEXT_PATH))


def should_extract(turn:int) ->bool:
    register = load_register()
    last_run = register["meta"].get("last_extracted_turn", 0)
    return (turn-last_run) >=   3 



def should_deep_read(question: str, tracker_state: dict, hub_triggered: bool = False) -> bool:

    # user explicitly says it

    user_triggers = ["do a reset", "think from start", "start over",  "reset context", "forget everything", "start fresh"]
    
    
    user_said = any( t in question.lower() for t in user_triggers)

    # tracker suggests it
    tracker_flagged = "consider a reset" in tracker_state.get("flags", [])

    return user_said or tracker_flagged or hub_triggered


def run_extract(turn : int):
    last_5        = load_last_5()
    tracker_state = load_tracker_state()
    register      = load_register()

    payload = {
        "model": MODEL_GROQ_REASONING,
        "messages": [
            {
                "role": "system",
                "content": EXTRACT_PROMPT
            },
            {
                "role": "user",
                "content": f"""
    Current turn: {turn}

    CURRENT REGISTER:
    {json.dumps(register, indent=2)}

    USER TRACKER STATE:
    {json.dumps(tracker_state, indent=2)}

    LAST 5 EXCHANGES:
    {json.dumps(last_5["Last_5_messages"], indent=2)}

    Extract new facts and return updated register JSON.
                """
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.2
    }


    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {API_KEY_MEMORY_MANAGER}"},
        json=payload
    )
    

    result = response.json()

    content = result["choices"][0]["message"]["content"]

    # updating register
    try:
        extracted = json.loads(content)

        register["USER_FACTS"]         = extracted.get("USER_FACTS",register["USER_FACTS"])
        register["CONVERSATION_FACTS"] = extracted.get("CONVERSATION_FACTS", register["CONVERSATION_FACTS"])
        register["meta"]["last_extracted_turn"] = turn

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        register["meta"]["extract_failed_turn"] = turn
        register["meta"]["extract_failure_reason"] = str(e)


    with open(REGISTER_PATH, "w", encoding="utf-8") as f:
        json.dump(register, f, indent=2)

    return register


 



def run_deep_run(turn : int):

    # TODO: validate LLM output
    # TODO: handle malformed JSON
    # TODO: partial update vs replace decision

    conversation_db = load_conversation_db()
    tracker_state   = load_tracker_state()
    last_context    = load_context()
    register        = load_register()



    payload = {
        "model" : MODEL_GROQ_REASONING,
        "messages": [
            {
                "role" : "system",
                "content": DEEP_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
Current turn: {turn}

CURRENT REGISTER (will be rebuilt):
{json.dumps(register, indent=2)}

USER TRACKER STATE:
{json.dumps(tracker_state,indent=2)}

LAST ASSEMBLED CONTEXT:
{json.dumps(last_context,indent=2)}

FULL CONVERSATION HISTORY:
{json.dumps(conversation_db["pairs"],indent=2)}

Perform deep read and return complete rebuilt register JSON.""" 

            } ],

        "max_tokens": 2000,
        "temperature": 0.2,
    }



    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {API_KEY_MEMORY_MANAGER}"},
        json=payload
    )

    result = response.json()

    content = result["choices"][0]["message"]["content"]
    
    # recreating from scratch, so we replace instead of partial update.
    try:
        deep_data = json.loads(content)



        if isinstance(deep_data,dict) and "register" in deep_data:
            deep_data = deep_data["register"]

        if not isinstance(deep_data, dict):
            raise ValueError("deep_data is not a dict")
        
        if "USER_FACTS" not in deep_data or "CONVERSATION_FACTS" not in deep_data:
            raise ValueError("missing required keys")
    

        register["USER_FACTS"]     = deep_data.get("USER_FACTS",register["USER_FACTS"])
        register["CONVERSATION_FACTS"] = deep_data.get("CONVERSATION_FACTS",register["CONVERSATION_FACTS"])
        register["meta"]["last_deep_turn"] =turn
        register["meta"]["deep_status"] = "success"

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        register["meta"]["deep_failed_turn"] = turn
        register["meta"]["deep_failure_reason"] = str(e)

    with open(REGISTER_PATH, "w", encoding="utf-8") as f:
        json.dump(register, f, indent = 2)


    return register


def run_memory_manager(question : str, turn:int,tracker_state: dict, user_said : bool = False , hub_triggered: bool = False):

    # deep read takes priority over regular extract
    if should_deep_read(question, tracker_state, hub_triggered):
        return run_deep_run(turn)

    elif should_extract(turn):
        return run_extract(turn)

    else:
        return load_register()






