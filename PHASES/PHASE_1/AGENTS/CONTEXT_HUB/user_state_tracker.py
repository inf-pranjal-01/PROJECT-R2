import os
import json
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_USER_STATE_TRACKER, GROQ_API_URL, MODEL_GROQ_REASONING

# paths
LAST_5_PATH = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "MEMORY_DB", "Last_5.json")
USER_TRACKER_STATE_PATH = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "MEMORY_DB", "User_Tracker_State.json")
PROMPT_PATH = os.path.join(BASE_DIR, "..", "PROMPTS", "user_state_tracker.txt")

# defining primary keys and defaults for user state
primary_defaults = {
    "session_id",
    "last_updated_time",
    "last_updated_turn",
    "user_intent",
    "answer_style_needed",
    "depth_needed",
    "pacing",
    "flags",
    "suggestions_for_hub"
}

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    TRACKER_PROMPT = f.read()

def load_tracker_state():
    return json.load(open(USER_TRACKER_STATE_PATH))

def load_last_5():
    return json.load(open(LAST_5_PATH))   

def should_run(turn: int) -> bool:
    state = load_tracker_state()
    last_run = state.get("last_updated_turn", 0)
    return (turn - last_run) >= 3

def run_user_state_tracker(question: str, turn: int):
    
    # To early exit this function if it was run in the last 3 turns
    if not should_run(turn):
        return load_tracker_state()
    

    current_state = load_tracker_state()
    dynamic_keys_current = [k for k in current_state if k not in primary_defaults]

    # If this is third turn or more since the last run, we will run the tracker
    last_5 = load_last_5()
    messages = json.dumps(last_5["Last_5_messages"], indent=2)  # json.load converted json to python object.
                                                                # json.dumps (see that 's'?) python object to STRING
                                                                # json.dump writes it.

    payload = {
                    "model": MODEL_GROQ_REASONING,
                    "messages": [
                        {
                            "role": "system",
                            "content": TRACKER_PROMPT },
                        {
                            "role": "user",
                            "content": f"""
                                Current turn: {turn}
                                Current question: {question}

                                CURRENT STATE (update this, do not start from scratch):
                                {json.dumps(current_state, indent=2)}

                                Dynamic fields you may compress or remove: {dynamic_keys_current}

                                LAST 5 EXCHANGES:
                                {messages}

                                Return complete updated state JSON.
                                """
                                }],

                    "max_tokens": 600,
                    "temperature": 0.3
                }
    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {API_KEY_USER_STATE_TRACKER}"},
        json=payload
    )                                                                
    User_state_response = response.json()["choices"][0]["message"]["content"]

    try:
        tracker_data = json.loads(User_state_response)
    except:
        tracker_data = load_tracker_state()
        tracker_data["flags"] = ["tracker parse failed this turn"]

    now = datetime.now()
    tracker_data["last_updated_turn"] = turn
    tracker_data["session_id"] = now.strftime("%Y%m%d_%H%M%S")
    tracker_data["last_updated_time"] = now.isoformat()
    

    if len(tracker_data) > 15:
        
        dynamic_keys = [k for k in tracker_data if k not in primary_defaults]
        
        # remove by shortest value length — shortest = least informative
        dynamic_keys.sort(key=lambda k: len(str(tracker_data[k])))   #lambda used we dont require to give function a name.
        for key in dynamic_keys[:len(tracker_data) - 20]:
            del tracker_data[key] 

    with open(USER_TRACKER_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(tracker_data, f, indent=2)

       

    return tracker_data    
    