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
EXTRACT_PROMPT_PATH   = os.path.join(BASE_DIR,"..","PROMPTS", "memory_manager_extract.txt")
DEEP_PROMPT_PATH      = os.path.join(BASE_DIR,"..","PROMPTS", "memory_manager_DEEP.txt")

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