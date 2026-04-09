import os
import json
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from utils.llm_call import API_KEY_HUB,GROQ_API_URL, MODEL_GROQ_REASONING

# paths
LAST_5_PATH = os.path.join(BASE_DIR, ".." , "..", "SESSION_MEMMORY", "MEMORY_DB", "last_5.json")
REGISTER_PATH = os.path.join(BASE_DIR, ".." , "..", "SESSION_MEMMORY", "MEMORY_DB", "register.json")
TRACKER_PATH = os.path.join(BASE_DIR, ".." , "..", "SESSION_MEMMORY", "MEMORY_DB", "user_tracker_state.json")
CONTEXT_PATH = os.path.join(BASE_DIR, ".." , "..", "SESSION_MEMMORY", "MEMORY_DB", "context.json")
JUDGE_RESPONSE_PATH = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "latest_judge_response.json")
CRITIC_RESPONSE_PATH = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "latest_critic_response.json")
SOLVER_RESPONSE_PATH = os.path.join(BASE_DIR, "..", "..", "SESSION_MEMORY", "latest_solver_response.json")
PROMPT_PATH = os.path.join(BASE_DIR, ".." , "..", "PROMPTS", "hub.txt")

with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    HUB_PROMPT = f.read()


def load_last_5():
    return json.load(open(LAST_5_PATH))

def load_register():
    return json.load(open(REGISTER_PATH))

def load_tracker_state():
    return json.load(open(TRACKER_PATH))


def load_last_judge():
    try:
        raw = open(JUDGE_RESPONSE_PATH).read()
        return json.loads(raw)
    except:
        return {"SCORE": None, "VERDICT": None, "REASON": None}
    
def load_last_critic():
    try:
        data = json.load(open(CRITIC_RESPONSE_PATH))
        return data.get("response", "none")
    except:
        return "none"


def load_context():
    return json.load(open(CONTEXT_PATH))



def run_hub(question: str, turn: int, hub_triggered_reset: bool = False):



    last_5 = load_last_5()
    register = load_register()
    tracker_state = load_tracker_state()
    last_judge = load_last_judge()
    last_critic = load_last_critic()
    context = load_context()

    payload = {
        "model": MODEL_GROQ_REASONING,
        "messages": [
            {
            "role": "system",
            "content": HUB_PROMPT
            },

            {
                "role" : "user",
                "content": f"""
Current Turn: {turn}
Current Question: {question}

LAST 5 EXCHANGES (always include verbatim in last_5_raw):
{json.dumps(last_5["Last_5_messages"], indent=2)}

FULL REGISTER (extract only what is relevant):
{json.dumps(register, indent=2)}

USER TRACKER STATE:
{json.dumps(tracker_state, indent=2)}

LAST ASSEMBLED CONTEXT (for continuity reference):
{json.dumps(context, indent=2)}
 
LAST JUDGE SCORE:   {last_judge.get("SCORE","none")}
LAST JUDGE VERDICT: {last_judge.get("VERDICT","none")}
LAST JUDGE REASON:  {last_judge.get("REASON","none")}

LAST CRITIC FEEDBACK:
{last_critic}

Assemble the perfect context payload for solver. Return valid JSON only
"""

            }],

            "max_tokens": 2000,
            "temperature": 0.2,
            "provider": {"zdr": True}
        }
    
    response = requests.post(
        url=GROQ_API_URL,
        headers={"Authorization": f"Bearer {API_KEY_HUB}"},
        json=payload
    )

    result = response.json()
    content = result["choices"][0]["message"]["content"]


    required_keys = [
            "last_5_raw",
            "relevant_from_register",
            "non_negotiable_facts",
            "user_state_snapshot",
            "instructions_for_solver"
 ]
    

    try:
        hub_data = json.loads(content)

        #  all required sections
        for key in required_keys:
            if key not in hub_data:
                raise ValueError(f"missing section: {key}")


        hub_data["last_5_raw"] = last_5["Last_5_messages"]

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        hub_data = {
            "last_5_raw":             last_5["Last_5_messages"],
            "relevant_from_register": register.get("USER_FACTS", {}),
            "non_negotiable_facts":   register.get("CONVERSATION_FACTS", {}),
            "user_state_snapshot": {
                "answer_style_needed": tracker_state.get("answer_style_needed", "none"),
                "depth_needed":        tracker_state.get("depth_needed", "none"),
                "pacing":              tracker_state.get("pacing", "none"),
                "active_flags":        tracker_state.get("flags", []),
                "style_instructions":  "hub failed — use best judgment"
            },
            "instructions_for_solver": "Hub assembly failed. Answer based on last 5 exchanges.",
            "suggest_reset_to_user":   False,
            "reset_message":           "",
            "autonomous_reset":        False,
            "hub_assembly_failed":     True,
            "hub_failure_reason":      str(e)
        }


    # add metadata
    hub_data["assembled_at_turn"]    = turn
    hub_data["timestamp"]            = datetime.now().isoformat()
    hub_data["hub_reset_triggered"]  = hub_triggered_reset

   
    # save context.json
    with open(CONTEXT_PATH, "w", encoding="utf-8") as f:
        json.dump(hub_data, f, indent=2)

    # read reset signals from hub output and then add instructions more explicitly
    suggest_reset    = hub_data.get("suggest_reset_to_user", False)
    autonomous_reset = hub_data.get("autonomous_reset", False)
    reset_message    = hub_data.get("reset_message", "")


    return hub_data, suggest_reset, autonomous_reset, reset_message



def build_solver_context_string(hub_data: dict) -> str:
    # parts is a container used to hold fields
    parts = []

    # primary fields
    PRIMARY_SECTIONS = [
        "instructions_for_solver",
        "non_negotiable_facts",
        "relevant_from_register",
        "user_state_snapshot",
        "additional_details",
        "last_5_raw"
    ]

    # fields that are metadata
    META_FIELDS = {
        "assembled_at_turn",
        "timestamp",
        "hub_reset_triggered",
        "hub_assembly_failed",
        "hub_failure_reason",
        "suggest_reset_to_user",
        "reset_message",
        "autonomous_reset"
    }

    # render primary sections first
    for key in PRIMARY_SECTIONS:
        if key not in hub_data:
            continue

        value = hub_data[key]

        if key == "instructions_for_solver":
            parts.append(f"INSTRUCTION:\n{value}")

        elif key == "non_negotiable_facts" and value:
            parts.append(f"NON-NEGOTIABLE CONTEXT:\n{json.dumps(value, indent=2)}")

        elif key == "relevant_from_register" and value:
            parts.append(f"RELEVANT FACTS:\n{json.dumps(value, indent=2)}")

        elif key == "user_state_snapshot" and value: 
            snap = value



            # collect dynamic fields from snapshot
            SNAPSHOT_PRIMARY = {"answer_style_needed", "depth_needed","pacing","active_flags","style_instructions"}
            dynamic_snap = {k: v for k,v in snap.items() if k not in SNAPSHOT_PRIMARY}

            # build primary style line
            style_line = " | ".join([
                f"Style: {snap.get('answer_style_needed', 'none')}",
                f"Depth:     {snap.get('depth_needed', 'none')}",
                f"Pacing: {snap.get('pacing', 'none')}"
            ])
            #single style line created
            section = f"RESPONSE STYLE:\n{style_line}"

            # more fields added via section after style line
            
            if snap.get("active_flags"):
                section += f"\nFlags: {snap['active_flags']}"

            if snap.get("style_instructions"):
                section += f"\n{snap['style_instructions']}"
            
            # dynamic fields from snapshot
            if dynamic_snap:
                section += f"\nMore context about user state:\n{json.dumps(dynamic_snap,indent=2)}"
            parts.append(section)

        elif key == "additional_details" and value:
            parts.append(f"ADDITIONAL DETAILS:\n{json.dumps(value,indent=2)}")

        elif key == "last_5_raw" and value:
            parts.append(f"LAST 5 RAW EXCHANGES:\n{json.dumps(value, indent=2)}")

    # dynamic fields hub created
    dynamic_keys = [
        k for k in hub_data
        if k not in PRIMARY_SECTIONS and k not in META_FIELDS
    ]

    if dynamic_keys:
        dynamic_section = {}
        for k in dynamic_keys:
            dynamic_section[k] = hub_data[k]
        parts.append(f"ADDITIONAL CONTEXT:\n{json.dumps(dynamic_section, indent=2)}")

    return "\n---\n".join(parts)