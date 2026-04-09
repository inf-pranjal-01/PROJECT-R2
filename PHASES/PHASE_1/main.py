import json
import os
from dotenv import load_dotenv

#unused imports for now, will be used in later phases
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from datetime import datetime

from AGENTS.solver import solver_run
from AGENTS.critic import critic_run
from AGENTS.judge  import judge_run
from AGENTS.REGISTER_HUB.Memory_manager import run_memory_manager
from AGENTS.CONTEXT_HUB.user_state_tracker import run_user_state_tracker
from AGENTS.CONTEXT_HUB.hub import run_hub, build_solver_context_string


load_dotenv()
from PHASES.PHASE_1.AGENTS.CONTEXT_HUB.user_state_tracker import run_user_state_tracker
from utils.llm_call import API_KEY_JUDGE

# EMBEDDING MODEL
embedding_model = OpenAIEmbeddings(
    api_key=API_KEY_JUDGE,
    base_url="https://openrouter.ai/api/v1",
    model="text-embedding-3-small"
)


# LOAD PERSONAL SEED FILE
with open("User_Preference.txt", "r", encoding="utf-8") as f:
    User_Preference_text = f.read()


# TEXT SPLITTING FUNCTION (adjust chunk_size and chunk_overlap later more optimally)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


documents = text_splitter.create_documents([User_Preference_text])

# VECTOR STORE
vector_store = FAISS.from_documents(documents, embedding_model)

# RETRIEVAL FUNCTION
def retrieve_context(query: str, k: int = 3):
    results = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])

#load session memory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# paths
MEMORY_DB_PATH       = os.path.join(BASE_DIR, "SESSION_MEMORY", "MEMORY_DB")
LAST_5_PATH          = os.path.join(MEMORY_DB_PATH, "Last_5.json")
CONVERSATION_DB_PATH = os.path.join(BASE_DIR, "SESSION_MEMORY", "CONVERSATION_DB.json")



# RESET SESSION MEMORY AT START OF EACH RUN
def reset_session_memory():
    empty = {"response": "none"}
    json.dump(empty, open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_solver_response.json"), "w"))
    json.dump(empty, open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_critic_response.json"), "w"))
    with open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_judge_response.json"), "w") as f:
        f.write("none")  # raw string for judge





#CLI Input and Preference Retrieval
question = input()
user_preference = retrieve_context(question)




def load_session_memory():
        solver_last = json.load(open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_solver_response.json")))["response"]
        critic_last = json.load(open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_critic_response.json")))["response"]
        judge_last  = open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_judge_response.json")).read()    
        return solver_last, critic_last, judge_last




def load_judge_output():
    try:
        raw = open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_judge_response.json")).read()
        return json.loads(raw)
    except:
        return {"SCORE": 0, "VERDICT": "retry", "REASON": None}
    


def update_last_5(question: str, final_answer: str, score: int, turn: int):
    try:
        data = json.load(open(LAST_5_PATH))
    except:
        data = {"Last_5_messages": []}
    
    entry = {
        "turn" : turn,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "final_answer": final_answer,
        "score": score
    }

    data["Last_5_messages"].append(entry)

    if len(data["Last_5_messages"]) > 5:
        data["Last_5_messages"] = data["Last_5_messages"][-5:]

    with open(LAST_5_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



def update_conversation_db(question: str, final_answer: str, turn: int):
    try:
        data = json.load(open(CONVERSATION_DB_PATH))
    except:
        data = {"total_pairs ": 0, "pairs": []}
    
    entry = {
        "turn" : turn,
        "timestamp": datetime.now().isoformat(),
        "question": question, 
        "final_answer": final_answer

    } 

    data["pairs"].append(entry)
    data["total_pairs "] = len(data["pairs"])

    if len(data["pairs"]) > 50:
        data["pairs"] = data["pairs"][-50:]
        data["total_pairs "] = len(data["pairs"])


    with open(CONVERSATION_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


reset_session_memory()


for i in range(6):

   turn = 0


   while True:
        question = input("You: \n ").strip()
        if not question:
            continue
        turn+=1


        # check for reset trigger
        reset_triggers = ["do a reset", "think from start", "start over",  "reset context", "forget everything", "start fresh" ]
    
        user_triggered_reset = any(trigger in question.lower() for trigger in reset_triggers)

        tracker_state = json.load(open(os.path.join(MEMORY_DB_PATH, "user_tracker_state.json")))

        if user_triggered_reset:
                print("Rebuilding my understanding.....Please think of your ex till then.")
                run_memory_manager(question, turn, tracker_state, hub_triggered=False)

                print("Context Rebuild, please ask the question again.")
                continue

        run_user_state_tracker(question, turn)
        run_memory_manager(question, turn, tracker_state, hub_triggered=False)
        hub_data, suggest_reset, autonomous_reset, reset_message = run_hub(question, turn)




        if autonomous_reset:
            print(f"Critical context loss detected : Rebuilding my understanding.....Please think of your ex till then.")
            run_memory_manager(question, turn, tracker_state, hub_triggered=True)
            hub_data, _, _, _ = run_hub(question, turn, hub_triggered_reset=True)
            print("Running agents now..........")

        elif suggest_reset:
            print(f"\n{reset_message}")
            

        solver_context = build_solver_context_string(hub_data)    
        
        final_answer = "none"

        #call agents (real mess)
        solver_run(question, solver_context)
        critic_run(question, solver_context)
        judge_run(question, solver_context)
        
        _, _, judge_last = load_session_memory()
        try:
            judge_data = json.loads(judge_last)
            score      = judge_data["SCORE"]
            verdict    = judge_data["VERDICT"]
            reason     = judge_data["REASON"]
            
        except:
            score   =  0
            verdict = "Malfunction in Judge, no response."
            reason  = "Malfunction in Judge, no response."

        print(f"Round {i+1} | Score: {score} | Verdict: {verdict} | Reason: {reason}")

        if score >= 90 or verdict == "PASS":
                break
        




        #load latest solver response
        try:
            solver_data  = json.load(open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_solver_response.json")))
            final_answer = solver_data["response"]
        except:
            final_answer = "Solver failed to respond."

        print(f"Final Answer: {final_answer}")
        


        update_last_5(question, final_answer, score, turn)
        update_conversation_db(question, final_answer, turn)