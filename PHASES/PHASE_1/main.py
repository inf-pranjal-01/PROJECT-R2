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

from AGENTS.solver import solver_run
from AGENTS.critic import critic_run
from AGENTS.judge  import judge_run

load_dotenv()
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

# RESET SESSION MEMORY AT START OF EACH RUN
def reset_session_memory():
    empty = {"response": "none"}
    json.dump(empty, open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_solver_response.json"), "w"))
    json.dump(empty, open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_critic_response.json"), "w"))
    with open(os.path.join(BASE_DIR, "SESSION_MEMORY", "latest_judge_response.json"), "w") as f:
        f.write("none")  # raw string for judge


reset_session_memory()


#CLI Input and Preference Retrieval
question = input()
user_preference = retrieve_context(question)




def load_session_memory():
        solver_last = json.load(open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_solver_response.json")))["response"]
        critic_last = json.load(open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_critic_response.json")))["response"]
        judge_last  = open(os.path.join(BASE_DIR, "SESSION_MEMORY/latest_judge_response.json")).read()    
        return solver_last, critic_last, judge_last

for i in range(3):

   
   
   
   #call agents (real mess)
   solver_run(question, user_preference)
   critic_run(question, user_preference)
   judge_run(question, user_preference)
   
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
solver_data = json.load(open("SESSION_MEMORY/latest_solver_response.json"))
print(f"Final Answer: {solver_data['response']}")
