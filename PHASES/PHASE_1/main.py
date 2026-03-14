import os
from dotenv import load_dotenv
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings




# EMBEDDING MODEL
embedding_model = OpenAIEmbeddings(
    api_key=API_KEY,
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




#CLI Input and Preference Retrieval
question = input()
user_preference = retrieve_context(question)


for i in range(3):
   solver_run()
   critic_run()
   judge_run()



#load latest solver response
print("Final Answer: ", final_answer)   
