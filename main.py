from flask import Flask, render_template, request, jsonify
import os
import requests
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter

app = Flask(__name__)

PERSIST = True
vectorstore = None
GROQ_API_KEY = "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"

def initialize_vectorstore():
    global vectorstore
    
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=HuggingFaceEmbeddings())
    else:
        loader = TextLoader("data.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, HuggingFaceEmbeddings(), persist_directory="persist")

def query_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are an enthusiastic and friendly assistant.Your Name is Sense Car. Provide humanized answers based on the given context. Do not mention any price details, instead suggest that price information will be shared by respective dealers. Your responses should be energetic, positive, and very natural-sounding."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    if 'choices' not in response_json:
        print("Unexpected response format:", response_json)
        return "Sorry, I couldn't process your request at this time."
    
    return response_json['choices'][0]['message']['content']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    message = request.json['message']
    chat_history = request.json.get('chat_history', [])
    relevant_docs = vectorstore.similarity_search(message, k=2)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""Context: {context}
    Question: {message}
    Answer:"""
    answer = query_groq(prompt)
    chat_history.append((message, answer))
    return jsonify({'answer': answer, 'chat_history': chat_history})

if __name__ == '__main__':
    initialize_vectorstore()
    app.run(debug=True)