from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import ServiceContext
# from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import Settings
from flask import Flask,jsonify
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import huggingface_hub
import posixpath
import torch
from flask import render_template, request,jsonify
import requests
from llama_index.core import PromptTemplate
import base64
import json
import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.llms import ChatMessage
import markdown
import re
import bleach
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# Log in using your access token
huggingface_hub.login(token="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")


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads_new'
# INDEX_FOLDER = 'index'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chatbot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define a model for storing questions and responses
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(1000))
    response = db.Column(db.String(1000))

# Create the database and tables
with app.app_context():
    db.create_all()

def store_in_database(question, response):
    chat_entry = ChatHistory(question=question, response=response)
    db.session.add(chat_entry)
    db.session.commit()


system_prompt = """
    As a representative of Advoice Law Consultancy, tasked with providing insights on legal matters, I'm equipped to analyze questions, identify key terms, and retrieve pertinent information from the law domain.
    The model comprehends the question's context, extracts relevant keywords, phrases, and sentences, and retrieves answers based on this analysis.
    It's capable of handling variations in question phrasing and generating intelligent responses akin to a legal consultancy bot.
    If the question is within the legal domain, the model will provide answers. If not, it will respectfully state that it cannot answer outside the law domain.
    When addressing legal matters, it's important to consult with a qualified legal professional for specific guidance and representation.
    Answer every question succinctly in up to three lines.
    Do not include that strictly that "I cannot provide legal Advice" in any answer instead of this include "According to Legal Consultancy".
    If greeted, respond accordingly, and if bid farewell, respond with "goodbye".
Context:\n {context}?\n
Question: \n{question}\n
"""
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

llm2 = HuggingFaceLLM(
    context_window=8192,
    max_new_tokens=1000,
    generate_kwargs={"temperature": 0.3, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=SimpleInputPrompt("{query_str}"),
    tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    device_map="auto",
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="nomic-ai/nomic-embed-text-v1", trust_remote_code=True)


# Set settings
Settings.llm = llm2
Settings.embed_model = embed_model


# Format document
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

documents1 = SimpleDirectoryReader('uploads').load_data()
index = VectorStoreIndex.from_documents(documents1)

memory = ChatMemoryBuffer.from_defaults(token_limit=20000)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=system_prompt,
    llm=llm2,
    verbose=False
)

def gen_response(question):
    if question=="hi" or question=="Hi" or question =="hi!" or question =="Hi!" or question =="Hello" or question =="hello" or question =="hey" or question =="Hey":
        return "Hello! How can I assist you today?"
        
    response = chat_engine.chat(question)
   
    return str(response)

def clean_response(response):
    pattern = r'^\s*assistant\s*'

    # Use re.sub to replace the pattern with an empty string
    cleaned_response = re.sub(pattern, '', response)
    return cleaned_response



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/pdf')
def files():
    return render_template('file.html')

@app.route('/try consulting')
def consulting():
    return render_template('chat.html')


@app.route('/ask_question', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        input_text = request.form['question']
        response1 = gen_response(input_text)
        cleaned_response = clean_response(response1)
        store_in_database(input_text, cleaned_response)
        # Process response to HTML
        html_response = markdown.markdown(cleaned_response)
        print(html_response)
        return jsonify({'response': html_response})
    else:
        return jsonify({'response': 'Unsupported language detected.'})

@app.route('/fetch_conversations', methods=['GET'])
def fetch_conversations():
    conversations = Conversation.query.all()
    data = [{'id': conv.id, 'question': conv.question, 'response': conv.response} for conv in conversations]
    return jsonify(data)

@app.route('/reset_chat_engine', methods=['POST'])
def reset_chat_engine():
    chat_engine_reset()
    return jsonify({"reply": "History Dumped Successfully"})

def chat_engine_reset():
    chat_engine.reset()
    return "History Dumped Successfully"


@app.route('/convert', methods=['POST'])
def convert_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part"})
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})

    if file:

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in {'.txt', '.pdf'}:
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return jsonify({"success": True, "message": "File uploaded successfully"})
        else:
            return jsonify({"success": False, "message": "Unsupported file format"})



@app.route('/ask pdf', methods=['POST'])
def ask_pdf():
    if request.method == 'POST':
        input_text = request.form['question']
        documents = SimpleDirectoryReader('uploads_new').load_data()
        Settings.llm = llm2
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        print("generating response----------------------")
        response = query_engine.query(input_text) 
        print(response)
        return str(response)
 
    else:
        return {'result': 'Unsupported language detected.'}

    
if __name__ == '__main__':
    app.run(debug=False,port=5001)



