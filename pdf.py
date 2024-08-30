# from flask import Flask, render_template, request, jsonify
# import os
# import json
# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import StorageContext
# from llama_index.core import VectorStoreIndex
# from werkzeug.utils import secure_filename
# from llama_index.core import StorageContext, load_index_from_storage
# from llama_index.core import Settings
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.gemini import Gemini
# from flask import Flask, render_template, request, jsonify
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.schema.runnable import RunnablePassthrough
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
# from langchain_community.chat_models import ChatOllama
# from langchain.schema import StrOutputParser
# from googletrans import Translator
# from llama_index.core import Prompt

# from llama_index.core import ServiceContext
# from llama_index.embeddings.langchain import LangchainEmbedding
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core.prompts.prompts import SimpleInputPrompt
# from llama_index.core import Settings
# from flask import Flask
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# import huggingface_hub
# import torch
# from flask import render_template, request


# # Log in using your access token
# huggingface_hub.login(token="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")

# app = Flask(__name__)



# # Load data from JSON file
# with open("FINAL_DATA.json", "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)

# # Convert data to string and write to a markdown file
# with open("indexFINAL.md", "w", encoding="utf-8") as file:
#     file.write(str(data))

# # Load documents using text loader
# loader = TextLoader("indexFINAL.md", encoding="utf-8")
# data = loader.load()

# UPLOAD_FOLDER = 'uploads'
# INDEX_FOLDER = 'index'
# ALLOWED_EXTENSIONS = {'pdf'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # Split documents into smaller chunks
# def split_docs(documents, chunk_size=1000, chunk_overlap=200):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     docs_new = text_splitter.split_documents(documents)
#     return docs_new

# docs_new = split_docs(data)

# os.environ['GOOGLE_API_KEY'] = "AIzaSyCB0FsriiPfyTLwZGM9z_cDLdl03MFjeFQ"

# system_prompt="""
#     "I have provided context information below.\n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#      You are a helpful and informative bot that answers questions using text from the reference passage included below. \
#      Answer every questions in max 3 lines or in bullet points as required. \
#      Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
#      However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
#      strike a friendly and converstional tone. \
#      If the passage is out of the context from the documents and also out of the law domain, do not answer it. \
#      Try not to include phrases like Based on the context provided or In the context provided instead use according to my knowledge or As per Advoice Consultancy  or as far as I know give answer in a  more genrative and smart manner like a bot AI agent does. \
#      If the passage is out of the context from the documents say that sorry but i am not allowed to answer outside law domain in a respectful manner.{query_str}\n\n.
# """
 
# ## Default format supportable by LLama2


# llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=256,
#     generate_kwargs={"temperature": 0.7, "do_sample": False},
#     system_prompt=system_prompt,

#     # tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
#     # model_name="meta-llama/Llama-2-7b-chat-hf",
#     model_name="mistralai/Mistral-7B-Instruct-v0.2",
#     tokenizer_name ="mistralai/Mistral-7B-Instruct-v0.2",
#     device_map="auto",
#     tokenizer_kwargs={"max_length": 4096},
#     # uncomment this if using CUDA to reduce memory usage
#     # model_kwargs={"torch_dtype": torch.float16}
# )

# embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
# # local_model = "mistral"
# # llm = ChatOllama(model=local_model)
# #llm = Gemini(api_key="AIzaSyCB0FsriiPfyTLwZGM9z_cDLdl03MFjeFQ", model_name="models/gemini-pro",temperature=0.7, top_p=0.9, top_k=50)

# Settings.llm=llm
# model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.9, top_k=50)

# gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # ollama_embedding = OllamaEmbedding(
# #     model_name="nomic-embed-text",
# #     base_url="http://localhost:11434",
# #     ollama_additional_kwargs={"mirostat": 0},
# # )
# Settings.embed_model = embed_model
# # Create vector store
# vectorstore = Chroma.from_documents(
#     documents=docs_new,
#     embedding=gemini_embeddings,
#     persist_directory="./chroma_db"
# )

# vectorstore_disk = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=gemini_embeddings
# )
# retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 3})

# def translate_hinglish_to_english(text):
#     translator = Translator()
#     detected_lang = translator.detect(text).lang
#     translated_text = translator.translate(text, src='hi', dest='en').text
#     return translated_text

# # Format document
# def format_docs(docs):
#     return "\n".join(doc.page_content for doc in docs)


# llm_prompt_template = """Use the following piece of context to answer the Question in paragraph along with proper grammar
# Use five sentences maximum and keep the answer concise and more human like.\n
# Question: {question} \nContext: {context} \nAnswer:"""

# llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | llm_prompt
#     | model
#     | StrOutputParser()
# )

# # Detect language
# def detect_language(text):
#     if any(ord(char) > 127 for char in text):
#         return "hinglish"
#     else:
#         return "english"

# # Process input
# def process_input(input_text):
#     lang = detect_language(input_text)
#     if lang == "english":
#         return "english", input_text
#     elif lang == "hinglish":
#         return "hinglish", translate_hinglish_to_english(input_text)
#     else:
#         print("Unsupported language detected.")
#         return None


# def format_to_markdown(text):
#   lines = text.strip().split('\n')
#   formatted_text = ""
#   for line in lines:
#     formatted_text += f"- {line.replace('*', '')}\n"
#   return formatted_text

# # template = (
# #     "I have provided context information below.\n"
# #     "---------------------\n"
# #     "{context_str}"
# #     "\n---------------------\n"
# #     "You are a helpful and informative bot that answers questions using text from the reference passage included below. \
# #      Answer every questions in max 3 lines or in bullet points as required. \
# #      Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
# #      However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
# #      strike a friendly and converstional tone. \
# #      If the passage is out of the context from the documents and also out of the law domain, do not answer it. \
# #      Try not to include phrases like Based on the context provided or In the context provided instead use according to my knowledge or As per Advoice Consultancy  or as far as I know give answer in a  more genrative and smart manner like a bot AI agent does. \
# #      If the passage is out of the context from the documents say that sorry but i am not allowed to answer outside law domain in a respectful manner. {query_str}\n"
# # )
# # qa_template = Prompt(template)

# # def create_and_save_index(file_path):
# #     # index_dir = 'index'
# #     # os.makedirs(index_dir, exist_ok=True)
    
# #     # Load documents
# #     documents = SimpleDirectoryReader('uploads').load_data()

# #     # Configure settings
# #     Settings.llm = llm
# #     Settings.embed_model = ollama_embedding

# #     # Create and save index
# #     index = VectorStoreIndex.from_documents(documents)
# #     # index.storage_context.persist(persist_dir=index_dir)

# # def load_saved_index():
# #     index_dir = 'index'
# #     storage_context = StorageContext.from_defaults(persist_dir=index_dir)
# #     return load_index_from_storage(storage_context)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/service')
# def service():
#     return render_template('service.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')
# @app.route('/pdf')
# def files():
#     return render_template('file.html')

# @app.route('/try consulting')
# def consulting():
#     return render_template('chat.html')


# @app.route('/ask question', methods=['POST'])
# def ask_question():
#     if request.method == 'POST':
#         input_text = request.form['question']
#         input_lang, processed_text = process_input(input_text)
#         if processed_text:
#             output = rag_chain.invoke(processed_text)
#             formatted_output = format_to_markdown(output)
#             return jsonify({'response': formatted_output})
#         else:
#             return jsonify({'response': 'Unsupported language detected.'})


# # @app.route('/convert', methods=['POST'])

# # def convert_file():
# #     if 'file' not in request.files:
# #         return jsonify({"success": False, "message": "No file part"})
    
# #     file = request.files['file']

# #     if file.filename == '':
# #         return jsonify({"success": False, "message": "No selected file"})

# #     if file:
# #         os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# #         os.makedirs(INDEX_FOLDER, exist_ok=True) 

# #         # Save the file with a secure filename
# #         filename = secure_filename(file.filename)
# #         file_ext = os.path.splitext(filename)[1].lower()

# #         file_path = os.path.join(UPLOAD_FOLDER, filename)

# #         if file_ext in {'.txt', '.pdf'}:
# #             # Check if file already exists in uploads folder
# #             if os.path.exists(file_path):
# #                 # Load index from existing documents
# #                 Settings.llm = llm
# #                 Settings.embed_model = ollama_embedding
# #                 index = load_saved_index()
# #                 return jsonify({"success": True, "message": "Index loaded successfully"})

# #             # Delete existing index
# #             for file_name in os.listdir(INDEX_FOLDER):
# #                 file_path = os.path.join(INDEX_FOLDER, file_name)
# #                 if os.path.isfile(file_path):
# #                     os.remove(file_path)

# #             # Save file to uploads folder
# #             file.save(file_path)
# #             # Create and save index
# #             create_and_save_index(file_path)
# #             return jsonify({"success": True, "message": "File uploaded successfully and new index created"})

# #         else:
# #             return jsonify({"success": False, "message": "Unsupported file format"})

# @app.route('/convert', methods=['POST'])
# def convert_file():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "message": "No file part"})
    
#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"success": False, "message": "No selected file"})

#     if file:
#         # upload_dir = 'uploads'  # Specify the directory where you want to save uploaded files
#         # os.makedirs(upload_dir, exist_ok=True)
#         os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#         os.makedirs(INDEX_FOLDER, exist_ok=True) 

#         # Save the file with a secure filename
#         filename = secure_filename(file.filename)
#         file_ext = os.path.splitext(filename)[1].lower()

#         if file_ext in {'.txt', '.pdf'}:
#             file.save(os.path.join(UPLOAD_FOLDER, filename))
#             return jsonify({"success": True, "message": "File uploaded successfully"})
#         else:
#             return jsonify({"success": False, "message": "Unsupported file format"})


# # @app.route('/ask pdf', methods=['POST'])
# # def ask_pdf():
# #     if request.method == 'POST':
# #         input_text = request.form['question']
# #         if not os.path.exists('index'):
# #             create_and_save_index()
# #         else:
# #             print("Index directory exists. Loading index...")


# #         index1 = load_saved_index()
# #         Settings.llm = llm
# #         Settings.embed_model = ollama_embedding
# #         query_engine = index1.as_chat_engine(text_qa_template=qa_template)
        
# #         print("generating response----------------------")
# #         response = query_engine.chat(input_text) 
# #         print(response)
# #         return str(response)
 
# #     else:
# #         return {'result': 'Unsupported language detected.'}

# @app.route('/ask pdf', methods=['POST'])
# def ask_pdf():
#     if request.method == 'POST':
#         input_text = request.form['question']
#         documents = SimpleDirectoryReader('uploads').load_data()
#         Settings.llm = llm
#         Settings.embed_model = embed_model
#         index = VectorStoreIndex.from_documents(documents)
#         query_engine = index.as_query_engine()
        
#         print("generating response----------------------")
#         response = query_engine.query(input_text) 
#         print(response)
#         return str(response)
 
#     else:
#         return {'result': 'Unsupported language detected.'}

    
# if __name__ == '__main__':
#     app.run(debug=False,port=5001)






