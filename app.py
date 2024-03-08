import os
import sys
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, session
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

import pickle
import shutil
import json
import psutil
import ast
import re

import helper_func.helper_func as helper_func


# Create a Flask application instance
app = Flask(__name__)

# Set the upload folder path for uploaded files
app.config['UPLOAD_FOLDER'] = './docs/'

# Set the data folder path for storing data
app.config['DATA'] = './data/'

# Set the saved variable folder path for saving variables
app.config['SAVED_VAR'] = './saved-var/'

# Set a secret key for the application
app.secret_key = 'question_answer_test'

# Define an upload folder variable for saved files
UPLOAD_FOLDER = 'saved-var'

# Load environment variables from a .env file
load_dotenv('.env')
            

# Route decorator for the home page
@app.route('/')
# Function to handle requests to the home page
def index():
    # Render the index.html template
    return render_template('index.html')

# Route decorator for uploading files with GET and POST methods
@app.route('/upload/', methods=['GET','POST'])
# Function to handle file uploads
def upload_pdf():

    # Get list of uploaded files
    uploaded_files = request.files.getlist('files')
    # Retrieve the upload folder path from app config
    upload_folder = app.config['UPLOAD_FOLDER']
    # Save the uploaded files to the specified upload folder
    helper_func.save_uploaded_files(uploaded_files, upload_folder)

    documents = []
    # Iterate through files in the "docs" directory for processing
    for file in os.listdir("docs"):
        # Load PDF files
        if file.endswith(".pdf"):
            pdf_path = "./docs/" + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        # Load Word documents
        elif file.endswith('.docx') or file.endswith('.doc'):
            doc_path = "./docs/" + file
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        # Load Txt documents
        # elif file.endswith('.txt'):
        #     text_path = "./docs/" + file
        #     loader = TextLoader(text_path)
        #     documents.extend(loader.load())

    # Delete uploaded files after processing
    helper_func.delete_all_files(upload_folder)

    # Delete all files in the data folder
    data = app.config['DATA']
    helper_func.delete_all_files(data)

    # Delete all files in the saved variable folder
    saved_var = app.config['SAVED_VAR']
    helper_func.delete_all_files(saved_var)

    # Serialize and save processed documents to a pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'wb') as f:
        pickle.dump(documents, f)

    # Set test_question_id to 0 in the session
    session['test_question_id'] = 0

    # session['shot'] = 0

    # Return a JSON response indicating successful file upload and processing
    return jsonify({'message': 'Files uploaded and processed successfully'})

# Route decorator for querying documents with GET and POST methods
@app.route('/query/', methods=['GET','POST'])
# Function to handle document queries
def query_document():

    # Load serialized documents from the pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'rb') as f:
        documents = pickle.load(f)

    # Split text into chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Create and persist a vector database for document retrieval
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    # Set up conversational retrieval with OpenAI model
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    # Extract user question from the request JSON data
    query_data = request.json
    user_question = query_data['question']

    # shot = session.get('shot')
    # if shot == 0:
    # Generate a prompt based on the user question
    query = helper_func.set_prompt_template(user_question)
    # shot += 1
    # session['shot'] = shot
    # else:
    #     query = user_question

    chat_history = []
    # Invoke the conversational retrieval process
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})

    results = str(result["answer"])

    # # Process the retrieved results
    # results = results.replace("'", '"')
    
    # # Try to load the results as JSON and extract relevant fields
    # try:
    #     results = json.loads(str(results))
    #     answer = results["answer"]
    #     bullet_points = results["bullet_points"]
    #     test_question = results["test_question"]
    #     test_answer = results["test_answer"]
    # # Handle exception if unable to extract fields from results
    # except:
    #     answer = "Couldn't get the answer. Please try again"
    #     bullet_points = ["Please try again"]
    #     test_question = "Please try again"
    #     test_answer = "Default test answer"

    # session['test_answer'] = test_answer
    print(results)

    try:
        sections = re.split(r'\b(?:1\.|2\.|3\.|4\.)\s*', results)

        # Remove empty parts
        sections = [part.strip() for part in sections if part.strip()]

        # Assign each section to a separate variable
        answer = sections[0].strip()  # Remove the numbering and leading/trailing whitespace
        bullet_points = ast.literal_eval(sections[1].strip())
        test_question = sections[2].strip()
        test_answer = sections[3].strip()  # Remove the "Test_answer: " prefix and leading/trailing whitespace

    except (IndexError, ValueError) as e:
        # Assign default values
        answer = ""
        bullet_points = []
        test_question = ""
        test_answer = ""

    # Print the variables
    print("Answer:", answer)
    print("Bullet Points:", bullet_points)
    print("Test Question:", test_question)
    print("Test Answer:", test_answer)

    # Retrieve the test_question_id from the session or set to 0 if not present
    test_question_id = session.get('test_question_id')
    # Increment the test_question_id by 1
    test_question_id += 1
    # Update the session with the new test_question_id
    session['test_question_id'] = test_question_id

    # Return the retrieved answer, bullet points, test question, and test question ID in JSON format
    return jsonify({'answer': answer, \
                    'bullet_points': bullet_points, \
                    'test_question': test_question, \
                    'test_question_id': test_question_id})

# Route decorator for handling evaluation requests with GET and POST methods
@app.route('/evaluate/', methods=['GET','POST'])
# Function to evaluate user understanding
def evaluate_understanding():

    # Load serialized documents from the pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'rb') as f:
        documents = pickle.load(f)

    # Split text into chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    documents = text_splitter.split_documents(documents)

    # Create and persist a vector database for document retrieval
    vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
    vectordb.persist()

    # Set up conversational retrieval with OpenAI model
    pdf_qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    # Extract user's answer from the request JSON data
    query_data = request.json
    user_answer = query_data['answer']

    # Retrieve the test answer from the session
    test_answer = session.get('test_answer')

    # Generate a prompt based on the user's answer and test answer
    query = helper_func.set_eval_prompt_template(user_answer, test_answer)

    chat_history = []
    # Invoke the conversational retrieval process
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})

    # Convert the answer from the result to a string
    results = str(result["answer"])

    # Replace single quotes with double quotes in the results string
    # results = results.replace("'", '"')

    # try:
    #     # Parse the results as JSON and extract knowledge understanding and confidence
    #     results = json.loads(results)
    #     knowledge_understood = results["knowledge_understood"]
    #     knowledge_confidence = results["knowledge_confidence"]

    # except:
    #     # Handle exceptions if unable to extract knowledge understanding and confidence
    #     knowledge_understood = False
    #     knowledge_confidence = "0"

    print(results)

    try:
        sections = re.split(r'\b(?:1\.|2\.|3\.|4\.)\s*', results)

        # Remove empty parts
        sections = [part.strip() for part in sections if part.strip()]

        # Assign each section to a separate variable
        knowledge_understood = bool(sections[0].strip())  # Remove the numbering and leading/trailing whitespace
        knowledge_confidence = sections[1].strip()  # prefix and leading/trailing whitespace

    except:
        # Assign default values
        knowledge_understood = False
        knowledge_confidence = 0

    print(knowledge_understood)  # Output: True
    print(knowledge_confidence)  # Output: 90


    # Return the knowledge understood and confidence level in JSON format
    return jsonify({'knowledge_understood': knowledge_understood, \
                    'knowledge_confidence': knowledge_confidence})

# Check if the script is being run directly as the main program
if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)