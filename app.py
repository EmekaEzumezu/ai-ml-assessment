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

import pickle
import shutil
import json
import psutil

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

# Function to save uploaded files to a specified upload folder
def save_uploaded_files(uploaded_files, upload_folder):
    # Iterate through each uploaded file
    for file in uploaded_files:
        # Save the file to the specified upload folder with its original filename
        file.save(os.path.join(upload_folder, file.filename))

# Function to delete all files and subdirectories in a specified folder
def delete_all_files(folder_path):
    # Get a list of all files and subdirectories in the specified folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # Check if the item is a file
        if os.path.isfile(item_path):
            # If it's a file, remove it
            os.remove(item_path)
        # Check if the item is a directory
        elif os.path.isdir(item_path):
            # If it's a directory, remove it recursively
            shutil.rmtree(item_path)
            
# Function to set a prompt template for generating a Python dictionary based on user input
def set_prompt_template(user_question):
    # Define the prompt template with placeholders for user input
    prompt_template = """
    To every question asked generate a Python dictionary with the following keys and values (the values will be as described):
    {{
        'answer': 'The answer provided by the system to the question ({user_question}) asked.',
        'bullet_points': 'A Python list (4 points in a list separated by comma) emphasizing key details in the answer to improve understanding.',
        'test_question': 'Generated question to evaluate if the user understood the answer',
        'test_answer': 'This test_answer would be used to evaluate the user answer for the provided test_question.'
    }}
    """.format(user_question=user_question)

    return prompt_template

# Function to set an evaluation prompt template for comparing user answers to test answers
def set_eval_prompt_template(user_answer, test_answer):
    # Define the evaluation prompt template with placeholders for user and test answers
    eval_prompt_template = """
    To every question asked generate a Python dictionary that compares the answer given by a user to test his/her understanding of {user_answer} with the answer generated based on the document/context {test_answer}. Then 

    {{
    "knowledge_understood": "This is a boolean value indicating that the user understood the
    answer provided. True if the user understood the answer, False if the user did not
    understand the answer.",
    “knowledge_confidence”: "This is an integer value (in %) indicating how confident the
    evaluation is."
    }}
    """.format(user_answer=user_answer, test_answer=test_answer)

    return eval_prompt_template

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
    save_uploaded_files(uploaded_files, upload_folder)

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
    delete_all_files(upload_folder)

    # Delete all files in the data folder
    data = app.config['DATA']
    delete_all_files(data)

    # Delete all files in the saved variable folder
    saved_var = app.config['SAVED_VAR']
    delete_all_files(saved_var)

    # Serialize and save processed documents to a pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'wb') as f:
        pickle.dump(documents, f)

    # Set test_question_id to 0 in the session
    session['test_question_id'] = 0

    # Return a JSON response indicating successful file upload and processing
    return jsonify({'message': 'Files uploaded and processed successfully'})

@app.route('/query/', methods=['GET','POST'])
def query_document():

    return jsonify({'answer': answer, \
                    'bullet_points': bullet_points, \
                    'test_question': test_question, \
                    'test_question_id': test_question_id})

@app.route('/evaluate/', methods=['GET','POST'])
def evaluate_understanding():
    # Implement evaluation logic here


    return jsonify({'knowledge_understood': knowledge_understood, \
                    'knowledge_confidence': knowledge_confidence})

if __name__ == '__main__':
    app.run(debug=True)
