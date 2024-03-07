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
    # Define the prompt template as a Python dictionary
    prompt_template = {
        'answer': f"Generate an answer to the question ({user_question}).",
        'bullet_points': 'Generate a Python list (4 points in a list separated by comma) emphasizing key details in the answer to improve understanding. Example: ["Point 1", "Point 2", "Point 3", "Point 4"]',
        'test_question': f"Generate a random question different from {user_question} to evaluate if the user understood the answer",
        'test_answer': 'Generate a test_answer which will be used to evaluate the user answer for the provided test_question.'
    }

    return json.dumps(prompt_template)

# Function to set an evaluation prompt template for comparing user answers to test answers
def set_eval_prompt_template(user_answer, test_answer):
    # Define the evaluation prompt template as a Python dictionary
    eval_prompt_template = {
        'knowledge_understood': f"Generate a boolean value indicating that the user understood the answer provided. That's when user answer ({user_answer}) is compare to the test answer ({test_answer}) previously generated. True if the user understood the answer, False if the user did not understand the answer.",
        'knowledge_confidence': "Generate an integer value (in %) indicating how confident the evaluation is."
    }

    return json.dumps(eval_prompt_template)
