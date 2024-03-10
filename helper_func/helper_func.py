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
            
# # # Function to set a prompt template for generating a Python dictionary based on user input
# # def set_prompt_template(user_question):
# #     # Define the prompt template as a Python dictionary
# #     prompt_template = {
# #         'answer': f"Generate an answer to the question ({user_question}).",
# #         'bullet_points': 'Generate a Python list (4 points in a list separated by comma) emphasizing key details in the answer to improve understanding. Example: ["Point 1", "Point 2", "Point 3", "Point 4"]',
# #         'test_question': f"Generate a random question different from {user_question} to evaluate if the user understood the answer",
# #         'test_answer': 'Generate a test_answer which will be used to evaluate the user answer for the provided test_question.'
# #     }

# #     return json.dumps(prompt_template)

# # Function to set a prompt template for generating a Python dictionary based on user input
# def set_prompt_template(user_question):
#     # Define the prompt template as a Python dictionary
#     prompt_template = f"Always return answers these 4 questions:\
#         1. Generate an answer to the question ({user_question}).\
#         2. Generate a Python list (4 points in a list separated by comma) emphasizing key details in the answer to improve understanding. Example: ['Point 1', 'Point 2', 'Point 3', 'Point 4']\
#         3. Generate a test question different from {user_question} to evaluate if the user understood the answer\
#         4. Generate a test_answer which will be used to evaluate the user answer for the provided test_question (on number 3).\
#     "

#     return prompt_template

# # Function to set an evaluation prompt template for comparing user answers to test answers
# def set_eval_prompt_template(user_answer, test_answer):
#     # Define the evaluation prompt template as a string
#     eval_prompt_template = f"Always return answers to these 2 questions:\
#         1. Generate a boolean value indicating that the user understood the answer provided. That's when user answer ({user_answer}) is compared to the test answer ({test_answer}) previously generated. True if the user understood the answer, False if the user did not understand the answer.\
#         2. Generate an integer value (in %), that's between 0-100, indicating how confident the evaluation is.\
#     "

#     return eval_prompt_template

# def process_data():
#     # Load serialized documents from the pickle file
#     with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'rb') as f:
#         documents = pickle.load(f)

#     # Split text into chunks for processing
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
#     documents = text_splitter.split_documents(documents)

#     # Create and persist a vector database for document retrieval
#     vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
#     vectordb.persist()

#     # Set up conversational retrieval with OpenAI model
#     pdf_qa = ConversationalRetrievalChain.from_llm(
#         ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
#         retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
#         return_source_documents=True,
#         verbose=False
#     )

#     return pdf_qa

# def get_query_results(pdf_qa, query):
#     chat_history = []
#     # Invoke the conversational retrieval process
#     result = pdf_qa.invoke(
#         {"question": query, "chat_history": chat_history})

#     results = str(result["answer"])

#     return results



# # # Define a model TestQuestion with id, question, and answer columns
# # class TestQuestion(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)  # Primary key column for the TestQuestion table
# #     question = db.Column(db.String(255), nullable=False)  # Column to store the question text
# #     answer = db.Column(db.String(255), nullable=False)  # Column to store the answer text

# #     def __repr__(self):
# #         return f'<TestQuestion {self.id}>'  # Representation method for TestQuestion model

# # # Define a model Question with id and answer_text columns
# # class Question(db.Model):
# #     id = db.Column(db.Integer, primary_key=True)  # Primary key column for the Question table
# #     answer_text = db.Column(db.String(255))  # Column to store the answer text            

