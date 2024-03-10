import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, session
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# from flask_sqlalchemy import SQLAlchemy
import pickle
from docx import Document as DocxDocument
import docx2txt
import helper_func.helper_func as helper_func



# Create a Flask application instance
app = Flask(__name__)

# Set the upload folder path for uploaded files
app.config['UPLOAD_FOLDER'] = './docs/'

# Set the saved variable folder path for saving variables
app.config['SAVED_VAR'] = './saved-var/'

# Set a secret key for the application
app.secret_key = 'question_answer_test'

# Define an upload folder variable for saved files
UPLOAD_FOLDER = 'saved-var'

# Load environment variables from a .env file
load_dotenv('.env')

# # Set the SQLALCHEMY_DATABASE_URI to connect to a SQLite database named test_answers.db
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_answers.db'

# # Initialize the SQLAlchemy object using the Flask app
# db = SQLAlchemy(app)


class DocxLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        doc = DocxDocument(self.file_path)
        return [para.text for para in doc.paragraphs]

class DocLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return [docx2txt.process(self.file_path)]
    
class TxtLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]


##########################################################
# GLOBAL VARIABLES
# Define a global variable to store the test_question_id
test_question_id = 0

# Declare global variables
glo_test_question = None
glo_test_answer = None

def store_data(question, answer):
    # Access the global variables and store the data
    global glo_test_question, glo_test_answer
    glo_test_question = question
    glo_test_answer = answer

def retrieve_data():
    # Access the global variables and retrieve the data
    global glo_test_question, glo_test_answer
    return glo_test_question, glo_test_answer


# Declare global variable
glo_test_question_id = 0

def increment_test_question_id():
    # Access the global variable and increment by 1
    global glo_test_question_id
    glo_test_question_id += 1

def retrieve_test_question_id():
    # Access the global variable and retrieve the value
    global glo_test_question_id
    return glo_test_question_id
#################################################################

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
    # (please check the helper function for implementations)
    helper_func.save_uploaded_files(uploaded_files, upload_folder)

    # Assuming documents is a list to store loaded document contents
    documents = []

    # Directory containing documents
    docs_dir = "./docs"

    # Iterate through files in the directory
    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        
        # Load PDF documents
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        
        # Load DOCX documents
        elif file.endswith('.docx'):
            loader = DocxLoader(file_path)
            documents.extend(loader.load())
        
        # Load DOC documents
        elif file.endswith('.doc'):
            loader = DocLoader(file_path)
            documents.extend(loader.load())

        # Load TXT documents
        elif file.endswith('.txt'):
            loader = TxtLoader(file_path)
            documents.extend(loader.load())

        else:
            return jsonify({'message': 'File(s) not supported'})

    # Delete uploaded files after processing
    # (please check the helper function for implementations)
    helper_func.delete_all_files(upload_folder)

    # Delete all files in the saved variable folder
    # (please check the helper function for implementations)
    saved_var = app.config['SAVED_VAR']
    helper_func.delete_all_files(saved_var)

    # Serialize and save processed documents to a pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'wb') as f:
        pickle.dump(documents, f)

    # Set test_question_id to 0 in the session
    session['test_question_id'] = 0

    # Return a JSON response indicating successful file upload and processing
    return jsonify({'message': 'Files uploaded and processed successfully'})

# Route decorator for querying documents with GET and POST methods
@app.route('/query/', methods=['GET','POST'])
# Function to handle document queries
def query_document():
    # Load serialized documents from the pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'rb') as f:
        documents = pickle.load(f)

    model = ChatOpenAI(temperature=0)

    class QatQuery(BaseModel):
        answer: str = Field(description="The answer provided by the system to the question asked")
        bullet_points: list = Field(description="A list of bullet points emphasizing key details in the answer to improve understanding")
        test_question: str = Field(description="Generated question to evaluate if the user understood the answer provided")
        test_answer: str = Field(description="Generate an answer which will be used to evaluate the user answer for the generated question")

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=QatQuery)

    prompt = PromptTemplate(
        template="Answer the user query.\n{context}\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions(), "context": documents},
    )

    chain = prompt | model | parser

    # Extract user question from the request JSON data
    query_data = request.json
    user_question = query_data['question']

    try:
        result = chain.invoke({"query": user_question})
    except Exception as e:
        result={}
        result['answer'] = ""
        result['bullet_points'] = []
        result['test_question'] = ""
        result['test_answer'] = ""


    store_data(result['test_question'], result['test_answer'])

    increment_test_question_id()

    test_question_id = retrieve_test_question_id()

    # Return the retrieved answer, bullet points, test question, and test question ID in JSON format
    return jsonify({'answer': result['answer'], \
                    'bullet_points': result['bullet_points'], \
                    'test_question': result['test_question'], \
                    'test_question_id': test_question_id})

# Route decorator for handling evaluation requests with GET and POST methods
@app.route('/evaluate/', methods=['GET','POST'])
# Function to evaluate user understanding
def evaluate_understanding():
    # Load serialized documents from the pickle file
    with open(os.path.join(UPLOAD_FOLDER, 'documents.pkl'), 'rb') as f:
        documents = pickle.load(f)

    model = ChatOpenAI(temperature=0)

    class QatEvaluate(BaseModel):
        knowledge_understood: bool = Field(description="This is a boolean value indicating that the user understood the answer provided. \
                                           True if the user understood the answer, False if the user did not understand the answer")
        knowledge_confidence: int = Field(description="This is an integer value (in %) indicating how confident the evaluation is.")

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=QatEvaluate)

    prompt = PromptTemplate(
        # template="Answer the user query.\n{context}\n{format_instructions}\n{query}\n",
        template="This {test_answer} would be used to evaluate the user {answer} for the provided {test_question}.\n{context}\n{format_instructions}\n",
        input_variables=["answer", "test_question", "test_answer"],
        partial_variables={"format_instructions": parser.get_format_instructions(), "context": documents},
    )

    chain = prompt | model | parser

    # Extract user's answer from the request JSON data
    query_data = request.json
    user_answer = query_data['answer']

    # test_question = session.get('test_question')
    # # Retrieve the test answer from the session
    # test_answer = session.get('test_answer')

    test_question, test_answer = retrieve_data()

    try:
        result = chain.invoke({"answer": user_answer, "test_question": test_question, "test_answer": test_answer})
    except Exception as e:
        result={}
        result['knowledge_understood'] = False
        result['knowledge_confidence'] = 0


    # Return the knowledge understood and confidence level in JSON format
    return jsonify({'knowledge_understood': result["knowledge_understood"], \
                    'knowledge_confidence': result["knowledge_confidence"]})

# Check if the script is being run directly as the main program
if __name__ == '__main__':
    # Run the Flask application in debug mode
    app.run(debug=True)


