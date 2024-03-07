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

# from PyPDF2 import PdfReader
# from docx import Document
import pickle
import shutil
import json
import psutil

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload/', methods=['GET','POST'])
def upload_pdf():

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
