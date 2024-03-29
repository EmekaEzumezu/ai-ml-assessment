# Nurovant AI ML Assessment: Multi Document Reader and Chatbot

## Overview
This app is a Q.A.T (Question, Answer & Test) system for research documents. The Q.A.T system is very similar to the typical Q&A (Question & Answer) system, the difference here lies in the response provided by the Q.A.T system. The workflow for a normal Q&A system entails asking a question w.r.t a specific document and a response is generated as an answer to the question. For a Q.A.T system a “test_question” is returned along with the answer and some other information (if necessary), the “test_question” is used to evaluate whether the user understood the response generated by the system. For this assessment, it is required to build out a flask API with 3 major endpoints:

The “upload/” endpoint: This endpoint would be used to upload the research document to the application.
The “query/” endpoint: This endpoint would be used as an interactive Q.A.T session between the user and the system.
The “evaluate/” endpoint: This endpoint would be used to evaluate the response provided by the user with the “test_question”.
The expected response format for the following endpoints are indicated below, respectively:

1. “upload/” : Feel free to return the response based on your own discretion.
2. “query/”: The json response should be as follows:
```
{
    “answer”: “ ...”,
    “bullet_points”: [ ..., ..., ..., ...],
    “test_question”: “...”,
    “test_question_id:”...
}
```

3. “evaluate/” :The json response should be as follows:
```
{
    “knowledge_understood”: bool,
    “knowledge_confidence”: int
}
```

Response definition for “query/”:

1. answer: The answer provided by the system to the question asked.
2. bullet_points: A list of bullet points emphasizing key details in the answer to improve understanding.
3. test_question : Generated question to evaluate if the user understood the answer provided.
4. test_question_id: id for every test_question generated.
Response definition for “evaluate/”:

1. knowledge_understood: This is a boolean value indicating that the user understood the answer provided. True if the user understood the answer, False if the user did not understand the answer.
2. knowledge_confidence: This is an integer value (in %) indicating how confident the evaluation is.

## Getting Started
1. Clone the repository, set up the conda environment, and install the required packages:
```
git clone https://github.com/EmekaEzumezu/ai-ml-assessment.git
cd ai-ml-assessment
conda create --name myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

### Store your OpenAI API key:
1. Copy the example environment file: cp .env.example .env.
2. Paste your OpenAI API key into the .env file: OPENAI_API_KEY=sk-.

### Start chatting:
Launch the the app, upload, and interact with your files. ctrl+c to exit the prompt at any time.

```
python app.py
```

### Conclusion
While the project provides a basic setup for a multi-document reader and chatbot, it acknowledges that achieving a high-performing chatbot requires further exploration and optimization. Future enhancements could include refining prompt templates, experimenting with different LM models, creating agents to refine results, and more.

### Screenshots

![Screenshot-1.png](./img/Screenshot-1.png)

![Screenshot-1.png](./img/Screenshot-2.png)

