from urllib.parse import unquote
from flask import Flask, request, abort
from dotenv import load_dotenv
from langchain import hub
from rag_function import compute
app = Flask(__name__)

load_dotenv()  # Load .env file

# web_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"# 127.0.0.1:5000/?question=what+is+an+agent%3F
prompt = hub.pull("rlm/rag-prompt") # Pulling prompts https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=f2ce30f8-79ed-4f3e-b060-9241176d510b
print (type(prompt))
@app.get('/')
def answer_question():

    question = request.args.get('question')
    if not question:
        abort(400, description="Bad Request: 'question' parameter is required.")

    question = unquote(question)    
    
    return compute(prompt, question)        

if __name__ == '__main__':
    app.run()
