import os
from urllib.parse import unquote

import bs4
from flask import Flask, request, abort
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


app = Flask(__name__)

load_dotenv()  # Load .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
persist_directory = "./vector_store"
embeddings = OpenAIEmbeddings()
web_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# 127.0.0.1:5000/?question=what+is+an+agent%3F

@app.get('/')
def answer_question():
    question = request.args.get('question')
    if not question:
        abort(400, description="Bad Request: 'question' parameter is required.")

    question = unquote(question)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # ----------------- Load, chunk and index the contents of the blog. ---------------- #
    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    # Check if we have loaded already the embed
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # ----------------- Split and store in vector store. ---------------- #
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    else:
        # ----------------- Just load the Vector Store. ---------------- #
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # ------ Retrieve and generate using the relevant snippets of the blog. --- #
    retriever = vector_store.as_retriever()

    # Pulling prompts https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=f2ce30f8-79ed-4f3e-b060-9241176d510b
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(piped_docs):
        return "\n\n".join(doc.page_content for doc in piped_docs)

    # NOTE: The pipes are python functional programing style
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return f"{rag_chain.invoke(question)}"


if __name__ == '__main__':
    app.run()
