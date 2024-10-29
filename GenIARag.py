from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

class GenIARag:
    
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self.retriever = retriever
        self.model = "gpt-4o-mini"
    
    def get_rag_chain(self, prompt: str):
        
        def format_docs(piped_docs):
            return "\n\n".join(doc.page_content for doc in piped_docs)

        llm = ChatOpenAI(model=self.model)

        return (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )