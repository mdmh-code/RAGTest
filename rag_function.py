from DocumentSource import DocumentSourceFile
from GenIARag import GenIARag
from VectorStore import VectorStore

def compute(rag_prompt: str, user_prompt):
    doc_source = DocumentSourceFile("./files/source.txt")
    vector_store = VectorStore(doc_source.get_docs())
    retriever = vector_store.get_vector_store().as_retriever()    
    rag = GenIARag(retriever)
    type(rag_prompt)
    rag_chain = rag.get_rag_chain(rag_prompt)

    return f"{rag_chain.invoke(user_prompt)}"