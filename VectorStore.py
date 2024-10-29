from abc import abstractmethod
import os
from typing import List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStore:
    def __init__(self, docs: List[Document]) -> None:
        self.docs = docs
        self.persist_directory = "./vector_store"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embeddings = OpenAIEmbeddings()
    
    def get_vector_store(self) -> Chroma:
        if self._exists_directory:
            return self._load_vector_store()
        else:            
            return self._create_vector_store()

    def _create_vector_store(self) -> Chroma:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        splits = text_splitter.split_documents(self.docs)
        
        return Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings, 
            persist_directory=self.persist_directory
        )
        
    def _load_vector_store(self) -> Chroma:
        return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
    
    @property
    def _exists_directory(self):
        return os.path.exists(self.persist_directory) and os.listdir(self.persist_directory)
