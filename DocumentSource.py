from typing import List
import bs4
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.documents import Document

class DocumentSouceWeb:
    def __init__(self, web_url: str) -> None:
        self.web_url = web_url
    
    def get_docs(self) ->List [Document]:
        loader = WebBaseLoader(
            web_paths=(self.web_url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        return loader.load()


class DocumentSourceFile:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        
    def get_docs(self) ->List [Document]:        

        # Create a TextFileDocumentLoader instance
        loader = TextLoader(self.file_path, encoding='utf-8')

        # Load the document from the text file
        return loader.load()
