import os
import re
from typing import List
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')

class Document:
    def __init__(self, name, text):
        self.name = name
        self.text = text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF файла."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def load_documents_from_pdfs(pdf_folder: str) -> List[Document]:
    """Читает все PDF файлы из папки и возвращает список объектов Document."""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            document = Document(name=filename, text=text)
            documents.append(document)
        print(f"Завершение чтение документа {filename}...")
    return documents

def save_embeddings(embeddings, file_path):
    torch.save(embeddings, file_path)
    print(f"Эмбеддинги сохранены в {file_path}")
    ok = torch.load(file_path)

def main():
    docs = load_documents_from_pdfs('C:/Users/ICSS2location10/Desktop/diplom-pract/pre-datasets')
    document_embeddings = [model.encode(doc.text) for doc in docs]
    save_embeddings(document_embeddings, "embeddings.pt")

if __name__ == "__main__":
    main()
