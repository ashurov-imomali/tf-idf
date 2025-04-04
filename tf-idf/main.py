import json
import math
import re

import pymorphy2
import pdfplumber
import os
from typing import List, TextIO

morph = pymorphy2.MorphAnalyzer()


class Document:
    def __init__(self, name, text):
        self.name = name
        self.text = text

class DocumentTermFrequency:
    def __init__(self, document_name):
        self.document_name = document_name
        self.term_frequencies = {}


class TFIDFResult:
    def __init__(self, document_name):
        self.document_name = document_name
        self.tfidf_values = []
    def to_dict(self):
        return {"document_name": self.document_name, "tfidf_values": self.tfidf_values}

class SearchResult:
    def __init__(self, document_name, relevance_score):
        self.document_name = document_name
        self.relevance_score = relevance_score
    def to_dict(self):
            return {"doc_name": self.document_name, "score": self.relevance_score}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF файла."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def remove_punctuation(text):
     return re.sub(r'[^a-zA-Z^а-яА-Я0-9 ]', '', text)

def load_documents_from_pdfs(pdf_folder: str) -> List[Document]:
    """Читает все PDF файлы из папки и возвращает список объектов Document."""
    documents = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            text = remove_punctuation(text)
            document = Document(name=filename, text=text)
            documents.append(document)
        print(f"Завершение чтение документа {filename}...")
    return documents



def save_to_json(filename, data):
    search_results_dict = [result.to_dict() for result in data]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(search_results_dict, f, ensure_ascii=False, indent=4)
    print(f"Данные успешно сохранены в файл: {filename}")



# оставил если не найдется датасет
def load_sample_documents_ru():
    return [
        Document("doc1", "Яблоки полезны для здоровья. Они содержат витамины и антиоксиданты."),
        Document("doc2", "Космос – это бесконечное пространство, полное загадок и тайн."),
        Document("doc3", "Программирование – это процесс создания программ для компьютеров."),
        Document("doc4", "Путешествия помогают расширять кругозор и узнавать новые культуры."),
        Document("doc5", "Спорт укрепляет организм и помогает поддерживать хорошую физическую форму."),
        Document("doc6", "Искусственный интеллект меняет мир, улучшая технологии и автоматизируя процессы."),
        Document("doc7", "Музыка влияет на настроение человека и может вызывать сильные эмоции."),
        Document("doc8", "История – это наука, изучающая прошлое человечества."),
        Document("doc9", "Книги помогают развивать мышление и воображение, обогащая словарный запас."),
        Document("doc10", "Экология важна для сохранения природы и здоровья будущих поколений.")
    ]


def count_documents_containing_words(documents):
    word_document_count = {}
    total_count = 0
    for document in documents:
        words_in_document = set()
        for token in document.text.split():
            normalized_word = morph.parse(token)[0].normal_form
            if normalized_word not in words_in_document:
                word_document_count[normalized_word] = word_document_count.get(normalized_word, 0) + 1
                total_count += 1
                words_in_document.add(normalized_word)
    return word_document_count, total_count


def calculate_term_frequencies(documents):
    term_frequencies_list = []
    for document in documents:
        term_frequency = DocumentTermFrequency(document.name)
        tokens = document.text.split()
        total_tokens = len(tokens)

        frequency_map = {}
        for token in tokens:
            normalized_word = morph.parse(token)[0].normal_form
            frequency_map[normalized_word] = frequency_map.get(normalized_word, 0) + 1

        for word, count in frequency_map.items():
            term_frequency.term_frequencies[word] = count / total_tokens

        term_frequencies_list.append(term_frequency)
        print(f"Завершение подсчёта tf для документа {document.name}...")
    return term_frequencies_list


def calculate_inverse_document_frequencies(documents):
    word_document_count, total_count = count_documents_containing_words(documents)
    total_documents = len(documents)
    inverse_document_frequencies = {}
    i = 0
    for word, document_count in word_document_count.items():
        inverse_document_frequencies[word] = math.log2(total_documents / document_count)
        i += 1
        print(f"\rProgress: {(100 * i)/total_count}%", end='', flush=True)

    print("")
    return inverse_document_frequencies

def main():
    pdf_folder_path = 'C:/Users/ICSS2location10/Desktop/diplom-pract/pre-datasets'
    documents = load_documents_from_pdfs(pdf_folder_path)

    # documents = load_sample_documents_ru()

    term_frequencies = calculate_term_frequencies(documents)

    inverse_document_frequencies = calculate_inverse_document_frequencies(documents)
    tfidf_results = []
    for term_frequency in term_frequencies:
        tfidf_result = TFIDFResult(term_frequency.document_name)
        for word, tf_value in term_frequency.term_frequencies.items():
            tfidf_result.tfidf_values.append({word: tf_value * inverse_document_frequencies[word]})
        print(f"Завершение подсчёта tf-idf для документа {term_frequency.document_name}...")
        tfidf_results.append(tfidf_result)

    save_to_json("docs-tf-idf.json", tfidf_results)


if __name__ == "__main__":
    main()