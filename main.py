import math
import re

import pymorphy2


morph = pymorphy2.MorphAnalyzer()

class Doc:
    def __init__(self, name, text):
        self.name = name
        self.text = text


def init_test_docs():
    return [
        Doc("doc1", "ananas banana apple"),
        Doc("doc2", "apple orange apple banana apple"),
        Doc("doc3", "grape ananas peach banana ananas"),
        Doc("doc4", "mango pineapple banana"),
        Doc("doc5", "strawberry blueberry raspberry ananas ananas"),
        Doc("doc6", "banana apple peach"),
        Doc("doc7", "pineapple mango orange"),
        Doc("doc8", "grape banana strawberry"),
        Doc("doc9", "blueberry raspberry ananas"),
        Doc("doc10", "orange mango banana orange orange orange ananas ananas")
    ]

def init_test_docs_ru():
    return [
        Doc("doc1", "Яблоки полезны для здоровья. Они содержат витамины и антиоксиданты."),
        Doc("doc2", "Космос – это бесконечное пространство, полное загадок и тайн."),
        Doc("doc3", "Программирование – это процесс создания программ для компьютеров."),
        Doc("doc4", "Путешествия помогают расширять кругозор и узнавать новые культуры."),
        Doc("doc5", "Спорт укрепляет организм и помогает поддерживать хорошую физическую форму."),
        Doc("doc6", "Искусственный интеллект меняет мир, улучшая технологии и автоматизируя процессы."),
        Doc("doc7", "Музыка влияет на настроение человека и может вызывать сильные эмоции."),
        Doc("doc8", "История – это наука, изучающая прошлое человечества."),
        Doc("doc9", "Книги помогают развивать мышление и воображение, обогащая словарный запас."),
        Doc("doc10", "Экология важна для сохранения природы и здоровья будущих поколений.")
    ]



def find_word_doc(docs):
    word_in_docs = {}
    for doc in docs:
        seen_in_current_doc = set()
        for token in doc.text.split():
            if token not in seen_in_current_doc:
                r_token = morph.parse(token)[0].normal_form
                word_in_docs[r_token] = word_in_docs.get(r_token, 0) + 1
                seen_in_current_doc.add(r_token)
    return word_in_docs


class DocTF:
    def __init__(self, name):
        self.name = name
        self.tf = {}


def find_doc_tf(docs):
    results = []
    for doc in docs:
        doc_tf = DocTF(doc.name)
        tokens = doc.text.split()
        length = len(tokens)

        freq_map = {}
        for token in tokens:
            r_token = morph.parse(token)[0].normal_form
            freq_map[r_token] = freq_map.get(r_token, 0) + 1

        for word, count in freq_map.items():
            doc_tf.tf[word] = count / length

        results.append(doc_tf)
    return results


def find_word_idf(docs):

    word_in_docs = find_word_doc(docs)
    total_docs = len(docs)
    idf_map = {}
    for word, doc_count in word_in_docs.items():
        idf_map[word] = math.log2(total_docs / doc_count)
    return idf_map


class TfIdf:
    def __init__(self, name):
        self.name = name
        self.tf_idf = []

class SearchResult:
    def __init__(self, doc_name, reality_sum):
        self.doc_name = doc_name
        self.reality_sum  = reality_sum


def search(search_info, tfIdfs):
    search_strings = search_info.split()
    results = []
    for value in tfIdfs:
        temp = 0
        for search_string in search_strings:
            for mp in value.tf_idf:
                term = morph.parse(search_string)[0].normal_form
                temp += mp.get(term, 0)
        results.append(SearchResult(value.name, temp))
    results.sort(key=lambda x: x.reality_sum, reverse=True)
    return results


def main():
    docs = init_test_docs_ru()
    for doc in docs:
        doc.text =  re.sub(r'[^\w\s]', '', doc.text)


    # for doc in docs:
    #     print(f"name: {doc.name}, text: {doc.text}")

    docs_tf = find_doc_tf(docs)

    words_idf = find_word_idf(docs)

    results = []
    for doc_tf in docs_tf:
        tfidf_obj = TfIdf(doc_tf.name)
        for word, tf_val in doc_tf.tf.items():
            tfidf_obj.tf_idf.append({word: tf_val * words_idf[word]})
        results.append(tfidf_obj)

    # for item in results:
    #     print(f"Document: {item.name}")
    #     print("TF-IDF: ", item.tf_idf)
    #     print("----------")
    search_data = 'здоровья яблоко поколение важный'
    search_res = search(search_data, results)
    for item in search_res:
        print(f"document: {item.doc_name} reality: {item.reality_sum}")


if __name__ == "__main__":
    main()
