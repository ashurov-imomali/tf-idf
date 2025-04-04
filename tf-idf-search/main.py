import json
from typing import List


class SearchResult:
    def __init__(self, document_name, relevance_score):
        self.document_name = document_name
        self.relevance_score = relevance_score

    def to_dict(self):
        return {"doc_name": self.document_name, "score": self.relevance_score}

class TFIDFResult:
    def __init__(self, document_name):
        self.document_name = document_name
        self.tfidf_values = []
    def to_dict(self):
        return {"document_name": self.document_name, "tfidf_values": self.tfidf_values}

def readFromFile(filename) -> List[TFIDFResult]:
    results = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            result = TFIDFResult(document_name=item.get("document_name", ""))
            result.tfidf_values = item.get("tfidf_values", [])
            results.append(result)
    return results

def search(search_info, tf_idfs):
    search_words = search_info.split()
    res = []

    for tfidf in tf_idfs:
        temp_sum = 0.0
        for word in search_words:
            for mp in tfidf.tfidf_values:
                temp_sum += mp.get(word, 0)

        res.append(SearchResult(document_name=tfidf.document_name, relevance_score=temp_sum))

    # Сортировка результатов по убыванию
    res.sort(key=lambda x: x.relevance_score, reverse=True)

    return res


def main():
    docs = readFromFile("../tf-idf/docs-tf-idf.json")
    query = 'уравнение'
    result = search(query, docs)
    for res in result:
        print(f"Документ {res.document_name} подходить с релевантностью: {res.relevance_score}")


if __name__ == "__main__":
    main()