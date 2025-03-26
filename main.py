import math
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

class Document:
    def __init__(self, name, text):
        self.name = name
        self.text = text

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
    for document in documents:
        words_in_document = set()
        for token in document.text.split():
            normalized_word = morph.parse(token)[0].normal_form
            if normalized_word not in words_in_document:
                word_document_count[normalized_word] = word_document_count.get(normalized_word, 0) + 1
                words_in_document.add(normalized_word)
    return word_document_count

class DocumentTermFrequency:
    def __init__(self, document_name):
        self.document_name = document_name
        self.term_frequencies = {}

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
    return term_frequencies_list

def calculate_inverse_document_frequencies(documents):
    word_document_count = count_documents_containing_words(documents)
    total_documents = len(documents)
    inverse_document_frequencies = {}

    for word, document_count in word_document_count.items():
        inverse_document_frequencies[word] = math.log2(total_documents / document_count)

    return inverse_document_frequencies

class TFIDFResult:
    def __init__(self, document_name):
        self.document_name = document_name
        self.tfidf_values = []

class SearchResult:
    def __init__(self, document_name, relevance_score):
        self.document_name = document_name
        self.relevance_score = relevance_score

def perform_search(query, tfidf_results):
    query_words = query.split()
    print("Search terms:", query_words)
    results = []

    for tfidf in tfidf_results:
        for query_word in query_words:
            for tfidf_entry in tfidf.tfidf_values:
                if tfidf_entry.get(query_word, 0) > 0:
                    print(tfidf_entry.get(query_word, 0))

    return results

def main():
    documents = load_sample_documents_ru()

    term_frequencies = calculate_term_frequencies(documents)
    inverse_document_frequencies = calculate_inverse_document_frequencies(documents)

    tfidf_results = []
    for term_frequency in term_frequencies:
        tfidf_result = TFIDFResult(term_frequency.document_name)
        for word, tf_value in term_frequency.term_frequencies.items():
            tfidf_result.tfidf_values.append({word: tf_value * inverse_document_frequencies[word]})
        tfidf_results.append(tfidf_result)

    for item in tfidf_results:
        print(f"Document: {item.document_name}")
        print("TF-IDF: ", item.tfidf_values)
        print("----------")

    search_query = 'здоровья яблоко'
    search_results = perform_search(search_query, tfidf_results)

    for item in search_results:
        print(f"Document: {item.document_name}, Relevance Score: {item.relevance_score}")

if __name__ == "__main__":
    main()