import os

from sentence_transformers import SentenceTransformer, util
import torch

from sbert.main import model


class Document:
    def __init__(self, name, text):
        self.name = name
        self.text = text


def load_embeddings(path):
    embeddings = torch.load(path, weights_only=False)
    return embeddings


def getDocs(pdf_folder):
    results = []
    for filename in os.listdir(pdf_folder):
        results.append(Document(filename, filename))
    return results


def search(query, document_embeddings):
    # Запрос пользователя
    query_embedding = model.encode([query])[0]

    # Преобразуем в тензоры
    document_embeddings_tensor = torch.tensor(document_embeddings)
    query_embedding_tensor = torch.tensor(query_embedding)

    # Рассчитываем косинусное сходство
    cosine_scores = util.cos_sim(query_embedding_tensor, document_embeddings_tensor)
    # Получаем результаты
    docs = getDocs("C:/Users/ICSS2location10/Desktop/diplom-pract/pre-datasets")
    scores = cosine_scores[0].tolist()
    sorted_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # Выводим результаты
    print("Запрос:", query)
    for doc, score in sorted_results:
        print(f"Сходство: {score:.4f}, Документ: {doc.name}")


def main():
    query = 'some query'
    embeddings = load_embeddings("C:/Users/ICSS2location10/Desktop/diplom-pract/sbert/embeddings.pt")
    search(query, embeddings)


if  __name__ == "__main__":
    main()