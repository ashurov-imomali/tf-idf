from sentence_transformers import SentenceTransformer, util
import torch

# Загружаем предобученную модель (мультиязычную)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Список документов
documents = [
    "Кот мурлычет на подоконнике.",
    "Собака лает и бегает по двору.",
    "Кот и собака играют вместе."
]

# Генерация эмбеддингов для всех документов
document_embeddings = model.encode(documents)

# Запрос пользователя
query = "Котенок спит на подоконнике."
query_embedding = model.encode([query])[0]

# Рассчитываем косинусное сходство
document_embeddings_tensor = torch.tensor(document_embeddings)
query_embedding_tensor = torch.tensor(query_embedding)

# Считаем сходство запроса с каждым документом
cosine_scores = util.cos_sim(query_embedding_tensor, document_embeddings_tensor)

scores = cosine_scores[0].tolist()
sorted_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

print("Запрос:", query)
for doc, score in sorted_results:
    print(f"Сходство: {score:.4f}, Документ: {doc}")

