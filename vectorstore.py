import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore():
    def __init__(self, documents, model):
        self.documents = documents
        self.vectors = {}
        self.embeddings_model = model
        self.build()

    def build(self):
        vectors = self.embeddings_model.encode(self.documents)
        for i, doc in enumerate(self.documents):
            self.vectors[doc] = vectors[i]

    def cosine_similarity(self, u, v):
        dot_product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return dot_product / (norm_u * norm_v)

    def get_top_n(self, query, n=5):
        scores = {}
        for key in self.vectors:
            embedded_query = self.embeddings_model.encode([query])[0]
            scores[key] = self.cosine_similarity(embedded_query, self.vectors[key])
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]


docs = ["I like apples", "I like pears", "I like dogs", "I like cats"]
model = SentenceTransformer("all-MiniLM-L6-v2")
vs = VectorStore(docs, model)
print(vs.get_top_n("I like apples", n=1))
print(vs.get_top_n("fruit", n=2))
