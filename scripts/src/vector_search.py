from typing import Any
from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct, ScoredPoint
from openai import OpenAI


class OpenAIEmbedder:
    def __init__(self, embedding_model="text-embedding-ada-002", batch_size=10):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.client = OpenAI()

    @property
    def embedding_size(self) -> int:
        return len(self(["foo"])[0])

    def batch_embed_openai(
        self,
        docs_to_embed: list[str],
        batch_size: int = 10,
        embedding_model="text-embedding-ada-002",
    ) -> list[list[float]]:
        embeddings = []
        for batch_start in range(0, len(docs_to_embed), batch_size):
            batch_end = batch_start + batch_size
            batch = docs_to_embed[batch_start:batch_end]
            response = self.client.embeddings.create(
                model=embedding_model,
                input=batch,
            )
            # response = openai.Embedding.create(model=embedding_model, input=batch)
            for i, be in enumerate(response.data):  # type: ignore
                assert (
                    i == be.index
                )  # double check embeddings are in same order as input
            batch_embeddings = [e.embedding for e in response.data]  # type: ignore
            embeddings.extend(batch_embeddings)
        return embeddings

    def __call__(self, docs_to_embed: list[str]) -> list[list[float]]:
        return self.batch_embed_openai(
            docs_to_embed,
            batch_size=self.batch_size,
            embedding_model=self.embedding_model,
        )


class QDrantVectorDatabase:
    def __init__(self, vector_database_url: str = ":memory:"):
        self.vector_database_url = vector_database_url
        self.client = QdrantClient(self.vector_database_url)
        self.collection_names: set[str] = set()
        self.embedder = OpenAIEmbedder()

    def create_collection(self, collection_name: str) -> bool:
        success = self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        return success

    def insert(self, docs: list[dict[str, Any]], collection_name: str = "default"):
        if collection_name not in self.collection_names:
            if self.create_collection(collection_name):
                self.collection_names.add(collection_name)
            else:
                raise RuntimeError(f"Could not create collection {collection_name}")

        embeddings = self.embedder([doc["key"] for doc in docs])
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(id=idx, vector=embedding, payload=payload)
                for idx, (embedding, payload) in enumerate(zip(embeddings, docs))
            ],
        )
        return operation_info

    def search(
        self,
        query: str,
        collection_name: str = "default",
        limit: int = 10,
    ) -> list[dict]:
        embeddings = self.embedder([query])
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=embeddings[0],
            limit=limit,
        )
        return [_.payload for _ in search_result]  # type: ignore
