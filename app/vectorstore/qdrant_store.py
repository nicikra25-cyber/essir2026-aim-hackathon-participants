"""A thin wrapper over qdrant-client.

Enough to store chunks and search them. The normal document collection and the
Level-3 table collection are kept physically separate so table extraction cannot
affect Level-1 or Level-2 retrieval.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient, models

from ..config import get_settings


class VectorStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    # --- inspection ---------------------------------------------------------
    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.get_collections().collections]

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0

        return self.client.count(
            collection_name=self.collection,
            exact=True,
        ).count

    # --- write --------------------------------------------------------------
    def ensure_collection(self, dim: int, reset: bool = False) -> None:
        """Create the collection sized to the embedding dimension.

        The vector size is fixed at creation, so if you change embedding models you
        must re-ingest or ingest into a differently named collection.
        """
        if reset and self.exists():
            self.client.delete_collection(self.collection)

        if not self.exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(self, points: list[models.PointStruct]) -> None:
        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    # --- read ---------------------------------------------------------------
    def search(
        self,
        vector: list[float],
        top_k: int,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )

        return response.points


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Return the normal PyMuPDF-backed collection used by Levels 1 and 2."""
    settings = get_settings()

    return VectorStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
    )


@lru_cache(maxsize=1)
def get_table_store() -> VectorStore:
    """Return the separate pdfplumber table collection used only by Level 3.

    This function is not called by normal ingestion or by the existing
    Level-1/2 `retrieve()` function.
    """
    settings = get_settings()

    return VectorStore(
        url=settings.qdrant_url,
        collection=f"{settings.qdrant_collection}_tables",
    )