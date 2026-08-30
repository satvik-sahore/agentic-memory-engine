import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manages Qdrant client lifecycle and collection operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        storage_path: Optional[str] = None,
        prefer_embedded: Optional[bool] = None,
        collection_name: Optional[str] = None,
    ):
        self.host = host if host is not None else settings.qdrant_host
        self.port = port if port is not None else settings.qdrant_port
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.storage_path = storage_path if storage_path is not None else settings.qdrant_storage_path
        self.prefer_embedded = (
            prefer_embedded if prefer_embedded is not None else settings.qdrant_prefer_embedded
        )
        self.collection_name = collection_name or settings.qdrant_collection_name
        self._client: Optional[QdrantClient] = None
        self._is_embedded: bool = False

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client with automatic fallback."""
        if self._client is None:
            if self.host == ":memory:" or self.storage_path == ":memory:":
                logger.info("Initializing Qdrant in transient in-memory mode (:memory:)")
                self._client = QdrantClient(":memory:")
                self._is_embedded = True
            elif self.prefer_embedded:
                logger.info(f"Initializing Qdrant in embedded local mode at '{self.storage_path}'")
                self._client = QdrantClient(path=self.storage_path)
                self._is_embedded = True
            else:
                try:
                    # Attempt connection to remote/Docker Qdrant instance
                    test_client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                        timeout=1,
                        check_compatibility=False,
                    )
                    test_client.get_collections()
                    self._client = test_client
                    self._is_embedded = False
                    logger.info(f"Connected to remote Qdrant at {self.host}:{self.port}")
                except Exception as e:
                    logger.info(
                        f"Remote Qdrant not detected at {self.host}:{self.port}. "
                        f"Using embedded local storage at '{self.storage_path}'."
                    )
                    self._client = QdrantClient(path=self.storage_path, check_compatibility=False)
                    self._is_embedded = True
        return self._client

    def is_healthy(self) -> bool:
        """Check if Qdrant instance is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed for Qdrant - {e}")
            return False

    def close(self):
        """Close Qdrant client connection cleanly."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance: rest_models.Distance = rest_models.Distance.COSINE,
        recreate_if_dim_mismatch: bool = True,
    ) -> bool:
        """
        Ensures that the vector collection exists and has the correct dimension.
        If the dimension does not match, optionally recreates it.
        """
        col_name = collection_name or self.collection_name
        dim = dimension or settings.embedding_dimension

        collections = self.client.get_collections().collections
        existing_names = [col.name for col in collections]

        if col_name in existing_names:
            col_info = self.client.get_collection(collection_name=col_name)
            current_dim = None
            vectors_config = col_info.config.params.vectors

            if isinstance(vectors_config, rest_models.VectorParams):
                current_dim = vectors_config.size
            elif isinstance(vectors_config, dict) and vectors_config:
                first_val = next(iter(vectors_config.values()))
                current_dim = getattr(first_val, "size", None)
            elif hasattr(vectors_config, "size"):
                current_dim = getattr(vectors_config, "size", None)

            if current_dim is not None and current_dim != dim:
                if recreate_if_dim_mismatch:
                    logger.warning(
                        f"Collection '{col_name}' vector dimension mismatch (existing: {current_dim}, expected: {dim}). "
                        f"Recreating collection..."
                    )
                    self.client.delete_collection(collection_name=col_name)
                else:
                    logger.warning(f"Collection '{col_name}' has dimension {current_dim} (expected {dim}).")
                    return False
            elif current_dim is not None and current_dim == dim:
                logger.info(f"Collection '{col_name}' is ready with dimension {dim}.")
                return False

        logger.info(f"Creating collection '{col_name}' with vector size {dim} and distance {distance.name}...")
        self.client.create_collection(
            collection_name=col_name,
            vectors_config=rest_models.VectorParams(
                size=dim,
                distance=distance,
            ),
        )
        # Create payload index for user_id to enable fast filtering in server mode
        if not self._is_embedded:
            self.client.create_payload_index(
                collection_name=col_name,
                field_name="user_id",
                field_schema=rest_models.PayloadSchemaType.KEYWORD,
            )
        logger.info(f"Collection '{col_name}' created successfully.")
        return True


# Singleton instance
qdrant_manager = QdrantManager()
