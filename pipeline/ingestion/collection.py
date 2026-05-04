from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType
)

VECTOR_SIZE = 2048
COLLECTIONS = [
    'gym_images',
    'gym_text',
    'gym_tables'
               ]

def get_client(
        host: str = 'localhost',
        port: int = 6333,
        timeout: int = 300
) -> QdrantClient:
    client = QdrantClient(host=host, port=port, timeout=timeout)
    try:
        client.get_collections()
    except Exception as exc:
        raise ConnectionError(
            f"Cannot connect to Qdrant at {host}:{port}. "
        ) from exc
    return client

def create_collections(
        client: QdrantClient,
        recreate: bool = False
) -> None:
    
    existing = {c.name for c in client.get_collections().collections}
    
    for name in COLLECTIONS:
        if name in existing:
            if recreate:
                client.delete_collection(name)
                print(f"[INFO-QDRANT] Dropped existing collection: {name}")
            else:
                print(f"[INFO-QDRANT] Collection already exists, skipping: {name}")
                continue

        try: 
            client.create_collection(
                collection_name = name,
                vectors_config = VectorParams(
                    size = VECTOR_SIZE,
                    distance = Distance.COSINE
                )
            )

            print(f"[INFO-QDRANT] - Created Collection: {name}  |   Dim: {VECTOR_SIZE}  COSINE")

        except Exception as e:
            if 'already exists' in str(e) or '409' in str(e):
                print('COLLECTION ALREADY EXISTS!')
            else:
                raise
    _create_payload_indexes(client)


def _create_payload_indexes(client: QdrantClient) -> None:
    index_specs = {
        'gym_images' : ['athlete_id', 'training_level', 'page_number', 'pdf_path'],
        'gym_text':    ['athlete_id', 'training_level', 'pdf_path'],
        'gym_tables':  ['athlete_id', 'training_level', 'block_phase', 'week', 'pdf_path']
    }

    for collection, fields in index_specs.items():
        for field in fields:
            try:
                client.create_payload_index(
                    collection_name = collection,
                    field_name = field,
                    field_schema = PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

def verify_collections(
        client: QdrantClient
) -> dict[str, int]:
    counts = {}
    for name in COLLECTIONS:
        try:
            info = client.get_collection(name)
            count = getattr(info, "points_count", None) or getattr(info, "vectors_count", None) or 0
            counts[name] = count
        except Exception:
            counts[name] = -1
    return counts