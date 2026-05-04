from __future__ import annotations
import pandas as pd
import numpy as np 
import sys
import uuid
import argparse
from tqdm import tqdm
from pathlib import Path 
from qdrant_client.models import PointStruct

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.ingestion.collection import (
    COLLECTIONS,
    verify_collections,
    create_collections,
    get_client
)

from config.settings import QDRANT_HOST, QDRANT_PORT, HF_DOWNLOAD_PATH
UPSERT_BATCH = 256
_NS = uuid.NAMESPACE_DNS
EMBEDDING_DIR = HF_DOWNLOAD_PATH / 'k2p' / 'gym-rag-embeddings'


def _load_gym_images(data_dir: Path) -> list[PointStruct]:
    vec_dir = data_dir / 'vectors'
    files = sorted(vec_dir.glob('athlete_*.npy'))
    print(f"[INFO_QDRANT] - Found {len(files)} Vector Image Embeddings")

    points: list[PointStruct] = []
    for npy_file in tqdm(files, desc = 'gym_images'):
        data = np.load(npy_file, allow_pickle = True).item()
        for page_key, vec, payload in zip(
            data['page_keys'],
            data['vectors'],
            data['payloads']
        ):
            points.append(PointStruct(
                id = str(uuid.uuid5(_NS, page_key)),
                vector = vec.tolist(),
                payload = payload
            ))

    return points



def _load_gym_text(data_dir: Path) -> list[PointStruct]:
    text_dir = data_dir / 'text_vectors'
    files = sorted(text_dir.glob('*_text.npy'))
    print(f"[INFO_QDRANT] - Found {len(files)} Vector Text Embeddings")

    points: list[PointStruct] = []
    for npy_file in tqdm(files, desc = 'gym_text'):
        data = np.load(npy_file, allow_pickle = True).item()
        for chunk_key, vec, payload in zip(
            data['chunk_keys'],
            data['vectors'],
            data['payloads']
        ):
            points.append(PointStruct(
                id = str(uuid.uuid5(_NS, chunk_key)),
                vector = vec.tolist(),
                payload = payload
            ))

    return points



def _load_gym_tables(data_dir: Path) -> list[PointStruct]:
    table_dir = data_dir / 'table_vectors'
    files = sorted(table_dir.glob('athlete_*.npy'))
    print(f"[INFO_QDRANT] - Found {len(files)} Vector Table Embeddings")

    points: list[PointStruct] = []
    for npy_file in tqdm(files, desc = 'gym_tables'):
        data = np.load(npy_file, allow_pickle = True).item()
        for key, vec, payload in zip(
            data['keys'],
            data['vectors'],
            data['payloads']
        ):
            points.append(PointStruct(
                id = str(uuid.uuid5(_NS, key)),
                vector = vec.tolist(),
                payload = payload
            ))

    return points


def _upsert_collection(
        client,
        collection: str,
        points: list[PointStruct]
) -> None:
    # print(len(points))
    for i in tqdm(range(0, len(points), UPSERT_BATCH), desc = f"--> Upserting {collection}"):
        batch = points[i : i + UPSERT_BATCH]
        client.upsert(collection_name = collection, points = batch)

    info = client.get_collection(collection)
    final = (
        getattr(info, "points_count", None)
        or getattr(info, "vectors_count", None)
        or 0
    )
    print(f"[INFO-QDRANT] - Collection {collection} Uploaded | {final}")




def load_to_Qdrant(
        COLLECTIONS: list[str],
        qdrant_host: int | None,
        qdrant_port: int | None
) -> None:
    
    separator = "=" * 50
    print(f"[INFO-QDRANT] - Connecting to QDRANT...")
    client = get_client(host=qdrant_host, port=qdrant_port)
    print(f"[INFO-QDRANT] - Connected to QDRANT")
    print(f"\n{separator}\n")

    print(f"[INFO-QDRANT] - Creating Collections...")
    create_collections(client, recreate = True)
    print(f"[INFO-QDRANT] - Collections Created")
    print(f"\n{separator}\n")



    for collection in COLLECTIONS:
        print(f"[INFO-QDRANT] - Upserting Collection: {collection}")
        if collection == 'gym_images':
            points = _load_gym_images(EMBEDDING_DIR)
            print(f"[INFO-QDRANT] - Completed Converting {collection} Embeddings\n")
            print("="*50)
        
        elif collection == 'gym_text':
            points = _load_gym_text(EMBEDDING_DIR)
            print(f"[INFO-QDRANT] - Completed Converting {collection} Embeddings\n")
            print("="*50)
        
        else:
            points = _load_gym_tables(EMBEDDING_DIR)
            print(f"[INFO-QDRANT] - Completed Converting {collection} Embeddings\n")
            print("="*50)

        _upsert_collection(
            client,
            collection,
            points
        )
        print(f"\n{separator}\n")

def main() -> None:
    load_to_Qdrant(
        COLLECTIONS = COLLECTIONS,
        qdrant_host = QDRANT_HOST,
        qdrant_port = QDRANT_PORT
    )

if __name__ == '__main__':
    main()