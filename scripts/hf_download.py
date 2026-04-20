from __future__ import annotations
import argparse
import sys
import os
import zipfile
from pathlib import Path
import tomllib
from huggingface_hub import snapshot_download
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import HF_DOWNLOAD_PATH


def _load_repo_ids() -> tuple[str, str]:
    config_path = ROOT / 'config' / 'hf_config.toml'
    with open(config_path, 'rb') as f:
        cfg = tomllib.load(f)

    return (cfg['repos']['dataset_repo'] ,
    cfg['repos']['embeddings_repo'])


def _get_token() -> str | None:
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    print(token)
    return token


def _hf_download(
        repo_id: str,
        token: str | None,
        ):
    
    snapshot_download(
    repo_id = repo_id,
    repo_type = "dataset",
    local_dir = HF_DOWNLOAD_PATH / repo_id,
    token = token,
    max_workers = 4,    
                          )
    # print(f"[INFO-HF_PULL] Pulling {repo_id} Dataset from HF")

def _unzip_embeddings(zip_path: Path, target_path: Path):
    target_path.mkdir(parents = True, exist_ok = True)
    with zipfile.ZipFile(zip_path, 'r') as f:
        files = f.namelist()
        for file in files:
            f.extract(file, target_path)
    
    print(f"[INFO-EMBEDDINGS - Extracted {len(files)} | from {zip_path}")


def main() -> None:
    dataset, embeddings = _load_repo_ids()

    print(f"[INFO-HF-DATASET] - Downloading Dataset from {dataset} repo on HF")
    print(f"[INFO-HF-DATASET] - Saving to {HF_DOWNLOAD_PATH} / {dataset}")

    print(f"[INFO-HF-EMBEDDINGS] - Downloading Dataset from {embeddings} repo on HF")
    print(f"[INFO-HF-EMBEDDINGS] - Saving to {HF_DOWNLOAD_PATH} | {embeddings} repo on HF")
    _hf_download(embeddings, token = _get_token())
    print(f"[INFO-HF-EMBEDDINGS] - Download Complete!")

    print(f"Extracting zip files...")
    extraction_directory = ['table_vectors', 'text_vectors', 'vectors']
    for ed in extraction_directory:
        print(f"[INFO-Extraction] {ed} ..." )
        extract_path = HF_DOWNLOAD_PATH / embeddings / ed
        zip_path = extract_path / f'{ed}.zip'
        _unzip_embeddings(zip_path, extract_path)


        if zip_path.exists():
            zip_path.unlink()
            print(f"[INFO-Cleanup] Removed {zip_path.name}")
        print("="*50)


if __name__ == "__main__":
    main()
