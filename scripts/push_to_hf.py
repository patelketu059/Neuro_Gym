from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATA_DIR, 
    OUT_DIR,
    PDF_DIR,
    SESSIONS_PATH, 
    BLOCK_SUMMARY_PATH
)


def _load_config(path: Path) -> dict:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)
    

def _get_token(cfg: dict) -> str:
    env_var = cfg['auth']['token_env_var']
    token = os.environ.get(env_var)

    if token: return token
    return "NO TOKEN FOUND!"

def _push(folder: Path, 
          repo_id: str, 
          token: str, 
          commit_msg: str):
    
    files = sorted(folder.rglob("*"))
    files = [f for f in files if f.is_file()]
    total = sum(f.stat().st_size for f in files) / (2**20)

    print("-" * 60)
    print(f"  Repo:   {repo_id}")
    print(f"  Source: {folder}")
    print(f"  Files:  {len(files)}  ({total:.1f} MB)")
    # print(token)

    try: 
        from huggingface_hub import HfApi
    except ImportError:
        print("[HF_ERROR] huggingface_hub not installed.")
        sys.exit(1)

    api = HfApi(token = token)

    api.create_repo(repo_id = repo_id,
                    repo_type = 'dataset',
                    exist_ok = True,
                    private = True)
    

    print(f"Starting HF upload...")
    api.upload_folder(
        folder_path = str(folder),
        repo_id = repo_id,
        repo_type = 'dataset',
        delete_patterns = "*"
        )

    print(f"Uploaded to: https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Push gym-rag data to HF",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--repo", choices = ['dataset', 'embeddings', 'all'], default = 'dataset')
    parser.add_argument('--config', default = 'config/hf_config.toml')
    args = parser.parse_args()

    cfg = _load_config(ROOT / args.config)
    token = _get_token(cfg)

    if not token:
        print(f"[TOKEN_ERROR] HF TOKEN MISSING")
        sys.exit()

    if args.repo in ('dataset', 'all'):
        # sessions = pd.read_csv(SESSIONS_PATH)
        # summary = pd.read_csv(BLOCK_SUMMARY_PATH)

        _push(
            folder = DATA_DIR,
            repo_id = cfg['repos']['dataset_repo'],
            token = token,
            commit_msg = cfg['upload']['commit_message_dataset'],
        )

if __name__ == "__main__":
    main()