from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def _load_env() -> None:
    env_path = ROOT / '.env'
    if env_path.is_file():
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)

_load_env()

from config.settings import DATA_DIR, PDF_DIR


def _load_config(path: Path) -> dict:
    import tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _get_token(cfg: dict) -> str:
    env_var = cfg['auth']['token_env_var']
    return os.environ.get(env_var, "NO TOKEN FOUND!")


def _push_folder(folder: Path, repo_id: str, token: str, path_in_repo: str = "") -> None:
    """Upload an entire folder to a HF dataset repo."""
    from huggingface_hub import HfApi

    files = sorted(folder.rglob("*"))
    files = [f for f in files if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / (2 ** 20)

    print("-" * 60)
    print(f"  Repo        : {repo_id}")
    print(f"  Source      : {folder}")
    print(f"  Destination : {path_in_repo or '(root)'}")
    print(f"  Files       : {len(files)}  ({total_mb:.1f} MB)")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True, private=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type='dataset',
        path_in_repo=path_in_repo,
    )
    print(f"  Done → https://huggingface.co/datasets/{repo_id}")


def _push_pdfs_individual(pdf_dir: Path, repo_id: str, token: str) -> None:
    """Upload each PDF individually to pdfs/<filename> in the dataset repo.

    Individual files allow on-demand per-athlete downloads at inference time —
    avoiding the need to fetch and extract a large zip archive on every request.
    """
    from huggingface_hub import HfApi

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No PDFs found in {pdf_dir}")
        return

    total_mb = sum(p.stat().st_size for p in pdfs) / (2 ** 20)
    print("-" * 60)
    print(f"  Repo        : {repo_id}")
    print(f"  Source      : {pdf_dir}")
    print(f"  Destination : pdfs/")
    print(f"  Files       : {len(pdfs)}  ({total_mb:.1f} MB)")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True, private=True)

    # upload_folder with path_in_repo="pdfs" pushes the folder contents under pdfs/
    api.upload_folder(
        folder_path=str(pdf_dir),
        repo_id=repo_id,
        repo_type='dataset',
        path_in_repo='pdfs',
        commit_message=f"Upload {len(pdfs)} individual PDFs to pdfs/",
    )
    print(f"  Done → https://huggingface.co/datasets/{repo_id}/tree/main/pdfs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push gym-rag data to HuggingFace Hub",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        choices=['dataset', 'pdfs', 'embeddings', 'all'],
        default='dataset',
        help=(
            "dataset = push DATA_DIR (sessions, block_summary, etc.) | "
            "pdfs    = push individual PDFs to pdfs/ | "
            "all     = dataset + pdfs"
        ),
    )
    parser.add_argument('--config', default='config/hf_config.toml')
    args = parser.parse_args()

    cfg   = _load_config(ROOT / args.config)
    token = _get_token(cfg)

    if not token or token == "NO TOKEN FOUND!":
        print("[TOKEN_ERROR] HF TOKEN MISSING")
        sys.exit(1)

    dataset_repo = cfg['repos']['dataset_repo']

    if args.repo in ('dataset', 'all'):
        _push_folder(
            folder      = DATA_DIR,
            repo_id     = dataset_repo,
            token       = token,
            path_in_repo= "",
        )

    if args.repo in ('pdfs', 'all'):
        _push_pdfs_individual(
            pdf_dir  = PDF_DIR,
            repo_id  = dataset_repo,
            token    = token,
        )


if __name__ == "__main__":
    main()
