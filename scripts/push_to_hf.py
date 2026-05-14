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

    Uploads one file at a time with retry logic. Re-running is safe — files
    already present on HF are skipped so progress is never lost on connection
    drops (WinError 10053 / timeout).
    """
    import time
    from huggingface_hub import HfApi

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARN] No PDFs found in {pdf_dir}")
        return

    total_mb = sum(p.stat().st_size for p in pdfs) / (2 ** 20)
    print("-" * 60)
    print(f"  Repo   : {repo_id}")
    print(f"  Source : {pdf_dir}")
    print(f"  Files  : {len(pdfs)}  ({total_mb:.1f} MB)")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True, private=True)

    # Build set of filenames already on HF so we can skip them
    print("  Checking existing files on HF …")
    try:
        existing = {
            Path(f.rfilename).name
            for f in api.list_repo_tree(
                repo_id=repo_id, repo_type='dataset', path_in_repo='pdfs', recursive=False,
            )
        }
    except Exception:
        existing = set()
    print(f"  Already uploaded: {len(existing)} / {len(pdfs)}")

    uploaded = skipped = failed = 0
    for i, pdf in enumerate(pdfs, 1):
        if pdf.name in existing:
            skipped += 1
            continue

        retries = 3
        for attempt in range(1, retries + 1):
            try:
                api.upload_file(
                    path_or_fileobj=str(pdf),
                    path_in_repo=f"pdfs/{pdf.name}",
                    repo_id=repo_id,
                    repo_type='dataset',
                    commit_message=f"Add {pdf.name}",
                )
                uploaded += 1
                print(f"  [{i}/{len(pdfs)}] ✓ {pdf.name}")
                break
            except Exception as exc:
                if attempt == retries:
                    failed += 1
                    print(f"  [{i}/{len(pdfs)}] ✗ {pdf.name} — FAILED after {retries} attempts: {exc}")
                else:
                    wait = 5 * attempt
                    print(f"  [{i}/{len(pdfs)}] retry {attempt}/{retries} for {pdf.name} (wait {wait}s): {exc}")
                    time.sleep(wait)

    print("-" * 60)
    print(f"  Uploaded: {uploaded}  Skipped: {skipped}  Failed: {failed}")
    if failed:
        print("  Re-run the script to retry failed files.")
    else:
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
