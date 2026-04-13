import shutil
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import PDF_DIR


def archive_and_clean_pdfs(target_path: str):
    pdf_dir = Path(target_path)
    zip_destination = pdf_dir.parent / "pdfs_archive"
    
    if not pdf_dir.exists():
        print(f"Error: Path {pdf_dir} does not exist.")
        return

    try:
        print(f"Archiving {len(list(pdf_dir.glob('*')))} files...")


        archive_path = shutil.make_archive(
            base_name=str(zip_destination), 
            format='zip', 
            root_dir=str(pdf_dir)
        )
        print(f"[[INFO-PDF-Archive] Successfully archived PDFs: {archive_path}")

        if os.path.exists(archive_path):
            print("Cleaning up original pdfs...")
            count = 0
            for item in pdf_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        count += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    print(f"Could not delete {item.name}: {e}")
            
            print(f"Cleanup complete. Deleted {count} files.")
            final_path = shutil.move(archive_path, pdf_dir / Path(archive_path).name)
            print(f"Moved archive to: {final_path}")
        else:
            print("Abort: Archive file not detected. Original files preserved.")

    except Exception as e:
        print(f"An error occurred during the process: {e}")

if __name__ == "__main__":
    PDF_PATH = PDF_DIR
    archive_and_clean_pdfs(PDF_PATH)
