import argparse
import sys
from pathlib import Path
import numpy as np

#!/usr/bin/env python3
"""
read_embedding.py

Read a .npy embedding file and print basic info and a small preview.
Usage:
    python read_embedding.py path/to/embeddings.npy [--preview N]
"""



def load_embedding(path: Path, allow_pickle: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return np.load(path, allow_pickle=allow_pickle)


def summarize(arr: np.ndarray, preview: int = 5) -> None:
    print(f"Embedding file: {arr}")
    print(f"shape: {arr.shape}")


def main():
    p = argparse.ArgumentParser(description="Read and summarize a .npy embedding file")
    p.add_argument("file", type=Path, help=".npy file to read")
    p.add_argument("--preview", "-p", type=int, default=5, help="number of items/vectors to preview")
    p.add_argument("--allow-pickle", action="store_true", help="allow loading pickled objects (unsafe)")
    args = p.parse_args()

    try:
        arr = load_embedding(args.file, allow_pickle=args.allow_pickle)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(2)

    summarize(arr, preview=args.preview)


if __name__ == "__main__":
    main()

#use case
#python read_embedding.py data/embeddings_cache/file_name.npy --preview 10