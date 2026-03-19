# Install dependencies first (if not installed)
# pip install openl3 librosa numpy faiss-cpu soundfile

import sys
import optparse
import argparse
import faiss
import os 
import numpy as np
from pathlib import Path
import music_processing as mp

def recommend(query_file, top_k=5):
    print(f"Extracting embedding for query: {query_file}")
    query_emb = mp.extract_embedding(query_file)
    query_emb = query_emb.astype("float32")
    index = faiss.read_index("data/index/index.faiss")
    #read file paths
    with open("data/file_name/file_paths.txt", "r",encoding="utf-8") as f:
        file_paths = [line.strip() for line in f.readlines()]
    D, I = index.search(np.expand_dims(query_emb, axis=0), top_k)
    recommendations = [file_paths[i] for i in I[0]]
    distances = D[0]
    return list(zip(recommendations, distances))



# Example: load all music in a folder

def main():
    parser = argparse.ArgumentParser(description="Convert all videos in a folder to MP3.")
    parser.add_argument("-b", "--build", help="Build embeddings database from music folder")
    parser.add_argument("-r", "--recommend", help="recommend similar tracks for the given audio file")
    parser.add_argument("--random", help="give random recommendations", action="store_true")
    args = parser.parse_args()
    if args.build:
        music_folder = "data/music_db"
        music_files = [os.path.join(music_folder, f) for f in os.listdir(music_folder) if f.endswith(".mp3")]
        mp.build_embeddings_database(music_files)
    if args.recommend:
        path_head = "data/input"
        results = recommend(os.path.join(path_head, args.recommend))
        print(f"Top Recommendations:{results}")
    if args.random:
        # Load file paths
        with open("data/file_name/file_paths.txt", "r") as f:
            file_paths = [line.strip() for line in f.readlines()]
        random_files = np.random.choice(file_paths, size=5, replace=False)
        print("Random Recommendations:")
        for f in random_files:
            print(f)

if __name__ == "__main__":
    main()

