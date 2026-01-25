# Install dependencies first (if not installed)
# pip install openl3 librosa numpy faiss-cpu soundfile

import torchopenl3 as openl3
import librosa
import numpy as np
import faiss
import os
import torch
import argparse

# ---------------------------
# Step 1: Audio Preprocessing & Embedding
# ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
def load_audio(file_path, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    return y

def extract_embedding(file_path, sr=48000,device="gpu"):
    """
    Extract OpenL3 embedding from audio file
    """
    audio, sr = librosa.core.load(file_path, sr=sr, mono=True)
    model = openl3.OpenL3Embedding(input_repr='mel128', 
                                        embedding_size=512, 
                                        content_type='music')
    audio = np.reshape(audio, (1, -1))  # reshape to (channels, samples)
    embedding = openl3.embed(model=model, 
                                audio=audio, # shape sould be (channels, samples)
                                sample_rate=sr, # sample rate of input file
                                hop_size=1, 
                                device=DEVICE) # use gpu?
    return embedding

def get_cache_embedding(file_path, cache_dir="data/embeddings_cache"):
    """
    Get cached embedding or compute and cache it
    """
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.basename(file_path)
    cache_path = os.path.join(cache_dir, base_name + ".npy")
    if os.path.exists(cache_path):
        embedding = np.load(cache_path)
    else:
        print(f"Extracting embedding for {file_path}")
        embedding = extract_embedding(file_path)
        np.save(cache_path, embedding)
        print(f"Saved embedding to {cache_path}")
    return embedding

def build_embeddings_database(music_files):
    embeddings = []
    file_paths = []
    for f in music_files:
        emb = get_cache_embedding(f)
        embeddings.append(emb)
        file_paths.append(f)
    embeddings = np.stack(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance
    index.add(embeddings)
    print(f"Index built with {index.ntotal} tracks")
    faiss.write_index(index, "data/index/index.faiss")
    #record file paths
    with open("data/file_name/file_paths.txt", "w") as f:
        for path in file_paths:
            f.write(f"{path}\n")
    print("File paths saved.")



def recommend(query_file, top_k=5):
    print(f"Extracting embedding for query: {query_file}")
    query_emb = extract_embedding(query_file)
    query_emb = query_emb.astype("float32")
    index = faiss.read_index("data/index/index.faiss")
    #read file paths
    with open("data/file_name/file_paths.txt", "r") as f:
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
        build_embeddings_database(music_files)
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

