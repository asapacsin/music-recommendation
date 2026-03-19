import torchopenl3 as openl3
import librosa
import faiss
import numpy as np
import os
import torch
import time

# ---------------------------
# Step 1: Audio Preprocessing & Embedding
# ---------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

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
        start_time = time.time()
        embedding = extract_embedding(file_path)
        end_time = time.time()
        print(f"Extraction took {end_time - start_time:.2f} seconds")
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