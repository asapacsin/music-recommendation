import clip
from mutagen.easyid3 import EasyID3
import transformers
import torch


def extract_text_embedding(text, model_name='ViT-H/14'):
    """
    Extracts text embedding using OpenCLIP model.

    Args:
        text (str): Text to embed
        model_name (str, optional): Model name to use. Defaults to 'ViT-H/14'.

    Returns:
        torch.FloatTensor: Embedding of text
    """
    model, preprocess = clip.load(model_name, jit=False)
    text_embedding = model.encode_text(preprocess(text))
    return text_embedding

def extract_info(music_path):
    """
    Extract topic and author name from music file metadata.
    
    Args:
        music_path (str): Path to the music file
        
    Returns:
        str: Combined string of author and topic
    """
    
    try:
        audio = EasyID3(music_path)
        topic = music_path.split('/')[-1].split('.')[0]  # Fallback to filename if metadata is missing
        author = audio.get('artist', ['Unknown'])[0]
        return f"{topic} by {author}"
    except Exception as e:
        return f"Error extracting info: {str(e)}"

def main():
    rule = """# CLAP Text Input Generation Rule (for LLM) - English Version

Purpose: Generate descriptive text for CLAP embedding input from music metadata.

Rule Template:
-----------------------
Input:
- Title: [song title]
- Artist: [artist name]

LLM Output Format (for CLAP):
"[Title] by [Artist], a [Genre] piece that is [Emotion/Mood]"

Example:
Input:
- Title: Wind Through the Town
- Artist: Yukiko Isomura

Output:
"Wind Through the Town by Yukiko Isomura, an ambient piece that is reflective and nostalgic, featuring piano and serene instrumentation."

Notes:
- Include at least Title and Artist for minimal input.
- Adding Genre, Mood, and Additional Features improves CLAP embedding matching.
- Output should be a single coherent sentence suitable as text input for CLAP.
"""
    music_path = "data\music_pre\01.【3】山下直人 - Astral Requiem.mp3"
    music_info = extract_info(music_path)
    model_id = "LLM-Research/Meta-Llama-3-8B"
    pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    pipeline("Hey how are you doing today?")

if __name__ == "__main__":
    main()