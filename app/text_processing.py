import clip
from mutagen.easyid3 import EasyID3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from utils import translator as trans

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
        topic = os.path.splitext(os.path.basename(music_path))[0]
        author = audio.get('artist', ['Unknown'])[0]
        topic = topic.replace(author, '').strip()
        describe_text = f"{topic} by {author}"
        return describe_text
    except Exception as e:
        return f"Error extracting info: {str(e)}"
    
def extract_info_list(music_paths):
    descriptions = []
    for music_path in music_paths:
        descriptions.append(extract_info(music_path))
    return descriptions
    
def get_music_name(music_file):
    """
    Extract music name from file path.

    Args:
        music_path (str): Path to the music file
        
    Returns:
        str: Music filename without extension
    """
    return music_file.rsplit('.', 1)[0]

def llm_describe(prompt,model,tokenizer,max_new_tokens=200,devices="cpu"):
    rule = """output only in "topic by author" format, only allow in English,
    filter out any irrelevant information, and do not add any additional words.
"""
    prompt = "rule:"+rule + "\nInput:" + prompt
    if devices == "cuda":
        inputs = tokenizer(prompt, return_tensors="pt").to(devices)
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_model(model_path="model/llama3", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, device_map=device)
    return model, tokenizer

def save_descriptions(descriptions, output_path="data/mapping/music_map.txt"):
    """
    Save descriptions mapping to a JSON file.

    Args:
        descriptions (dict): Dictionary mapping music names to descriptions
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

def filter_text_list(text):
    """
    Filter out numbers and symbols from text, keeping only letters and spaces.

    Args:
        text (str): Text to filter

    Returns:
        list: List with filtered text containing only letters and spaces
    """
    return ''.join(element for element in text if element.isalpha() or element.isspace())

def generate_map():
    music_dir = "data/music_db"
    music_files = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.flac', '.wav', '.m4a'))]
    descriptions = extract_info_list([os.path.join(music_dir, f) for f in music_files])
    print("finish extracting music info")
    filtered_files = [filter_text_list(desc) for desc in descriptions]
    print("finish filtering music info")
    results = trans.translate_text(filtered_files)
    print("finish translating music info")
    map = {music_file: result for music_file, result in zip(music_files, results)}
    save_descriptions(map)

def main():
    generate_map()

if __name__ == "__main__":
    main()
    
