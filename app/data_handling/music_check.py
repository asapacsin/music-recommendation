import os
import librosa
from pathlib import Path

def check_music_files(directory_path):
    """
    Load every audio file in the given directory and print files that fail to load.
    
    Args:
        directory_path: Path to the directory containing audio files
    """
    failed_files_path = []
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return
    count = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if os.path.isfile(file_path):
            try:
                librosa.load(file_path)
            except Exception as e:
                failed_files_path.append(file_path)
                print(f"✗ Failed to load: {filename} - {type(e).__name__}")
        count += 1
        if count %100 == 0:
            print(f"Checked {count} files...")
    
    if failed_files_path:
        print(f"\n--- Summary: {len(failed_files_path)} file(s) failed to load ---")
        for file_path in failed_files_path:
            print(f"  {os.path.basename(file_path)}")
    else:
        print("\nAll files loaded successfully!")
    return failed_files_path
    
def remove_files(file_paths):
    """
    Remove all files in the given list.
    
    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"✓ Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"✗ Failed to remove: {os.path.basename(file_path)} - {type(e).__name__}")

if __name__ == "__main__":
    directory = "data\\music_db"
    failed_files_path =check_music_files(directory)
    if failed_files_path:
        remove_files(failed_files_path)
    
