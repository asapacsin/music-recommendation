import os

def load_music_files(folder):
    """
    Load music files under given folder.

    Parameters
    ----------
    folder : str
        Path to the folder containing music files.

    Returns
    -------
    files : list
        List of music file paths.
    """
    files = []
    for root, dirs, filenames in os.walk(folder):
        for f in filenames:
            if f.endswith(".mp3"):
                files.append(os.path.join(root, f))
    return files

def process_files():
    folder = "data/music_pre"
    files = load_music_files(folder)
