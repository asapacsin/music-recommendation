import os
import numpy as np
from app.recommend import load_audio

def test_load_audio():
    # Use a small test audio file (add one to tests/data/)
    test_file = "tests/data/【殺戮の天使】 彼岸 BGM.mp3"
    assert os.path.exists(test_file), "Test audio file missing"
    
    audio = load_audio(test_file)
    assert isinstance(audio, np.ndarray), "Output should be numpy array"
    assert len(audio) > 0, "Audio should not be empty"