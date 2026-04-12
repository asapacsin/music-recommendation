import os
import numpy as np
from app.recommend import load_audio
from config import settings

def test_load_audio():
    # Use a small test audio file (add one to tests/data/)
    test_file = settings.TEST_DATA_DIR / "【殺戮の天使】 彼岸 BGM.mp3"
    assert test_file.exists(), "Test audio file missing"
    
    audio = load_audio(str(test_file))
    assert isinstance(audio, np.ndarray), "Output should be numpy array"
    assert len(audio) > 0, "Audio should not be empty"