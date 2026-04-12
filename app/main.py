from app.init_model import model_creation
from config import settings

if __name__ == "__main__":
    params = {
        'learning_rate': 1e-4,
        'num_epochs': 5,
        'batch_size': 128,
        'temperature': 100,
        'unfreeze_layers': {
            'audio_projection': True,
            'audio_transform': True,
            'text_projection': True,
            'text_transform': True
        },
        'save_path': str(settings.BEST_MODEL_FILE),
        'early_stopping': {
            'enabled': True,
            'metric': 'similarity',
            'mode': 'max'
        }
    }
    model_creation(params=params)