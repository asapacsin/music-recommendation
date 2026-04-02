from app.init_model import model_creation

if __name__ == "__main__":
    # 模型训练配置参数
    params = {
        'learning_rate': 1e-4,
        'num_epochs': 5,
        'batch_size': 128,
        'temperature': 100,  # 对比学习温度系数
        'unfreeze_layers': {
            'audio_projection': True,
            'audio_transform': True,
            'text_projection': True,
            'text_transform': True
        },
        'save_path': "../model/best_model.pt",
        'early_stopping': {
            'enabled': True,
            'metric': 'similarity',
            'mode': 'max'
        }
    }
    model_creation(params = params)