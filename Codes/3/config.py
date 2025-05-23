import warnings

warnings.filterwarnings('ignore')

DATASET_CONFIG = {
    'file_path': 'Graph\paysim_full.csv',
    'sample_size': 50000,
    'random_state': 42,
    'balance_ratio': 5.0  
}

MODEL_CONFIG = {
    'gcn': {
        'hidden_channels': 32,
        'dropout': 0.3,
        'learning_rate': 0.01,
        'weight_decay': 5e-4
    },
    'gat': {
        'hidden_channels': 16,
        'heads': 2,
        'dropout': 0.3,
        'learning_rate': 0.01,
        'weight_decay': 5e-4
    }
}

TRAINING_CONFIG = {
    'max_epochs': 100,
    'patience': 10,
    'use_focal_loss': True,
    'focal_loss_alpha': 0.25,
    'focal_loss_gamma': 2.0,
    'train_split': 0.7,
    'validation_frequency': 5
}

VIZ_CONFIG = {
    'figure_size': (10, 6),
    'style': 'whitegrid',
    'save_plots': True,
    'plot_dir': 'plots/'
}

FILE_PATHS = {
    'models_dir': 'models/',
    'plots_dir': 'plots/',
    'results_dir': 'results/'
}