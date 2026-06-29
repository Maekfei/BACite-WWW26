"""
Configuration file for the two-stage HGT citation prediction model.
Supports tuning of key parameters and simplifies configuration management.
"""

import os
import torch

# Data root: set the DATA_DIR env var to point at your data location.
# Defaults to "." so paths resolve relative to where you run the script
# (e.g. the repo root, with data under ./maindata/).
DATA_DIR = os.environ.get("DATA_DIR", ".")

class TwoStageConfig:
    """Configuration class for the two-stage HGT citation prediction model."""

    # ==================== Data configuration ====================
    DATA_CONFIG = {
        'train_path': os.path.join(DATA_DIR, "maindata/test1/train_processsed.jsonl"),
        'eval_path': os.path.join(DATA_DIR, "maindata/test1/eval_processsed.jsonl"),
        'test_path': os.path.join(DATA_DIR, "maindata/test1/test_processsed.jsonl"),
        'venue_threshold': 0.8,  # Threshold for high-tier venues
    }

    # ==================== Model architecture configuration ====================
    MODEL_CONFIG = {
        # Node types and edge types
        'node_types': ['paper', 'author', 'venue', 'topic'],
        'edge_types': [
            ('paper', 'written_by', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'published_in', 'venue'),
            ('venue', 'publishes', 'paper'),
            ('author', 'publishes_in', 'venue'),
            ('venue', 'has_author', 'author'),
            ('paper', 'has_topic', 'topic'),
            ('topic', 'topic_of', 'paper'),
            ('paper', 'cites', 'paper')
        ],
        
        # Node feature dimensions
        'node_type_to_feature_dim': {
            'paper': 8,    # [R,Q,C,H,Y,A1,A2,A3]
            'author': 1,
            'venue': 1,
            'topic': 1
        },

        # Model structure parameters
        'hidden_dim': 32,        # Hidden layer dimension
        'num_heads': 4,           # Number of attention heads
        'num_layers': 2,          # Number of HGT layers
        'dropout': 0.3,           # Dropout rate

        # Model feature switches
        'use_negative_binomial': True,   # Use a negative binomial distribution
        'use_adversarial': True,         # Whether to use adversarial training
        'aux_loss_weight': 0.01,          # Auxiliary loss weight
    }

    # ==================== Training configuration ====================
    TRAINING_CONFIG = {
        # Optimizer configuration
        'optimizer': {
            'type': 'AdamW',
            'lr': 0.0001,              # Initial learning rate
            'weight_decay': 1e-4,     # Weight decay
        },

        # Training parameters
        'num_epochs': 50,            # Number of training epochs
        'patience': 15,               # Early-stopping patience
        'verbose': True,              # Whether to print training information

        # Device configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }


    # ==================== Save configuration ====================
    SAVE_CONFIG = {
        'model_save_path': 'two_stage_model_standalone.pth',
    }

    # ==================== Random seed configuration ====================
    RANDOM_SEED = 42  # Fixed random seed
    
    @classmethod
    def set_random_seed(cls, seed=None):
        """Set all random seeds."""
        if seed is None:
            seed = cls.RANDOM_SEED

        import random
        import numpy as np
        import torch

        # Set the Python random seed
        random.seed(seed)

        # Set the NumPy random seed
        np.random.seed(seed)

        # Set the PyTorch random seed
        torch.manual_seed(seed)

        # Set the CUDA random seed (if using a GPU)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Enable deterministic behavior in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"✅ Random seed set to: {seed}")
        print("✅ Experiment results should now be fully reproducible!")

    # ==================== Configuration methods ====================

    @classmethod
    def get_data_paths(cls):
        """Return a dictionary of data paths."""
        return {
            'train': cls.DATA_CONFIG['train_path'],
            'eval': cls.DATA_CONFIG['eval_path'],
            'test': cls.DATA_CONFIG['test_path']
        }
    
    @classmethod
    def get_model_params(cls):
        """Return the model parameters."""
        return {
            'node_types': cls.MODEL_CONFIG['node_types'],
            'edge_types': cls.MODEL_CONFIG['edge_types'],
            'node_type_to_feature_dim': cls.MODEL_CONFIG['node_type_to_feature_dim'],
            'hidden_dim': cls.MODEL_CONFIG['hidden_dim'],
            'num_heads': cls.MODEL_CONFIG['num_heads'],
            'num_layers': cls.MODEL_CONFIG['num_layers'],
            'dropout': cls.MODEL_CONFIG['dropout'],
            'use_negative_binomial': cls.MODEL_CONFIG['use_negative_binomial'],
            'use_adversarial': cls.MODEL_CONFIG['use_adversarial'],
            'aux_loss_weight': cls.MODEL_CONFIG['aux_loss_weight']
        }
    
    @classmethod
    def get_optimizer_params(cls):
        """Return the optimizer parameters."""
        return cls.TRAINING_CONFIG['optimizer']


    @classmethod
    def get_training_params(cls):
        """Return the training parameters."""
        return {
            'num_epochs': cls.TRAINING_CONFIG['num_epochs'],
            'patience': cls.TRAINING_CONFIG['patience'],
            'verbose': cls.TRAINING_CONFIG['verbose'],
            'device': cls.TRAINING_CONFIG['device']
        }
    
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of the configuration."""
        print("="*60)
        print("Two-stage HGT citation prediction model configuration summary")
        print("="*60)

        print(f"\n📊 Data configuration:")
        print(f"  - Training data: {cls.DATA_CONFIG['train_path']}")
        print(f"  - Validation data: {cls.DATA_CONFIG['eval_path']}")
        print(f"  - Test data: {cls.DATA_CONFIG['test_path']}")
        print(f"  - Venue threshold: {cls.DATA_CONFIG['venue_threshold']}")

        print(f"\n🏗️ Model configuration:")
        print(f"  - Hidden dimension: {cls.MODEL_CONFIG['hidden_dim']}")
        print(f"  - Number of attention heads: {cls.MODEL_CONFIG['num_heads']}")
        print(f"  - Number of HGT layers: {cls.MODEL_CONFIG['num_layers']}")
        print(f"  - Dropout rate: {cls.MODEL_CONFIG['dropout']}")
        print(f"  - Negative binomial distribution: {cls.MODEL_CONFIG['use_negative_binomial']}")
        print(f"  - Adversarial training: {cls.MODEL_CONFIG['use_adversarial']}")
        print(f"  - Auxiliary loss weight: {cls.MODEL_CONFIG['aux_loss_weight']}")

        print(f"\n🎯 Training configuration:")
        print(f"  - Initial learning rate: {cls.TRAINING_CONFIG['optimizer']['lr']}")
        print(f"  - Number of training epochs: {cls.TRAINING_CONFIG['num_epochs']}")
        print(f"  - Early-stopping patience: {cls.TRAINING_CONFIG['patience']}")
        print(f"  - Device: {cls.TRAINING_CONFIG['device']}")

        print(f"\n💾 Save configuration:")
        print(f"  - Model save path: {cls.SAVE_CONFIG['model_save_path']}")

        print("="*60)

    # ==================== Parameter tuning methods ====================

    @classmethod
    def set_model_architecture(cls, hidden_dim=None, num_heads=None, num_layers=None, dropout=None):
        """Set the model architecture parameters."""
        if hidden_dim is not None:
            cls.MODEL_CONFIG['hidden_dim'] = hidden_dim
            print(f"✅ Hidden dimension set to: {hidden_dim}")

        if num_heads is not None:
            cls.MODEL_CONFIG['num_heads'] = num_heads
            print(f"✅ Number of attention heads set to: {num_heads}")

        if num_layers is not None:
            cls.MODEL_CONFIG['num_layers'] = num_layers
            print(f"✅ Number of HGT layers set to: {num_layers}")

        if dropout is not None:
            cls.MODEL_CONFIG['dropout'] = dropout
            print(f"✅ Dropout rate set to: {dropout}")

    @classmethod
    def set_training_params(cls, lr=None, num_epochs=None, patience=None):
        """Set the training parameters."""
        if lr is not None:
            cls.TRAINING_CONFIG['optimizer']['lr'] = lr
            print(f"✅ Learning rate set to: {lr}")

        if num_epochs is not None:
            cls.TRAINING_CONFIG['num_epochs'] = num_epochs
            print(f"✅ Number of training epochs set to: {num_epochs}")

        if patience is not None:
            cls.TRAINING_CONFIG['patience'] = patience
            print(f"✅ Early-stopping patience set to: {patience}")


    @classmethod
    def toggle_model_features(cls, use_negative_binomial=None, use_adversarial=None):
        """Toggle the model feature switches."""
        if use_negative_binomial is not None:
            cls.MODEL_CONFIG['use_negative_binomial'] = use_negative_binomial
            print(f"✅ Negative binomial distribution: {'enabled' if use_negative_binomial else 'disabled'}")

        if use_adversarial is not None:
            cls.MODEL_CONFIG['use_adversarial'] = use_adversarial
            print(f"✅ Adversarial training: {'enabled' if use_adversarial else 'disabled'}")


# Create a global configuration instance
config = TwoStageConfig()

if __name__ == "__main__":
    # Print the configuration summary
    config.print_config_summary()

    # Example: tune parameters
    print("\n📝 Parameter tuning example:")
    config.set_model_architecture(hidden_dim=256, num_heads=8)
    config.set_training_params(lr=0.0005, num_epochs=150)
    config.toggle_model_features(use_adversarial=True)

    print("\nUpdated configuration:")
    config.print_config_summary()
