"""
Standalone configuration file for the multivariate counterfactual two-stage HGT model.
Supports manual selection of counterfactual variables and flexible configuration of model parameters.
"""

import torch

class CounterfactualConfig:
    """Configuration class for the multivariate counterfactual two-stage HGT model."""

    # ==================== Data configuration ====================
    DATA_CONFIG = {
        'train_path': "maindata/Openalex/test1_distribution_aware/train_processsed.jsonl",
        'eval_path': "maindata/Openalex/test1_distribution_aware/eval_processsed.jsonl",
        'test_path': "maindata/Openalex/test1_distribution_aware/test_processsed.jsonl",
        'venue_threshold': 0.8,  # Threshold for high-tier venues
    }

    # ==================== Counterfactual variable configuration ====================
    COUNTERFACTUAL_CONFIG = {
        # Available counterfactual variables and their configurations
        'available_variables': {
            'R': {
                'name': 'reproducibility',
                'description': 'Reproducibility (0=not open-source, 1=open-source)',
                'index': 0,  # Index within the paper features
                'high_value': 1.0,  # Counterfactual high value
                'low_threshold': 0.0,  # Low-value threshold
                'monotonic_direction': 1,  # Monotonicity direction: 1 means a higher value should increase citations
            },
            'Q': {
                'name': 'content_quality',
                'description': 'Content quality (1-5 score)',
                'index': 1,
                'high_value': 1.0,  # Maximum value after normalization
                'low_threshold': 0.5,  # Below the median
                'monotonic_direction': 1,
            },
            'C': {
                'name': 'collaboration_network',
                'description': 'Collaboration network strength (1-5 score)',
                'index': 2,
                'high_value': 1.0,
                'low_threshold': 0.5,
                'monotonic_direction': 1,
            },
            'H': {
                'name': 'hotness',
                'description': 'Topic hotness',
                'index': 3,
                'high_value': 1.0,  # High value after standardization
                'low_threshold': 0.0,  # Below the mean
                'monotonic_direction': 1,
            },
            'A1': {
                'name': 'first_author_reputation',
                'description': 'First-author reputation (1-5 score)',
                'index': 5,
                'high_value': 1.0,
                'low_threshold': 0.5,
                'monotonic_direction': 1,
            },
            'A2': {
                'name': 'last_author_reputation',
                'description': 'Last-author reputation (1-5 score)',
                'index': 6,
                'high_value': 1.0,
                'low_threshold': 0.5,
                'monotonic_direction': 1,
            },
            'A3': {
                'name': 'other_authors_reputation',
                'description': 'Other-authors reputation (1-5 score)',
                'index': 7,
                'high_value': 1.0,
                'low_threshold': 0.5,
                'monotonic_direction': 1,
            }
        },

        # Manually select the counterfactual variables to use
        'enabled_variables': {
            'R': True,   # Reproducibility
            'Q': True,   # Content quality
            'C': False,   # Collaboration network
            'H': False,  # Topic hotness (disabled by default)
            'A1': False, # First-author reputation (disabled by default)
            'A2': False, # Last-author reputation (disabled by default)
            'A3': False, # Other-authors reputation (disabled by default)
        },

        # Counterfactual loss configuration
        'loss_weight': 0.001,  # Counterfactual loss weight (reduced 10x to stabilize the main task first)
        'smoothness_weight': 0.05,  # Smoothness constraint weight (halved)
        'monotonic_weight': 0.5,  # Monotonicity constraint weight (halved)
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
        'hidden_dim': 128,        # Hidden layer dimension
        'num_heads': 4,           # Number of attention heads
        'num_layers': 2,          # Number of HGT layers
        'dropout': 0.4,           # Dropout rate

        # Model feature switches
        'use_negative_binomial': False,   # Use a negative binomial distribution
        'use_adversarial': False,        # Pause adversarial training to stabilize the main task first
        'aux_loss_weight': 0.03,         # Auxiliary loss weight (slightly reduced)
    }

    # ==================== Training configuration ====================
    TRAINING_CONFIG = {
        # Optimizer configuration
        'optimizer': {
            'type': 'AdamW',
            'lr': 0.001,              # Initial learning rate
            'weight_decay': 1e-4,     # Weight decay
        },

        # Learning-rate scheduler configuration
        'scheduler': {
            'type': 'warmup_cosine',          # Scheduler type: 'cosine', 'step', 'plateau', 'warmup_cosine'
            'warmup_epochs': 10,       # Number of warmup epochs
            'min_lr': 1e-5,           # Minimum learning rate
            'step_size': 50,           # Step size for StepLR
            'gamma': 0.1,              # Decay factor for StepLR
            'patience': 20,            # Patience for ReduceLROnPlateau
            'factor': 0.5,             # Decay factor for ReduceLROnPlateau
        },

        # Training parameters
        'num_epochs': 200,            # Number of training epochs
        'patience': 50,               # Early-stopping patience
        'batch_size': 128,             # Batch size
        'verbose': True,              # Whether to print training information

        # Device configuration
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # ==================== GroupDRO configuration ====================
    GROUPDRO_CONFIG = {
        'alpha': 0.05,                 # GroupDRO learning rate (halved)
        'num_environments': 2,        # Number of environments
        'min_weight': 0.2,            # Minimum environment weight (narrowed)
        'max_weight': 0.8,            # Maximum environment weight (narrowed)
    }

    # ==================== Loss function configuration ====================
    LOSS_CONFIG = {
        'main_loss_weight': 1.0,      # Main loss weight
        'aux_loss_weight': 0.005,      # Auxiliary loss weight
        'adversarial_loss_weight': 0.02,  # Adversarial loss weight (significantly reduced)
        'correlation_loss_weight': 0.05,  # Correlation constraint loss weight (reduced)
        # y1 regression head: whether to apply log1p regression to the target (enabled by default)
        'use_log1p_y': True,
        # Long-tail focal weighting (applied only to the main training loss)
        'focal_tail': True,
        'focal_gamma': 2.0,
        'focal_min_weight': 0.2,
        'focal_ref_percentile': 95.0,
    }

    # ==================== Evaluation configuration ====================
    EVALUATION_CONFIG = {
        'metrics': ['mae', 'rmse', 'loss'],  # Evaluation metrics
        'log_transform': True,        # Whether to use a log1p transform
        'save_predictions': True,     # Whether to save prediction results
    }

    # ==================== Save configuration ====================
    SAVE_CONFIG = {
        'model_save_path': 'gnn/saved_models/counterfactual_two_stage_model.pth',
        'plot_save_path': 'gnn/pics/counterfactual_training_curves.png',
        'best_model_name': 'best_counterfactual_model.pth',
    }

    # ==================== Random seed configuration ====================
    RANDOM_SEED = 42  # Fixed random seed

    # ==================== Dynamic parameter configuration ====================
    _current_year = 5  # Current year, defaults to 5

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

    # ==================== Print configuration ====================
    PRINT_CONFIG = {
        'print_interval': 10,         # Print interval
        'analysis_interval': 20,      # Analysis interval
        'show_weight_changes': True,  # Whether to show weight changes
        'show_env_analysis': True,    # Whether to show environment analysis
    }

    # ==================== Configuration methods ====================

    @classmethod
    def enable_variable(cls, var_name, enabled=True):
        """Enable or disable a counterfactual variable."""
        if var_name in cls.COUNTERFACTUAL_CONFIG['available_variables']:
            cls.COUNTERFACTUAL_CONFIG['enabled_variables'][var_name] = enabled
            print(f"✅ Variable {var_name} ({cls.COUNTERFACTUAL_CONFIG['available_variables'][var_name]['name']}) {'enabled' if enabled else 'disabled'}")
        else:
            print(f"❌ Variable {var_name} does not exist")

    @classmethod
    def get_enabled_variables(cls):
        """Return the enabled counterfactual variables."""
        enabled = []
        for var_name, enabled_flag in cls.COUNTERFACTUAL_CONFIG['enabled_variables'].items():
            if enabled_flag:
                enabled.append(var_name)
        return enabled
    
    @classmethod
    def get_variable_config(cls, var_name):
        """Return the configuration for a specified variable."""
        if var_name in cls.COUNTERFACTUAL_CONFIG['available_variables']:
            return cls.COUNTERFACTUAL_CONFIG['available_variables'][var_name]
        else:
            return None

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
    def get_scheduler_params(cls):
        """Return the learning-rate scheduler parameters."""
        return cls.TRAINING_CONFIG['scheduler']

    @classmethod
    def get_training_params(cls):
        """Return the training parameters."""
        return {
            'num_epochs': cls.TRAINING_CONFIG['num_epochs'],
            'patience': cls.TRAINING_CONFIG['patience'],
            'batch_size': cls.TRAINING_CONFIG['batch_size'],
            'verbose': cls.TRAINING_CONFIG['verbose'],
            'device': cls.TRAINING_CONFIG['device']
        }
    
    @classmethod
    def get_groupdro_params(cls):
        """Return the GroupDRO parameters."""
        return {
            'alpha': cls.GROUPDRO_CONFIG['alpha'],
            'num_environments': cls.GROUPDRO_CONFIG['num_environments'],
            'min_weight': cls.GROUPDRO_CONFIG['min_weight'],
            'max_weight': cls.GROUPDRO_CONFIG['max_weight']
        }
    
    @classmethod
    def get_loss_weights(cls):
        """Return the loss weights."""
        return {
            'main_loss_weight': cls.LOSS_CONFIG['main_loss_weight'],
            'aux_loss_weight': cls.LOSS_CONFIG['aux_loss_weight'],
            'adversarial_loss_weight': cls.LOSS_CONFIG['adversarial_loss_weight'],
            'correlation_loss_weight': cls.LOSS_CONFIG['correlation_loss_weight']
        }
    
    @classmethod
    def get_counterfactual_config(cls):
        """Return the counterfactual configuration."""
        return cls.COUNTERFACTUAL_CONFIG

    @classmethod
    def print_config_summary(cls):
        """Print a summary of the configuration."""
        print("="*60)
        print("Multivariate counterfactual two-stage HGT model configuration summary")
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
        print(f"  - Auxiliary loss weight: {cls.MODEL_CONFIG['aux_loss_weight']}")

        print(f"\n🎯 Training configuration:")
        print(f"  - Initial learning rate: {cls.TRAINING_CONFIG['optimizer']['lr']}")
        print(f"  - Learning-rate scheduler: {cls.TRAINING_CONFIG['scheduler']['type']}")
        print(f"  - Number of warmup epochs: {cls.TRAINING_CONFIG['scheduler']['warmup_epochs']}")
        print(f"  - Minimum learning rate: {cls.TRAINING_CONFIG['scheduler']['min_lr']}")
        print(f"  - Number of training epochs: {cls.TRAINING_CONFIG['num_epochs']}")
        print(f"  - Batch size: {cls.TRAINING_CONFIG['batch_size']}")
        print(f"  - Early-stopping patience: {cls.TRAINING_CONFIG['patience']}")
        print(f"  - Device: {cls.TRAINING_CONFIG['device']}")

        print(f"\n⚖️ GroupDRO configuration:")
        print(f"  - GroupDRO learning rate: {cls.GROUPDRO_CONFIG['alpha']}")
        print(f"  - Number of environments: {cls.GROUPDRO_CONFIG['num_environments']}")
        print(f"  - Weight range: [{cls.GROUPDRO_CONFIG['min_weight']}, {cls.GROUPDRO_CONFIG['max_weight']}]")

        print(f"\n🔬 Counterfactual configuration:")
        print(f"  - Counterfactual loss weight: {cls.COUNTERFACTUAL_CONFIG['loss_weight']}")
        print(f"  - Smoothness constraint weight: {cls.COUNTERFACTUAL_CONFIG['smoothness_weight']}")
        print(f"  - Monotonicity constraint weight: {cls.COUNTERFACTUAL_CONFIG['monotonic_weight']}")

        enabled_vars = cls.get_enabled_variables()
        print(f"  - Enabled counterfactual variables: {enabled_vars}")
        for var_name in enabled_vars:
            var_config = cls.get_variable_config(var_name)
            print(f"    * {var_name}: {var_config['description']}")

        print(f"\n💾 Save configuration:")
        print(f"  - Model save path: {cls.SAVE_CONFIG['model_save_path']}")
        print(f"  - Plot save path: {cls.SAVE_CONFIG['plot_save_path']}")

        print("="*60)

    @classmethod
    def print_available_variables(cls):
        """Print all available counterfactual variables."""
        print("\n🔬 Available counterfactual variables:")
        print("-" * 50)
        for var_name, var_config in cls.COUNTERFACTUAL_CONFIG['available_variables'].items():
            enabled = "✅" if cls.COUNTERFACTUAL_CONFIG['enabled_variables'][var_name] else "❌"
            print(f"{enabled} {var_name}: {var_config['description']}")
            print(f"   Index: {var_config['index']}, high value: {var_config['high_value']}, low-value threshold: {var_config['low_threshold']}")
        print("-" * 50)

    @classmethod
    def get_all_config(cls):
        """Return all configuration information (for saving to a file)."""
        return {
            'experiment_info': {
                'model_name': 'GroupDRO+Counterfactual Two-Stage HGT',
                'description': 'Two-stage HGT citation prediction model (GroupDRO + counterfactual version)',
                'timestamp': cls._get_timestamp(),
                'random_seed': cls.RANDOM_SEED,
                'enabled_variables': cls.get_enabled_variables(),
                'enabled_variables_count': len(cls.get_enabled_variables())
            },
            'data_config': cls.DATA_CONFIG,
            'counterfactual_config': cls.COUNTERFACTUAL_CONFIG,
            'model_config': cls.MODEL_CONFIG,
            'training_config': cls.TRAINING_CONFIG,
            'groupdro_config': cls.GROUPDRO_CONFIG,
            'loss_config': cls.LOSS_CONFIG,
            'evaluation_config': cls.EVALUATION_CONFIG,
            'save_config': cls.SAVE_CONFIG,
            'print_config': cls.PRINT_CONFIG,
            'variable_details': {
                var_name: {
                    'enabled': cls.COUNTERFACTUAL_CONFIG['enabled_variables'][var_name],
                    **var_config
                }
                for var_name, var_config in cls.COUNTERFACTUAL_CONFIG['available_variables'].items()
            }
        }
    
    @classmethod
    def _get_timestamp(cls):
        """Return the current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def set_current_year(cls, year):
        """Set the current year."""
        cls._current_year = year
        print(f"✅ Current year set to: {year}")

    @classmethod
    def get_current_year(cls):
        """Return the current year."""
        return cls._current_year

    @classmethod
    def set_parameter(cls, param_name, param_value):
        """Dynamically set a parameter value."""
        if param_name in cls.COUNTERFACTUAL_CONFIG:
            cls.COUNTERFACTUAL_CONFIG[param_name] = param_value
            print(f"✅ Counterfactual parameter {param_name} set to: {param_value}")
        elif param_name in cls.GROUPDRO_CONFIG:
            cls.GROUPDRO_CONFIG[param_name] = param_value
            print(f"✅ GroupDRO parameter {param_name} set to: {param_value}")
        elif param_name in cls.LOSS_CONFIG:
            cls.LOSS_CONFIG[param_name] = param_value
            print(f"✅ Loss parameter {param_name} set to: {param_value}")
        else:
            print(f"❌ Parameter {param_name} does not exist")

    @classmethod
    def get_parameter(cls, param_name):
        """Return a parameter value."""
        if param_name in cls.COUNTERFACTUAL_CONFIG:
            return cls.COUNTERFACTUAL_CONFIG[param_name]
        elif param_name in cls.GROUPDRO_CONFIG:
            return cls.GROUPDRO_CONFIG[param_name]
        elif param_name in cls.LOSS_CONFIG:
            return cls.LOSS_CONFIG[param_name]
        else:
            return None


# Create a global configuration instance
config = CounterfactualConfig()

if __name__ == "__main__":
    # Print the configuration summary
    config.print_config_summary()

    # Print the available variables
    config.print_available_variables()

    # Example: enable/disable variables
    print("\n📝 Variable configuration example:")
    config.enable_variable('H', True)  # Enable topic hotness
    config.enable_variable('A1', True)  # Enable first-author reputation
    config.enable_variable('C', False)  # Disable collaboration network

    print(f"Currently enabled variables: {config.get_enabled_variables()}")