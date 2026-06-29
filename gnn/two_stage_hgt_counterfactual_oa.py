"""
Two-stage HGT citation prediction model (GroupDRO + counterfactual version)
Builds a counterfactual side-head on top of GroupDRO environment-split training to learn the causal effects of actionable variables

Main features:
1. Two-stage architecture: V->E->Y path
2. GroupDRO training: addresses the 'top-venue halo' problem via environment-split training
3. Counterfactual prediction: learns the causal effect of the R variable (reproducibility)
4. Monotonicity constraint: ensures that raising R from 0->1 does not lower the citation count
5. Uses real data: keeps the same data-loading approach as the original model
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, LinearLR, SequentialLR
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import NeighborLoader
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import pickle
import os

# Import the base two-stage model and related components
from two_stage_hgt_oa import (
    TwoStageCitationHGT, 
    ModifiedHeterographDataBuilder,
    PaperFeatures,
    AuthorFeatures, 
    VenueFeatures,
    TopicFeatures,
    load_separated_data_modified
)

# Import the standalone counterfactual config file
from counterfactual_config_oa import config


class CounterfactualTwoStageHGT(TwoStageCitationHGT):
    """Two-stage HGT model with a flexible multi-variable counterfactual head"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get the dropout parameter from kwargs and store it
        self.dropout = kwargs.get('dropout', 0.2)

        # Get the enabled counterfactual variables
        self.enabled_variables = config.get_enabled_variables()

        # Add flexible counterfactual heads: create a head only for enabled variables
        # Input dim: hidden_dim + 8 (paper features) + 1 (early exposure)
        counterfactual_input_dim = self.hidden_dim + 8 + 1

        # Create a counterfactual head for each enabled variable
        self.counterfactual_heads = nn.ModuleDict()
        for var_name in self.enabled_variables:
            var_config = config.get_variable_config(var_name)
            self.counterfactual_heads[var_name] = nn.Sequential(
                nn.Linear(counterfactual_input_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1)
            )
        
        # Counterfactual loss weight
        cf_config = config.get_counterfactual_config()
        self.counterfactual_loss_weight = cf_config['loss_weight']

        print(f"Flexible counterfactual head input dim: {counterfactual_input_dim}")
        print(f"Counterfactual loss weight: {self.counterfactual_loss_weight}")
        print(f"Enabled counterfactual variables: {self.enabled_variables}")
        for var_name in self.enabled_variables:
            var_config = config.get_variable_config(var_name)
            print(f"  - {var_name}: {var_config['description']}")

    def forward(self, x_dict, edge_index_dict, paper_features):
        """Forward pass, including multi-variable counterfactual prediction"""
        # Original forward pass
        output = super().forward(x_dict, edge_index_dict, paper_features)

        # Counterfactual prediction (only during training)
        if self.training:
            # Get paper embeddings and early exposure
            paper_embeddings = output['embeddings']  # [num_papers, hidden_dim]
            early_exposure = output['early_exposure']  # [num_papers, 1]

            # Concatenate features: paper_embeddings + paper_features + early_exposure
            # Ensure early_exposure is 2-dimensional
            if early_exposure.dim() == 1:
                early_exposure = early_exposure.unsqueeze(1)  # [num_papers, 1]

            counterfactual_input = torch.cat([
                paper_embeddings,  # [num_papers, hidden_dim]
                paper_features,    # [num_papers, 8]
                early_exposure     # [num_papers, 1]
            ], dim=1)  # [num_papers, hidden_dim + 8 + 1]

            # Multi-variable counterfactual prediction
            counterfactual_predictions = {}
            for var_name, head in self.counterfactual_heads.items():
                counterfactual_predictions[var_name] = head(counterfactual_input)
            
            output['counterfactual'] = counterfactual_predictions
        
        return output
    
    def counterfactual_loss(self, original_pred, counterfactual_preds, variable_values):
        """
        Compute the flexible multi-variable counterfactual loss

        Args:
            original_pred: original prediction [batch_size]
            counterfactual_preds: dict of counterfactual predictions {var_name: [batch_size]}
            variable_values: variable values [batch_size, 8]

        Returns:
            loss: total counterfactual loss
            stats: statistics
        """
        total_loss = 0.0
        all_stats = {}

        cf_config = config.get_counterfactual_config()
        smoothness_weight = cf_config['smoothness_weight']
        monotonic_weight = cf_config['monotonic_weight']

        for var_name, counterfactual_pred in counterfactual_preds.items():
            var_config = config.get_variable_config(var_name)
            var_idx = var_config['index']
            var_values = variable_values[:, var_idx:var_idx+1]  # [batch_size, 1]

            # Compute the causal effect: Δ = Y(var=high) - Y(var=low)
            delta = counterfactual_pred - original_pred

            # Monotonicity constraint: for low-value samples, Δ should be > 0 (improving quality should increase citations)
            low_value_mask = (var_values.squeeze() < var_config['low_threshold']).float()
            target_delta = var_config['monotonic_direction']  # expected direction of change

            # Monotonicity loss: for low-value samples, expect a positive change
            monotonic_loss = F.relu(-delta * low_value_mask * target_delta).mean()

            # Smoothness constraint: avoid extreme changes
            smoothness_loss = F.mse_loss(delta, torch.zeros_like(delta))

            # Variable-specific counterfactual loss
            var_loss = monotonic_weight * monotonic_loss + smoothness_weight * smoothness_loss
            total_loss += var_loss

            # Statistics
            all_stats[f'{var_name}_monotonic_loss'] = monotonic_loss.item()
            all_stats[f'{var_name}_smoothness_loss'] = smoothness_loss.item()
            all_stats[f'{var_name}_delta_mean'] = delta.mean().item()
            all_stats[f'{var_name}_delta_std'] = delta.std().item()
            all_stats[f'{var_name}_low_value_count'] = low_value_mask.sum().item()
        
        return total_loss, all_stats


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics (consistent with dygformer_original_style.py)"""
    # Basic regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Log-scale error metrics (implementation consistent with run.py)
    mask = (y_true >= 0) & (y_pred >= 0)
    if np.any(mask):
        rmsle = np.sqrt(np.mean((np.log1p(y_pred[mask]) - np.log1p(y_true[mask]))**2))
        male = np.mean(np.abs(np.log1p(y_pred[mask]) - np.log1p(y_true[mask])))
    else:
        rmsle = np.nan
        male = np.nan

    # Correlation metrics
    spearman_corr, _ = spearmanr(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    # Ranking metrics
    def calculate_ndcg(y_true, y_pred, k):
        """
        Normalized Discounted Cumulative Gain (user version: linear relevance function)
        """
        try:
            if len(y_true) == 0:
                return 0.0

            # Ensure k does not exceed the data length
            k = min(k, len(y_true))

            # Sort by predicted score
            sorted_indices = np.argsort(y_pred)[::-1]

            # Compute DCG
            dcg = 0.0
            for i in range(k):
                idx = sorted_indices[i]
                dcg += y_true[idx] / np.log2(i + 2)

            # Compute IDCG (ideal DCG)
            ideal_sorted_indices = np.argsort(y_true)[::-1]
            idcg = 0.0
            for i in range(k):
                idx = ideal_sorted_indices[i]
                idcg += y_true[idx] / np.log2(i + 2)

            # Avoid division by zero
            if idcg == 0:
                return 0.0

            return dcg / idcg
        except Exception as e:
            print(f"⚠️ Error computing NDCG@{k}: {e}")
            return 0.0

    def calculate_hit(y_true, y_pred, k):
        """
        Hit@K metric (user version: high-value detection based on the 75th percentile)
        """
        try:
            if len(y_true) == 0:
                return 0.0

            # Sort by predicted score and take the top k
            top_k_indices = np.argsort(y_pred)[::-1][:k]

            # For the regression problem, check whether the top k contain high-value samples
            # Use the 75th percentile as the "high-value" threshold
            threshold = np.percentile(y_true, 75)
            high_value_indices = np.where(y_true >= threshold)[0]

            # Check whether the top k predictions contain high-value samples
            hit_count = len(set(top_k_indices) & set(high_value_indices))
            return hit_count / k if k > 0 else 0.0
        except Exception as e:
            print(f"⚠️ Error computing Hit@{k}: {e}")
            return 0.0

    # Use K=10 and 20 for ranking-metric evaluation
    ndcg_10 = calculate_ndcg(y_true, y_pred, 10)
    ndcg_20 = calculate_ndcg(y_true, y_pred, 20)
    hit_10 = calculate_hit(y_true, y_pred, 10)
    hit_20 = calculate_hit(y_true, y_pred, 20)

    # Percentage error metric
    def safe_divide(a, b, eps=5):
        return a / (b + eps)
    
    mape = np.mean(np.abs(safe_divide(y_true - y_pred, y_true, eps=5))) * 100
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'RMSLE': float(rmsle) if not np.isnan(rmsle) else np.nan,
        'MALE': float(male) if not np.isnan(male) else np.nan,
        'Spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        'Pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        'NDCG@10': float(ndcg_10),
        'NDCG@20': float(ndcg_20),
        'Hit@10': float(hit_10),
        'Hit@20': float(hit_20),
        'MAPE': float(mape)
    }


def print_detailed_metrics(metrics: Dict[str, float], split_name: str):
    """Print detailed evaluation metrics (consistent with dygformer_original_style.py)"""
    print("=" * 60)
    print(f"📊 {split_name} evaluation metric results (counterfactual two-stage HGT):")
    print("=" * 60)
    print("📈 Basic regression metrics:")
    print("-" * 40)
    print(f"MSE   : {metrics['MSE']:.6f}")
    print(f"RMSE  : {metrics['RMSE']:.6f}")
    print(f"MAE   : {metrics['MAE']:.6f}")
    rmsle_val = "N/A" if np.isnan(metrics['RMSLE']) else f"{metrics['RMSLE']:.6f}"
    male_val = "N/A" if np.isnan(metrics['MALE']) else f"{metrics['MALE']:.6f}"
    print(f"RMSLE : {rmsle_val}")
    print(f"MALE  : {male_val}")
    print(f"MAPE  : {metrics['MAPE']:.6f}")
    print()
    print("🔗 Correlation metrics:")
    print("-" * 40)
    print(f"R2       : {metrics['R2']:.6f}")
    print(f"Spearman : {metrics['Spearman']:.6f}")
    print(f"Pearson  : {metrics['Pearson']:.6f}")
    print()
    print("🏆 Ranking metrics:")
    print("-" * 40)
    print(f"NDCG@10 : {metrics['NDCG@10']:.6f} (linear relevance version)")
    print(f"NDCG@20 : {metrics['NDCG@20']:.6f} (linear relevance version)")
    print(f"Hit@10  : {metrics['Hit@10']:.6f} (proportion of high-value papers in top 10)")
    print(f"Hit@20  : {metrics['Hit@20']:.6f} (proportion of high-value papers in top 20)")
    print("=" * 60)


class GroupDROCounterfactualTrainer:
    """GroupDRO + counterfactual two-stage model trainer"""

    def __init__(self,
                 model: CounterfactualTwoStageHGT,
                 data: HeteroData,
                 device: str = None,
                 venue_labels: Optional[torch.Tensor] = None,
                 groupdro_alpha: float = None):
        # Use defaults from the config file
        self.device = device or config.TRAINING_CONFIG['device']
        self.groupdro_alpha = groupdro_alpha or config.GROUPDRO_CONFIG['alpha']

        # Make sure the random seed is set
        config.set_random_seed()

        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.venue_labels = venue_labels

        # Create data splits
        self._create_splits()

        # Create environment labels (for GroupDRO)
        self._create_environment_labels()

        # Initialize the optimizer
        optimizer_params = config.get_optimizer_params()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=optimizer_params['lr'],
            weight_decay=optimizer_params['weight_decay']
        )

        # Initialize the learning-rate scheduler
        scheduler_params = config.get_scheduler_params()
        self.scheduler = self._create_scheduler(scheduler_params)

        # Initialize GroupDRO weights
        self.num_environments = config.GROUPDRO_CONFIG['num_environments']
        self.env_weights = torch.ones(self.num_environments, device=self.device) / self.num_environments

        # Record training history
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.env_losses = []  # record per-environment losses
        self.env_weights_history = []  # record environment-weight changes
        self.counterfactual_losses = []
        self.delta_r_stats = []

        # Compute the quantile of the training-set targets, used only to clip extreme values in the loss (does not affect evaluation or saving)
        with torch.no_grad():
            train_targets_full = self.data['paper'].y[self.train_mask].float().to(self.device)
            # Use the 95th percentile as the clipping upper bound (experimental)
            self.target_clip_percentile = 95.0
            try:
                self.target_clip_value = torch.quantile(train_targets_full, 0.95).item()
            except Exception:
                self.target_clip_value = train_targets_full.max().item()
            # Compute the focal reference strength (same percentile)
            self.focal_ref_value = self.target_clip_value
        print(f"In-loss target clipping: p{self.target_clip_percentile} = {self.target_clip_value:.2f}")

    def _create_splits(self):
        """Create train/val/test splits (using the pre-split data)"""
        paper_splits = self.data['paper'].split
        
        self.train_mask = torch.tensor([split == 'train' for split in paper_splits], device=self.device)
        self.val_mask = torch.tensor([split == 'eval' for split in paper_splits], device=self.device)
        self.test_mask = torch.tensor([split == 'test' for split in paper_splits], device=self.device)
        
        print(f"Train samples: {self.train_mask.sum().item()}")
        print(f"Val samples: {self.val_mask.sum().item()}")
        print(f"Test samples: {self.test_mask.sum().item()}")
    
    def _create_environment_labels(self):
        """Create environment labels (based on venue tier)"""
        if self.venue_labels is None:
            # If no venue labels are provided, extract them from venue features
            venue_features = self.data['venue'].x
            venue_threshold = config.DATA_CONFIG['venue_threshold']
            venue_labels = (venue_features >= venue_threshold).float().squeeze()

            # Map venue labels onto papers
            paper_venue_edges = self.data['paper', 'published_in', 'venue'].edge_index
            paper_venue_labels = torch.zeros(self.data['paper'].x.size(0), device=self.device)

            for paper_idx, venue_idx in paper_venue_edges.t():
                paper_venue_labels[paper_idx] = venue_labels[venue_idx]

            self.venue_labels = paper_venue_labels

        # Create environment labels: 0 = low-tier venue, 1 = high-tier venue
        self.environment_labels = (self.venue_labels > 0.5).long()

        # Count samples per environment
        env_counts = torch.bincount(self.environment_labels)
        print(f"Environment distribution: low-tier venue={env_counts[0].item()}, high-tier venue={env_counts[1].item()}")

    def _create_scheduler(self, scheduler_params):
        """Create the learning-rate scheduler"""
        scheduler_type = scheduler_params['type']

        if scheduler_type == 'cosine':
            # Cosine annealing scheduler
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.TRAINING_CONFIG['num_epochs'],
                eta_min=scheduler_params['min_lr']
            )
        elif scheduler_type == 'warmup_cosine':
            # Warmup + cosine annealing scheduler
            warmup_epochs = scheduler_params['warmup_epochs']
            total_epochs = config.TRAINING_CONFIG['num_epochs']

            # Warmup phase: linear increase
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs
            )

            # Cosine annealing phase
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=scheduler_params['min_lr']
            )

            # Combined scheduler
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        elif scheduler_type == 'step':
            # Step scheduler
            return StepLR(
                self.optimizer,
                step_size=scheduler_params['step_size'],
                gamma=scheduler_params['gamma']
            )
        elif scheduler_type == 'plateau':
            # Validation-loss-based scheduler
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params['factor'],
                patience=scheduler_params['patience'],
                min_lr=scheduler_params['min_lr']
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _get_paper_features(self, mask):
        """Get the paper feature vector [R,Q,C,H,Y,A1,A2,A3] - consistent with the original model"""
        # Get the 8-dim paper features directly from the data
        paper_x = self.data['paper'].x[mask]  # already 8-dim features
        return paper_x

    def _create_counterfactual_features(self, paper_features, var_name, high_value=None):
        """
        Create counterfactual features: set the specified variable to its high value

        Args:
            paper_features: original paper features [batch_size, 8]
            var_name: variable name
            high_value: the high value of the variable (if None, use the default from config)

        Returns:
            counterfactual_features: counterfactual features [batch_size, 8]
        """
        counterfactual_features = paper_features.clone()
        
        var_config = config.get_variable_config(var_name)
        if var_config is not None:
            var_idx = var_config['index']
            if high_value is None:
                high_value = var_config['high_value']
            counterfactual_features[:, var_idx] = high_value
        
        return counterfactual_features
    
    def group_dro_loss(self, predictions, targets, environments, model_output=None, train_mask=None):
        """GroupDRO loss function (consistent with the original GroupDRO version)"""
        env_losses = []
        env_counts = []

        for env_id in range(self.num_environments):
            env_mask = (environments == env_id)  # mask over the training set
            if env_mask.sum() > 0:
                # Use the same loss computation as the main loss function
                if self.model.use_negative_binomial and model_output is not None and train_mask is not None:
                    # Use the negative binomial loss
                    # Need to apply the training-set mask to the output over the whole dataset
                    train_env_mask = train_mask.clone()
                    train_env_mask[train_mask] = env_mask  # map env_mask onto the whole dataset

                    env_loss = self.model.negative_binomial_loss(
                        model_output['citation']['mu'][train_env_mask],
                        model_output['citation']['alpha'][train_env_mask],
                        targets[env_mask]
                    )
                else:
                    # Use the MSE log loss
                    log_predictions = torch.log1p(predictions[env_mask])
                    log_targets = torch.log1p(targets[env_mask])
                    env_loss = F.mse_loss(log_predictions, log_targets)

                env_losses.append(env_loss)
                env_counts.append(env_mask.sum().item())
            else:
                env_losses.append(torch.tensor(0.0, device=self.device))
                env_counts.append(0)

        # Compute the weighted loss
        weighted_loss = sum(w * loss for w, loss in zip(self.env_weights, env_losses))

        # Update environment weights (focus on the worst environment)
        if len(env_losses) > 0:
            with torch.no_grad():
                # Compute loss gradients
                loss_gradients = torch.stack(env_losses)

                # Update weights: increase the weight of the worst environment
                # Use normalized loss differences to adjust weights and avoid exponential blow-up
                loss_diff = loss_gradients - loss_gradients.mean()
                loss_diff_std = loss_diff.std() + 1e-8  # avoid division by zero
                normalized_loss_diff = loss_diff / loss_diff_std  # normalize
                new_weights = self.env_weights * torch.exp(self.groupdro_alpha * normalized_loss_diff)
                new_weights = new_weights / new_weights.sum()  # normalize

                # Weight smoothing: prevent extremes while allowing more flexible adjustment
                min_weight = config.GROUPDRO_CONFIG['min_weight']
                max_weight = config.GROUPDRO_CONFIG['max_weight']
                new_weights = torch.clamp(new_weights, min=min_weight, max=max_weight)
                new_weights = new_weights / new_weights.sum()  # re-normalize

                self.env_weights = new_weights

        return weighted_loss, env_losses, env_counts

    def train_epoch(self, epoch: int):
        """Train for one epoch (GroupDRO + counterfactual)"""
        self.model.train()
        self.optimizer.zero_grad()

        # Generate features for all papers
        all_features = self._get_paper_features(torch.ones(self.data['paper'].x.size(0), dtype=torch.bool))

        # Forward pass (original features)
        output = self.model(
            self.data.x_dict,
            self.data.edge_index_dict,
            all_features
        )

        # Get the training data
        train_mask = self.train_mask
        targets = self.data['paper'].y[train_mask].float()
        # Use clipped targets only in the loss, to keep an extreme long tail from dominating training
        targets_clipped = torch.clamp(targets, min=0.0, max=self.target_clip_value)
        environments = self.environment_labels[train_mask]
        train_features = all_features[train_mask]

        # Main task loss (citation prediction)
        if self.model.use_negative_binomial:
            main_loss = self.model.negative_binomial_loss(
                output['citation']['mu'][train_mask],
                output['citation']['alpha'][train_mask],
                targets_clipped
            )
            original_predictions = output['citation']['mu'][train_mask]
        else:
            # Standard log-MSE or original-scale MSE on y1 (configurable)
            lw_cfg = config.get_loss_weights()
            if lw_cfg.get('use_log1p_y', True):
                base_loss = self.model.mse_log_loss(
                    output['citation']['log_citation'][train_mask],
                    targets_clipped
                )
            else:
                # Direct original-scale MSE
                preds = torch.expm1(output['citation']['log_citation'][train_mask])
                base_loss = F.mse_loss(preds, targets_clipped)
            # Tail focal weighting (lowers the weight of samples with extremely large targets)
            lw = config.get_loss_weights()
            if lw.get('focal_tail', False):
                gamma = lw.get('focal_gamma', 2.0)
                min_w = lw.get('focal_min_weight', 0.2)
                ref_v = self.focal_ref_value
                # Build weights based on target magnitude: w = (1 + y/ref_v)^(-gamma)
                y_norm = targets_clipped / max(ref_v, 1.0)
                weights = torch.pow(1.0 + y_norm, -gamma)
                weights = torch.clamp(weights, min=min_w, max=1.0)
                # Approximately rescale the scalar loss by the weights (scalar approximation: multiply by the mean weight)
                weight_mean = weights.mean()
                main_loss = base_loss * weight_mean
            else:
                main_loss = base_loss
            original_predictions = torch.expm1(output['citation']['log_citation'][train_mask])

        # GroupDRO loss
        groupdro_loss, env_losses, env_counts = self.group_dro_loss(
            original_predictions, targets_clipped, environments, output, train_mask
        )

        # Flexible counterfactual prediction: create counterfactual predictions only for enabled variables
        counterfactual_predictions = {}

        # Counterfactual warmup (skip or down-weight for the first several epochs)
        cf_warmup_epochs = 20
        if epoch >= cf_warmup_epochs:
            for var_name in self.model.enabled_variables:
                # Create counterfactual features for all papers, but modify only the specified variable on the training set
                all_counterfactual_features = all_features.clone()

                var_config = config.get_variable_config(var_name)
                var_idx = var_config['index']
                all_counterfactual_features[train_mask, var_idx] = var_config['high_value']

                # Forward pass with the counterfactual features
                counterfactual_output = self.model(
                    self.data.x_dict,
                    self.data.edge_index_dict,
                    all_counterfactual_features
                )

                # Get the counterfactual prediction
                if self.model.use_negative_binomial:
                    counterfactual_predictions[var_name] = counterfactual_output['citation']['mu'][train_mask]
                else:
                    counterfactual_predictions[var_name] = torch.expm1(
                        counterfactual_output['citation']['log_citation'][train_mask]
                    )

        # Counterfactual loss
        if epoch >= cf_warmup_epochs and len(counterfactual_predictions) > 0:
            cf_loss, cf_stats = self.model.counterfactual_loss(
                original_predictions,
                counterfactual_predictions,
                train_features
            )
        else:
            cf_loss = torch.tensor(0.0, device=self.device)
            cf_stats = {}

        # Auxiliary loss (quartile prediction)
        aux_loss = self.model.quartile_loss(
            output['quartile_prediction'][train_mask],
            targets_clipped
        )

        # Adversarial loss (if enabled)
        adversarial_loss = torch.tensor(0.0, device=self.device)
        if self.model.use_adversarial and output['adversarial'] is not None:
            adversarial_loss = self.model.adversarial_loss(
                output['adversarial'][train_mask],
                self.venue_labels[train_mask]
            )

        # Correlation constraint loss
        correlation_loss = torch.tensor(0.0, device=self.device)
        if len(targets) > 1:
            early_exposure_train = output['early_exposure'][train_mask]
            targets_normalized = (targets_clipped.float() - targets_clipped.float().mean()) / (targets_clipped.float().std() + 1e-8)
            exposure_normalized = (early_exposure_train - early_exposure_train.mean()) / (early_exposure_train.std() + 1e-8)

            correlation = torch.mean(targets_normalized * exposure_normalized)
            correlation_loss = -correlation

        # Total loss (GroupDRO + counterfactual + adversarial)
        loss_weights = config.get_loss_weights()
        total_loss = (loss_weights['main_loss_weight'] * groupdro_loss + 
                     loss_weights['aux_loss_weight'] * aux_loss + 
                     loss_weights['adversarial_loss_weight'] * adversarial_loss +
                     loss_weights['correlation_loss_weight'] * correlation_loss +
                     self.model.counterfactual_loss_weight * cf_loss)

        # Backpropagation
        total_loss.backward()
        self.optimizer.step()

        # Record losses
        self.train_losses.append(total_loss.item())
        self.env_losses.append([loss.item() for loss in env_losses])
        self.env_weights_history.append(self.env_weights.clone().cpu().numpy())
        self.counterfactual_losses.append(cf_loss.item())
        self.delta_r_stats.append(cf_stats)
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'groupdro_loss': groupdro_loss.item(),
            'aux_loss': aux_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'correlation_loss': correlation_loss.item(),
            'counterfactual_loss': cf_loss.item() if isinstance(cf_loss, torch.Tensor) else 0.0,
            'counterfactual_stats': cf_stats,
            'env_losses': [loss.item() for loss in env_losses],
            'env_counts': env_counts,
            'env_weights': self.env_weights.clone().cpu().numpy()
        }
    
    def evaluate(self, split: str = 'val'):
        """Evaluate the model"""
        self.model.eval()

        with torch.no_grad():
            if split == 'val':
                mask = self.val_mask
            else:  # test
                mask = self.test_mask

            # Generate features for all papers
            all_features = self._get_paper_features(torch.ones(self.data['paper'].x.size(0), dtype=torch.bool))
            output = self.model(
                self.data.x_dict,
                self.data.edge_index_dict,
                all_features
            )

            targets = self.data['paper'].y[mask].cpu().numpy()
            eval_features = all_features[mask]

            # Citation prediction evaluation
            if self.model.use_negative_binomial:
                predictions = output['citation']['mu'][mask].cpu().numpy()
                loss = self.model.negative_binomial_loss(
                    output['citation']['mu'][mask],
                    output['citation']['alpha'][mask],
                    self.data['paper'].y[mask]
                ).item()
            else:
                lw_cfg = config.get_loss_weights()
                if lw_cfg.get('use_log1p_y', True):
                    log_predictions = output['citation']['log_citation'][mask].cpu().numpy()
                    predictions = np.expm1(log_predictions)
                    loss = self.model.mse_log_loss(
                        output['citation']['log_citation'][mask],
                        self.data['paper'].y[mask]
                    ).item()
                else:
                    # Original-scale evaluation loss
                    predictions = torch.expm1(output['citation']['log_citation'][mask]).cpu().numpy()
                    loss = F.mse_loss(
                        torch.tensor(predictions, device=self.device),
                        self.data['paper'].y[mask].float()
                    ).item()

            # Compute the full set of metrics (consistent with dygformer_original_style.py)
            metrics = calculate_metrics(targets, predictions)

            # Flexible counterfactual analysis: analyze only the enabled variables
            counterfactual_effects = {}

            for var_name in self.model.enabled_variables:
                # Create counterfactual features for all papers, but modify only the specified variable on the current split
                all_counterfactual_features = all_features.clone()

                var_config = config.get_variable_config(var_name)
                var_idx = var_config['index']
                all_counterfactual_features[mask, var_idx] = var_config['high_value']

                counterfactual_output = self.model(
                    self.data.x_dict,
                    self.data.edge_index_dict,
                    all_counterfactual_features
                )

                if self.model.use_negative_binomial:
                    counterfactual_predictions = counterfactual_output['citation']['mu'][mask].cpu().numpy()
                else:
                    counterfactual_predictions = np.expm1(counterfactual_output['citation']['log_citation'][mask].cpu().numpy())

                # Compute the counterfactual effect
                delta = counterfactual_predictions - predictions
                counterfactual_effects[var_name] = delta
            # Flexible multi-variable analysis
            eval_features_np = eval_features.cpu().numpy()

            # Flexible counterfactual statistics
            cf_stats = {}
            for var_name in self.model.enabled_variables:
                delta = counterfactual_effects[var_name]
                var_config = config.get_variable_config(var_name)

                # Group analysis by variable value
                var_values = eval_features_np[:, var_config['index']]
                low_mask = var_values < var_config['low_threshold']
                high_mask = var_values >= var_config['low_threshold']

                cf_stats.update({
                    f'{var_name.lower()}_delta_mean': delta.mean(),
                    f'{var_name.lower()}_delta_std': delta.std(),
                    f'{var_name.lower()}_delta_low_mean': delta[low_mask].mean() if low_mask.sum() > 0 else 0,
                    f'{var_name.lower()}_delta_high_mean': delta[high_mask].mean() if high_mask.sum() > 0 else 0,
                    f'{var_name.lower()}_low_count': low_mask.sum(),
                    f'{var_name.lower()}_high_count': high_mask.sum(),
                    f'{var_name.lower()}_monotonic_violation_rate': (delta[low_mask] < 0).mean() if low_mask.sum() > 0 else 0
                })
            
            return {
                'loss': loss,
                'predictions': predictions,
                'targets': targets,
                'counterfactual_effects': counterfactual_effects,
                'counterfactual_stats': cf_stats,
                'early_exposure_pred': output['early_exposure'][mask].cpu().numpy(),
                **metrics  # include all computed metrics
            }

    def train(self, num_epochs: int = 100, patience: int = 20, verbose: bool = True):
        """Train the model"""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['total_loss'])

            # Validate
            val_metrics = self.evaluate('val')
            val_loss = val_metrics['loss']
            val_mae = val_metrics['MAE']
            
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            
            if verbose and (epoch + 1) % config.PRINT_CONFIG['print_interval'] == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Total Loss: {train_metrics['total_loss']:.4f} | "
                      f"GroupDRO Loss: {train_metrics['groupdro_loss']:.4f} | "
                      f"Aux Loss: {train_metrics['aux_loss']:.4f} | "
                      f"Adv Loss: {train_metrics['adversarial_loss']:.4f} | "
                      f"Corr: {train_metrics['correlation_loss']:.4f} | "
                      f"CF Loss: {train_metrics['counterfactual_loss']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
                
                # Print GroupDRO info
                env_weights = train_metrics['env_weights']
                print(f"  Environment Weights: Low={env_weights[0]:.3f}, High={env_weights[1]:.3f}")

                env_losses = train_metrics['env_losses']
                print(f"  Environment Losses: Low={env_losses[0]:.4f}, High={env_losses[1]:.4f}")

                env_counts = train_metrics['env_counts']
                print(f"  Environment Counts: Low={env_counts[0]}, High={env_counts[1]}")

                # Print flexible counterfactual statistics
                cf_stats = train_metrics['counterfactual_stats']
                print(f"  Flexible Counterfactual Stats:")
                for var_name in self.model.enabled_variables:
                    var_lower = var_name.lower()
                    print(f"    {var_name}: Δ_mean={cf_stats.get(f'{var_lower}_delta_mean', 0):.4f}, "
                          f"Δ_std={cf_stats.get(f'{var_lower}_delta_std', 0):.4f}, "
                          f"Monotonic={cf_stats.get(f'{var_lower}_monotonic_loss', 0):.4f}")

            # Learning-rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_counterfactual_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load the best model
        self.model.load_state_dict(torch.load('best_counterfactual_model.pth'))
        return self.model

    def plot_training_curves(self):
        """Plot training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # MAE curve
        ax2.plot(self.val_maes, label='Val MAE', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Validation MAE')
        ax2.legend()
        ax2.grid(True)

        # Counterfactual loss curve
        if self.counterfactual_losses:
            ax3.plot(self.counterfactual_losses, label='Counterfactual Loss', color='green')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Counterfactual Loss')
            ax3.set_title('Counterfactual Loss')
            ax3.legend()
            ax3.grid(True)

        # ΔR statistics
        if self.delta_r_stats:
            delta_r_means = [stats['delta_r_mean'] for stats in self.delta_r_stats]
            ax4.plot(delta_r_means, label='ΔR Mean', color='red')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('ΔR Mean')
            ax4.set_title('Counterfactual Effect (ΔR)')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('gnn/pics/counterfactual_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def final_evaluation(self):
        """Final evaluation"""
        print("=" * 60)
        print("Final evaluation results of the counterfactual two-stage model")
        print("=" * 60)

        for split in ['val', 'test']:
            metrics = self.evaluate(split)
            print_detailed_metrics(metrics, split.upper())

            # Counterfactual analysis
            cf_stats = metrics['counterfactual_stats']
            print(f"  Counterfactual Analysis:")
            print(f"    ΔR Mean: {cf_stats['delta_r_mean']:.4f}")
            print(f"    ΔR Std: {cf_stats['delta_r_std']:.4f}")
            print(f"    ΔR for R=0: {cf_stats['delta_r_r0_mean']:.4f}")
            print(f"    ΔR for R=1: {cf_stats['delta_r_r1_mean']:.4f}")
            print(f"    R=0 samples: {cf_stats['r0_count']}")
            print(f"    R=1 samples: {cf_stats['r1_count']}")
            print(f"    Monotonic Violation Rate: {cf_stats['monotonic_violation_rate']:.4f}")

            # Prediction statistics
            preds = metrics['predictions']
            targets = metrics['targets']
            print(f"  Prediction range: [{preds.min():.2f}, {preds.max():.2f}]")
            print(f"  Target range: [{targets.min():.2f}, {targets.max():.2f}]")
            print(f"  Mean prediction: {preds.mean():.2f}")
            print(f"  Mean target: {targets.mean():.2f}")


def train_groupdro_counterfactual_model_standalone(year=5, save_dir=None):
    """Main function for standalone training of the GroupDRO + counterfactual two-stage model"""
    print("="*60)
    print("Standalone training of the GroupDRO + counterfactual two-stage HGT citation prediction model")
    print("="*60)

    # Set the random seed to make experiments reproducible
    config.set_random_seed()

    # Print a config summary
    config.print_config_summary()

    # Print the currently enabled counterfactual variables
    enabled_vars = config.get_enabled_variables()
    print(f"\n🔬 Currently enabled counterfactual variables ({len(enabled_vars)}):")
    for var_name in enabled_vars:
        var_config = config.get_variable_config(var_name)
        print(f"  ✅ {var_name}: {var_config['description']}")

    if len(enabled_vars) == 0:
        print("  ⚠️  Warning: no counterfactual variables are enabled!")

    print("="*60)

    # Get data paths from the config file
    data_paths = config.get_data_paths()

    # 1. Load data
    print("Loading data...")
    builder = load_separated_data_modified(
        data_paths['train'],
        data_paths['eval'],
        data_paths['test']
    )
    data = builder.build_hetero_data()

    # Print statistics
    stats = builder.get_statistics()
    print(f"Data statistics: {stats}")

    # 2. Create the model
    print("\nCreating the GroupDRO + counterfactual two-stage model...")
    model_params = config.get_model_params()

    # Create the GroupDRO + counterfactual two-stage model
    model = CounterfactualTwoStageHGT(**model_params)

    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Create venue labels (for analysis)
    def create_venue_labels(data, device, threshold=None):
        """Create venue labels (for analysis)"""
        if threshold is None:
            threshold = config.DATA_CONFIG['venue_threshold']

        venue_features = data['venue'].x
        venue_labels = (venue_features >= threshold).float().squeeze()

        # Map venue labels onto papers
        paper_venue_edges = data['paper', 'published_in', 'venue'].edge_index
        paper_venue_labels = torch.zeros(data['paper'].x.size(0), device=device)

        for paper_idx, venue_idx in paper_venue_edges.t():
            paper_venue_labels[paper_idx] = venue_labels[venue_idx]

        print(f"Venue label statistics: high-tier proportion={paper_venue_labels.mean():.3f}")
        return paper_venue_labels

    # 4. Create the GroupDRO + counterfactual trainer
    print("\nCreating the GroupDRO + counterfactual trainer...")
    training_params = config.get_training_params()
    device = training_params['device']
    venue_labels = create_venue_labels(data, device)

    trainer = GroupDROCounterfactualTrainer(
        model=model,
        data=data,
        device=device,
        venue_labels=venue_labels
    )

    # 5. Train the model
    print("\nStarting GroupDRO + counterfactual training...")
    trainer.train(
        num_epochs=training_params['num_epochs'],
        patience=training_params['patience'],
        verbose=training_params['verbose']
    )

    # 6. Evaluate the model
    print("\nEvaluating the model...")
    val_metrics = trainer.evaluate('val')
    test_metrics = trainer.evaluate('test')

    # 7. Print detailed results
    print("\n" + "="*50)
    print("Final evaluation results")
    print("="*50)
    print_detailed_metrics(val_metrics, "Validation set")
    print_detailed_metrics(test_metrics, "Test set")

    # 8. Flexible counterfactual analysis
    print("\n" + "="*50)
    print("Flexible multi-variable counterfactual analysis")
    print("="*50)
    for split_name, metrics in [('Validation set', val_metrics), ('Test set', test_metrics)]:
        print(f"\n{split_name}:")
        cf_stats = metrics['counterfactual_stats']

        for var_name in model.enabled_variables:
            var_config = config.get_variable_config(var_name)
            var_lower = var_name.lower()

            print(f"  {var_name} ({var_config['name']}):")
            print(f"    Δ{var_name} ({var_config['description']}): {cf_stats[f'{var_lower}_delta_mean']:.4f} ± {cf_stats[f'{var_lower}_delta_std']:.4f}")
            print(f"    For papers with {var_name}=low: Δ{var_name} = {cf_stats[f'{var_lower}_delta_low_mean']:.4f}")
            print(f"    For papers with {var_name}=high: Δ{var_name} = {cf_stats[f'{var_lower}_delta_high_mean']:.4f}")
            print(f"    Monotonicity violation rate: {cf_stats[f'{var_lower}_monotonic_violation_rate']:.4f}")

    # 9. Save results
    results = {
        'model': model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }
    
    # Save the model
    if save_dir is None:
        model_save_path = 'gnn/saved_models/groupdro_counterfactual_two_stage_model.pth'
    else:
        model_save_path = f'{save_dir}/model_year{year}.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'env_weights_history': trainer.env_weights_history,
        'env_losses': trainer.env_losses,
        'counterfactual_losses': trainer.counterfactual_losses,
        'delta_r_stats': trainer.delta_r_stats
    }, model_save_path)
    
    # Save metrics to a JSON file (only the test-set metrics)
    if save_dir is None:
        metrics_save_path = f'gnn/results/openalex_2/year_{year}.json'
    else:
        metrics_save_path = f'{save_dir}/year_{year}.json'

    metrics_to_save = {
        'test_metrics': test_metrics  # only save the test-set metrics
    }

    # Ensure the metrics are serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float)) and np.isnan(obj):
            return None
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # For other types, try converting to a string
            try:
                return str(obj)
            except:
                return None

    metrics_serializable = make_serializable(metrics_to_save)

    # Make sure the directory exists
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False)

    # Separately save the config information to a config file
    if save_dir is None:
        config_save_path = f'gnn/results/openalex_2/config_year{year}.json'
    else:
        config_save_path = f'{save_dir}/config_year{year}.json'
    config_to_save = config.get_all_config()

    # Add a summary of experiment results to the config
    config_to_save['experiment_summary'] = {
        'final_val_mae': val_metrics['MAE'],
        'final_test_mae': test_metrics['MAE'],
        'final_val_rmse': val_metrics['RMSE'],
        'final_test_rmse': test_metrics['RMSE'],
        'final_val_r2': val_metrics['R2'],
        'final_test_r2': test_metrics['R2'],
        'training_epochs': len(trainer.train_losses),
        'best_val_loss': min(trainer.val_losses) if trainer.val_losses else None,
        'counterfactual_effects': {
            var_name: {
                'val_delta_mean': val_metrics['counterfactual_stats'].get(f'{var_name.lower()}_delta_mean', 0),
                'test_delta_mean': test_metrics['counterfactual_stats'].get(f'{var_name.lower()}_delta_mean', 0),
                'val_monotonic_violation': val_metrics['counterfactual_stats'].get(f'{var_name.lower()}_monotonic_violation_rate', 0),
                'test_monotonic_violation': test_metrics['counterfactual_stats'].get(f'{var_name.lower()}_monotonic_violation_rate', 0)
            }
            for var_name in model.enabled_variables
        }
    }
    
    config_serializable = make_serializable(config_to_save)
    
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\nModel saved to: {model_save_path}")
    print(f"Test-set evaluation metrics saved to: {metrics_save_path}")
    print(f"Config information saved to: {config_save_path}")
    print("GroupDRO + counterfactual training complete!")

    return results


if __name__ == "__main__":
    print("GroupDRO + flexible multi-variable counterfactual two-stage HGT citation prediction model is ready!")
    print("\nMain features:")
    print("1. Two-stage architecture: V->E->Y path")
    print("2. Early-exposure latent-variable learning (stage A, no ground-truth labels needed)")
    print("3. Citation prediction (stage B, does not directly use V)")
    print("4. Quartile-prediction auxiliary head (regularization)")
    print("5. GroupDRO training: addresses the 'top-venue halo' problem via environment-split training")
    print("6. Adversarial debiasing: reduces venue information leakage")
    print("7. Flexible counterfactual prediction: manually choose among the R, Q, C, H, A1, A2, A3 variables")
    print("   - R: reproducibility")
    print("   - Q: content_quality")
    print("   - C: collaboration_network")
    print("   - H: hotness (topic hotness)")
    print("   - A1: first_author_reputation")
    print("   - A2: last_author_reputation")
    print("   - A3: other_authors_reputation")
    print("8. Monotonicity constraint: ensures that raising a variable from low->high does not lower the citation count")

    # Print the current config
    print("\n" + "="*60)
    print("Current counterfactual variable config:")
    print("="*60)
    config.print_available_variables()

    # Config example
    print("\n📝 Config example:")
    print("To enable/disable variables, modify config.COUNTERFACTUAL_CONFIG['enabled_variables'] before running")
    print("For example:")
    print("  config.enable_variable('H', True)   # enable topic hotness")
    print("  config.enable_variable('A1', True)  # enable first-author reputation")
    print("  config.enable_variable('C', False)  # disable the collaboration network")

    print("\n" + "="*60)
    print("Starting GroupDRO + flexible counterfactual training...")
    print("="*60)

    # Run GroupDRO + counterfactual training
    results = train_groupdro_counterfactual_model_standalone()
