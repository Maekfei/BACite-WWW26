"""
Two-stage HGT citation prediction model (latent-variable version)
Implements the V->E->Y path: the venue influences final citations Y through early exposure E

Architecture design:
- Stage A: learn the early-exposure latent variable E_hat = g_phi([z_p; A,C,V,H,Q,R])
- Stage B: predict citations Y = f_theta([z_p; A,C,H,Q,R,E_hat]) (V does not enter directly)
- Auxiliary head: use E_hat to predict the quartile of Y, as a regularizer

Key improvements:
- E is a latent variable, no ground-truth labels required
- Main-task loss + lightweight auxiliary loss
- Preserves the V->E->Y causal structure
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
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

# Import configuration file
from counterfactual_config_oa import config


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics (kept consistent with two_stage_hgt_counterfactual.py)."""
    # Basic regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Log-scale error metrics (implementation kept consistent with run.py)
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
            print(f"⚠️ Error while computing NDCG@{k}: {e}")
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
            # Use the 75th percentile as the threshold for "high value"
            threshold = np.percentile(y_true, 75)
            high_value_indices = np.where(y_true >= threshold)[0]
            
            # Check whether the top k predictions contain high-value samples
            hit_count = len(set(top_k_indices) & set(high_value_indices))
            return hit_count / k if k > 0 else 0.0
        except Exception as e:
            print(f"⚠️ Error while computing Hit@{k}: {e}")
            return 0.0
    
    # Evaluate ranking metrics using K = 10 and 20
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
    """Print detailed evaluation metrics (kept consistent with two_stage_hgt_counterfactual.py)."""
    print("=" * 60)
    print(f"📊 {split_name} evaluation metrics (two-stage HGT):")
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
    print(f"Hit@10  : {metrics['Hit@10']:.6f} (fraction of high-value papers in the top 10)")
    print(f"Hit@20  : {metrics['Hit@20']:.6f} (fraction of high-value papers in the top 20)")
    print("=" * 60)


@dataclass
class PaperFeatures:
    """Paper node features."""
    paper_id: str
    year: int
    reproducibility: float  # 0-1
    content_quality: float  # 1-5
    collaboration_network: float  # 1-5
    hotness: float  # sum of last-year hotness over all topics this paper involves
    first_author_reputation: float  # first author reputation value
    last_author_reputation: float  # last author reputation value
    other_authors_reputation: float  # other authors reputation value
    citation_count: int
    split: str  # 'train', 'eval', 'test'

@dataclass
class AuthorFeatures:
    """Author node features."""
    author_id: str
    name: str
    paper_count: int  # total number of papers
    
@dataclass
class VenueFeatures:
    """Venue node features."""
    venue_id: str
    name: str
    tier: float  # 1-5
    
@dataclass
class TopicFeatures:
    """Topic node features."""
    topic_id: str
    keyword: str


class ModifiedHeterographDataBuilder:
    """Modified heterogeneous graph data builder - adds author reputation features and removes edge weights."""
    
    def __init__(self):
        self.paper_features = []
        self.author_features = []
        self.venue_features = []
        self.topic_features = []
        
        # Mapping dictionaries
        self.paper_id_to_idx = {}
        self.author_id_to_idx = {}
        self.venue_id_to_idx = {}
        self.topic_id_to_idx = {}
        
        # Edge lists - weights removed
        self.paper_author_edges = []  # stores (paper_idx, author_idx) tuples
        self.paper_venue_edges = []
        self.paper_topic_edges = []
        self.paper_cite_edges = []
        self.author_venue_edges = []  # stores (author_idx, venue_idx) tuples
    
    def add_paper(self, paper_data: Dict[str, Any], split: str):
        """Add a paper node - modified version that adds author reputation features."""
        paper_id = paper_data['id']
        
        if paper_id not in self.paper_id_to_idx:
            paper_idx = len(self.paper_features)
            self.paper_id_to_idx[paper_id] = paper_idx
            
            # Extract and clean paper features (clamp/binarize at load time)
            def _to_float(value, default=0.0):
                try:
                    return float(value)
                except Exception:
                    return default

            def _clamp_0_5(value):
                v = _to_float(value, 0.0)
                if v < 0.0:
                    return 0.0
                if v > 5.0:
                    return 5.0
                return v

            # binarize reproducibility (set to 1 if >0, else 0)
            repro_raw = _to_float(paper_data.get('reproducibility'), 0.0)
            repro_bin = 1.0 if repro_raw > 0.0 else 0.0

            content_quality_f = _clamp_0_5(paper_data.get('content_quality'))
            collaboration_network_f = _clamp_0_5(paper_data.get('collaboration_network'))
            hotness_f = _clamp_0_5(paper_data.get('field_hotness'))
            a1_f = _clamp_0_5(paper_data.get('author1_reputation'))
            a2_f = _clamp_0_5(paper_data.get('author2_reputation'))
            a3_f = _clamp_0_5(paper_data.get('other_authors_reputation'))

            paper_feature = PaperFeatures(
                paper_id=paper_id,
                year=paper_data.get('year', 2010),
                reproducibility=repro_bin,
                content_quality=content_quality_f,
                collaboration_network=collaboration_network_f,
                hotness=hotness_f,
                first_author_reputation=a1_f,
                last_author_reputation=a2_f,
                other_authors_reputation=a3_f,
                citation_count=paper_data.get(f'citations_year_{config.get_current_year()}'), # here
                split=split
            )
            self.paper_features.append(paper_feature)
            
            # Add author relations - weights removed
            authors = paper_data.get('authors', [])
            author_indices = []  # store author indices, used later to add author-venue edges
            for i, author_info in enumerate(authors):
                author_idx = self._add_author(author_info)
                author_indices.append(author_idx)
                # weights removed, only the edge index is stored
                self.paper_author_edges.append((paper_idx, author_idx))
            
            # Add venue relation
            venue_idx = self._add_venue(
                paper_data.get('venue', ''),
                _clamp_0_5(paper_data.get('venue_score'))
            )
            self.paper_venue_edges.append((paper_idx, venue_idx))
            
            # Add author-venue relations
            for author_idx in author_indices:
                self.author_venue_edges.append((author_idx, venue_idx))
            
            # Add topic relations
            for keyword in paper_data.get('keywords', []):
                topic_idx = self._add_topic(keyword)
                self.paper_topic_edges.append((paper_idx, topic_idx))
            
            # Add citation relations
            for ref_id in paper_data.get('references', []):
                if ref_id in self.paper_id_to_idx:
                    ref_idx = self.paper_id_to_idx[ref_id]
                    self.paper_cite_edges.append((paper_idx, ref_idx))
    
    def _add_author(self, author_info: Dict[str, str]) -> int:
        """Add an author node."""
        author_id = author_info.get('id', author_info.get('name', 'unknown'))
        
        if author_id not in self.author_id_to_idx:
            author_idx = len(self.author_features)
            self.author_id_to_idx[author_id] = author_idx
            
            author_feature = AuthorFeatures(
                author_id=author_id,
                name=author_info.get('name', 'unknown'),
                paper_count=1  # initial paper count is 1
            )
            self.author_features.append(author_feature)
            return author_idx
        else:
            # Update the paper count of an existing author (+1)
            existing_idx = self.author_id_to_idx[author_id]
            self.author_features[existing_idx].paper_count += 1
            return existing_idx
    
    def _add_venue(self, venue_name: str, tier: float) -> int:
        """Add a venue node."""
        venue_id = venue_name.lower().replace(' ', '_') if venue_name else 'unknown'
        
        if venue_id not in self.venue_id_to_idx:
            venue_idx = len(self.venue_features)
            self.venue_id_to_idx[venue_id] = venue_idx
            
            venue_feature = VenueFeatures(
                venue_id=venue_id,
                name=venue_name,
                tier=tier
            )
            self.venue_features.append(venue_feature)
            return venue_idx
        else:
            return self.venue_id_to_idx[venue_id]
    
    def _add_topic(self, keyword: str) -> int:
        """Add a topic node."""
        topic_id = keyword.lower().replace(' ', '_') if keyword else 'unknown'
        
        if topic_id not in self.topic_id_to_idx:
            topic_idx = len(self.topic_features)
            self.topic_id_to_idx[topic_id] = topic_idx
            
            topic_feature = TopicFeatures(
                topic_id=topic_id,
                keyword=keyword
            )
            self.topic_features.append(topic_feature)
            return topic_idx
        else:
            return self.topic_id_to_idx[topic_id]
    
    def build_hetero_data(self, include_venue: bool = True) -> HeteroData:
        """Build heterogeneous graph data - supports building the full or simplified graph."""
        data = HeteroData()
        
        # First compute hotness statistics for standardization
        hotness_values = [pf.hotness for pf in self.paper_features]
        hotness_mean = sum(hotness_values) / len(hotness_values) if hotness_values else 0.0
        hotness_std = (sum((x - hotness_mean) ** 2 for x in hotness_values) / len(hotness_values)) ** 0.5 if hotness_values else 1.0
        # Avoid division-by-zero errors
        if hotness_std == 0.0:
            hotness_std = 1.0
        
        # Build node features
        # Paper node features: [reproducibility, content_quality, collaboration_network, hotness_standardized, year_normalized, first_author_reputation, last_author_reputation, other_authors_reputation]
        paper_x = []
        paper_y = []
        paper_years = []
        paper_splits = []
        
        for pf in self.paper_features:
            # Normalize the year to the 0-1 range (assuming 2010-2025)
            year_norm = (pf.year - 2010) / (2025 - 2010)
            # standardize hotness
            hotness_normalized = (pf.hotness - hotness_mean) / hotness_std
            paper_x.append([
                pf.reproducibility / 2,
                pf.content_quality / 5.0,  # normalize to 0-1
                pf.collaboration_network / 5.0,  # normalize to 0-1
                hotness_normalized,  # standardized hotness value
                year_norm,
                pf.first_author_reputation / 5.0,  # normalize to 0-1
                pf.last_author_reputation / 5.0,   # normalize to 0-1
                pf.other_authors_reputation / 5.0  # normalize to 0-1
            ])
            paper_y.append(pf.citation_count)
            paper_years.append(pf.year)
            paper_splits.append(pf.split)
        
        data['paper'].x = torch.FloatTensor(paper_x)
        data['paper'].y = torch.LongTensor(paper_y)
        data['paper'].year = torch.LongTensor(paper_years)
        data['paper'].split = paper_splits  # save the split information
        
        # Author node features: [paper_count] - normalized paper count
        max_paper_count = max([af.paper_count for af in self.author_features]) if self.author_features else 1
        author_x = [[af.paper_count / max_paper_count] for af in self.author_features]
        data['author'].x = torch.FloatTensor(author_x)
        
        # Venue node features: [tier] - only added when include_venue=True
        if include_venue:
            venue_x = [[vf.tier / 5.0] for vf in self.venue_features]
            data['venue'].x = torch.FloatTensor(venue_x)
        
        # Topic node features: [1.0] - topic nodes have no extra features, use a constant feature
        topic_x = [[1.0] for _ in self.topic_features]
        data['topic'].x = torch.FloatTensor(topic_x)
        
        # Build edges - whether to include venue-related edges depends on the include_venue argument
        if self.paper_author_edges:
            # Use the edge index directly, without weights
            paper_author_edge_index = torch.LongTensor(self.paper_author_edges).t().contiguous()
            
            data['paper', 'written_by', 'author'].edge_index = paper_author_edge_index
            data['author', 'writes', 'paper'].edge_index = paper_author_edge_index.flip([0])
        
        # Venue-related edges - only added when include_venue=True
        if include_venue:
            if self.paper_venue_edges:
                paper_venue_edge_index = torch.LongTensor(self.paper_venue_edges).t().contiguous()
                data['paper', 'published_in', 'venue'].edge_index = paper_venue_edge_index
                data['venue', 'publishes', 'paper'].edge_index = paper_venue_edge_index.flip([0])
            
            if self.author_venue_edges:
                author_venue_edge_index = torch.LongTensor(self.author_venue_edges).t().contiguous()
                data['author', 'publishes_in', 'venue'].edge_index = author_venue_edge_index
                data['venue', 'has_author', 'author'].edge_index = author_venue_edge_index.flip([0])
        
        if self.paper_topic_edges:
            paper_topic_edge_index = torch.LongTensor(self.paper_topic_edges).t().contiguous()
            data['paper', 'has_topic', 'topic'].edge_index = paper_topic_edge_index
            data['topic', 'topic_of', 'paper'].edge_index = paper_topic_edge_index.flip([0])
        
        if self.paper_cite_edges:
            paper_cite_edge_index = torch.LongTensor(self.paper_cite_edges).t().contiguous()
            data['paper', 'cites', 'paper'].edge_index = paper_cite_edge_index
        
        return data
    
    def get_statistics(self):
        """Get graph statistics."""
        return {
            'num_papers': len(self.paper_features),
            'num_authors': len(self.author_features),
            'num_venues': len(self.venue_features),
            'num_topics': len(self.topic_features),
            'num_paper_author_edges': len(self.paper_author_edges),
            'num_paper_venue_edges': len(self.paper_venue_edges),
            'num_paper_topic_edges': len(self.paper_topic_edges),
            'num_author_venue_edges': len(self.author_venue_edges),
            'num_citation_edges': len(self.paper_cite_edges)
        }


class TwoStageCitationHGT(nn.Module):
    """Two-stage HGT citation prediction model (true two-stage-graph version)."""
    
    def __init__(self, 
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 node_type_to_feature_dim: Dict[str, int],
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 use_negative_binomial: bool = False,
                 use_adversarial: bool = False,
                 aux_loss_weight: float = 0.003):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.use_negative_binomial = use_negative_binomial
        self.use_adversarial = use_adversarial
        self.aux_loss_weight = aux_loss_weight
        
        # Define the different graph structures for the two stages
        # Stage A: full graph (includes venue)
        self.stage_a_node_types = node_types  # ['paper', 'author', 'venue', 'topic']
        self.stage_a_edge_types = edge_types  # includes all edge types
        
        # Stage B: simplified graph (venue nodes and edges removed)
        self.stage_b_node_types = [nt for nt in node_types if nt != 'venue']  # ['paper', 'author', 'topic']
        self.stage_b_edge_types = [et for et in edge_types if 'venue' not in et]  # remove venue-related edges
        
        # Input projection layers (shared by both stages)
        self.input_projections = nn.ModuleDict()
        for node_type, feature_dim in node_type_to_feature_dim.items():
            self.input_projections[node_type] = Linear(feature_dim, hidden_dim)
        
        # Stage A HGT layers (full-graph encoder)
        self.stage_a_hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.stage_a_hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(self.stage_a_node_types, self.stage_a_edge_types),
                    heads=num_heads
                )
            )
        
        # Stage B HGT layers (simplified-graph encoder)
        self.stage_b_hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.stage_b_hgt_layers.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(self.stage_b_node_types, self.stage_b_edge_types),
                    heads=num_heads
                )
            )
        
        # Stage A: early-exposure latent-variable learner
        # Input: [z_p; A,C,V,H,Q,R] - includes all features
        self.early_exposure_encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + 8, hidden_dim // 2),  # +8 for paper features (R,Q,C,H,Y,A1,A2,A3)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # learn the early-exposure latent variable
        )
        
        # Stage B: citation prediction head
        # Input: [z_p; A,C,H,Q,R,E_hat] - excludes V but includes E_hat
        self.citation_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim + 9, hidden_dim // 2),  # +6 for A,C,H,Q,R,Ê
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2 if use_negative_binomial else 1)
        )
        
        # Auxiliary head: use E_hat to predict the quartile of Y (regularization)
        self.quartile_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1, hidden_dim // 4),  # input is E_hat
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 4)  # predict the 4 quartiles
        )
        
        # Adversarial debiasing module (optional)
        if use_adversarial:
            self.venue_discriminator = nn.Sequential(
                nn.Linear(hidden_dim + 9, hidden_dim // 2),  # +6 for A,C,H,Q,R,E_hat (excludes V)
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)  # predict whether it is a high-tier venue
            )
    
    def extract_paper_features(self, paper_embeddings, paper_features):
        """Extract the extra feature vector of a paper."""
        # paper_features: [R,Q,C,H,Y,A1,A2,A3] - 8-dim features from CP_HGT_s1
        # R: reproducibility, Q: content_quality, C: collaboration_network, H: hotness
        # Y: year_norm, A1: first_author_reputation, A2: last_author_reputation, A3: other_authors_reputation
        return paper_features
    
    def forward(self, x_dict, edge_index_dict, paper_features=None):
        """Forward pass - true two-stage-graph architecture."""
        # Input projection
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.input_projections[node_type](x)
        
        if paper_features is None:
            # If no extra features are provided, use a zero vector
            batch_size = h_dict['paper'].size(0)
            paper_features = torch.zeros(batch_size, 8, device=h_dict['paper'].device)
        
        # ===== Stage A: use the full graph to learn the early-exposure latent variable =====
        # Stage A HGT layers (includes venue information)
        stage_a_h_dict = h_dict.copy()
        for hgt_layer in self.stage_a_hgt_layers:
            stage_a_h_dict = hgt_layer(stage_a_h_dict, edge_index_dict)
        
        # Get Stage A paper embeddings (includes venue information)
        stage_a_paper_embeddings = stage_a_h_dict['paper']
        
        # Stage A: learn the early-exposure latent variable E_hat = g_phi([z_p; A,C,V,H,Q,R])
        stage_a_input = torch.cat([stage_a_paper_embeddings, paper_features], dim=1)
        early_exposure = self.early_exposure_encoder(stage_a_input)
        
        # ===== Stage B: use the simplified graph to predict citations =====
        # Build the simplified-graph input (remove venue nodes)
        stage_b_x_dict = {k: v for k, v in x_dict.items() if k != 'venue'}
        stage_b_h_dict = {k: v for k, v in h_dict.items() if k != 'venue'}
        
        # Build the simplified-graph edge index (remove venue-related edges)
        stage_b_edge_index_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if 'venue' not in edge_type:
                stage_b_edge_index_dict[edge_type] = edge_index
        
        # Stage B HGT layers (does not include venue information)
        for hgt_layer in self.stage_b_hgt_layers:
            stage_b_h_dict = hgt_layer(stage_b_h_dict, stage_b_edge_index_dict)
        
        # Get Stage B paper embeddings (does not include venue information)
        stage_b_paper_embeddings = stage_b_h_dict['paper']
        
        # Stage B: predict citations Y = f_theta([z_p; A,C,H,Q,R,E_hat])
        # Note: V is not included here, but E_hat is
        stage_b_features = torch.cat([
            paper_features,  # all 8 features [R,Q,C,H,Y,A1,A2,A3]
            early_exposure   # Ê
        ], dim=1)
        stage_b_input = torch.cat([stage_b_paper_embeddings, stage_b_features], dim=1)
        
        if self.use_negative_binomial:
            citation_params = self.citation_predictor(stage_b_input)
            log_mu, log_alpha = citation_params[:, 0], citation_params[:, 1]
            mu = torch.exp(log_mu)
            alpha = torch.exp(log_alpha)
            citation_output = {'mu': mu, 'alpha': alpha}
        else:
            log_citation = self.citation_predictor(stage_b_input).squeeze()
            citation_output = {'log_citation': log_citation}
        
        # Auxiliary prediction: use E_hat to predict the quartile of Y
        quartile_prediction = self.quartile_predictor(early_exposure)
        
        # Adversarial debiasing (optional)
        adversarial_output = None
        if self.use_adversarial:
            # The discriminator tries to identify venue information from the Stage B input
            adversarial_output = self.venue_discriminator(stage_b_input)
        
        return {
            'early_exposure': early_exposure.squeeze(),
            'citation': citation_output,
            'quartile_prediction': quartile_prediction,
            'embeddings': stage_b_paper_embeddings,  # use Stage B paper embeddings (no venue information)
            'stage_a_embeddings': stage_a_paper_embeddings,  # Stage A paper embeddings (includes venue information)
            'adversarial': adversarial_output
        }
    
    def negative_binomial_loss(self, mu, alpha, targets):
        """Negative binomial loss function."""
        targets = targets.float()
        
        log_likelihood = (
            torch.lgamma(targets + alpha) 
            - torch.lgamma(targets + 1) 
            - torch.lgamma(alpha)
            + alpha * torch.log(alpha / (alpha + mu))
            + targets * torch.log(mu / (alpha + mu))
        )
        
        return -log_likelihood.mean()
    
    def mse_log_loss(self, log_predictions, targets):
        """MSE log loss function."""
        log_targets = torch.log1p(targets.float())
        return F.mse_loss(log_predictions, log_targets)
    
    def quartile_loss(self, quartile_predictions, citation_targets):
        """Quartile prediction loss function (auxiliary loss)."""
        if citation_targets is None:
            return torch.tensor(0.0, device=quartile_predictions.device)
        
        # Compute the quartiles of the citation counts
        citation_targets = citation_targets.float()
        q1 = torch.quantile(citation_targets, 0.25)
        q2 = torch.quantile(citation_targets, 0.5)
        q3 = torch.quantile(citation_targets, 0.75)
        
        # Convert citation counts into quartile labels
        quartile_labels = torch.zeros_like(citation_targets, dtype=torch.long)
        quartile_labels[citation_targets <= q1] = 0
        quartile_labels[(citation_targets > q1) & (citation_targets <= q2)] = 1
        quartile_labels[(citation_targets > q2) & (citation_targets <= q3)] = 2
        quartile_labels[citation_targets > q3] = 3
        
        # Cross-entropy loss
        return F.cross_entropy(quartile_predictions, quartile_labels)
    
    def adversarial_loss(self, adversarial_output, venue_labels):
        """Adversarial loss function."""
        if adversarial_output is None or venue_labels is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Discriminator loss: tries to correctly identify the venue tier
        discriminator_loss = F.binary_cross_entropy_with_logits(
            adversarial_output.squeeze(), venue_labels.float()
        )
        return discriminator_loss


class TwoStageTrainer:
    """Trainer for the two-stage model."""
    
    def __init__(self, 
                 model: TwoStageCitationHGT,
                 data: HeteroData,
                 device: str = None,
                 venue_labels: Optional[torch.Tensor] = None):
        # Use the defaults from the configuration file
        self.device = device or config.get_training_params()['device']
        
        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.venue_labels = venue_labels
        
        # Create the data splits
        self._create_splits()
        
        # Optimizer - uses parameters from the configuration file
        optimizer_params = config.get_optimizer_params()
        self.optimizer = AdamW(
            model.parameters(), 
            lr=optimizer_params['lr'], 
            weight_decay=optimizer_params['weight_decay']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.aux_losses = []
        self.adversarial_losses = []
        self.correlation_losses = []
    
    def _create_splits(self):
        """Create the data splits."""
        paper_splits = self.data['paper'].split
        
        self.train_mask = torch.tensor([split == 'train' for split in paper_splits])
        self.val_mask = torch.tensor([split == 'eval' for split in paper_splits])
        self.test_mask = torch.tensor([split == 'test' for split in paper_splits])
        
        print(f"Train samples: {self.train_mask.sum().item()}")
        print(f"Val samples: {self.val_mask.sum().item()}")
        print(f"Test samples: {self.test_mask.sum().item()}")
    
    def _get_paper_features(self, mask):
        """Get the paper feature vector [R,Q,C,H,Y,A1,A2,A3] - directly uses the 8-dim features from CP_HGT_s1."""
        # Directly fetch the 8-dim paper features from the data
        # [reproducibility/2, content_quality/5, collaboration_network/5, hotness, year_norm, first_author_reputation/5, last_author_reputation/5, other_authors_reputation/5]
        paper_x = self.data['paper'].x[mask]  # already 8-dim features
        
        return paper_x
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        self.optimizer.zero_grad()
        
        # Forward pass - generate features for all papers
        all_features = self._get_paper_features(torch.ones(self.data['paper'].x.size(0), dtype=torch.bool))
        output = self.model(
            self.data.x_dict, 
            self.data.edge_index_dict, 
            all_features
        )
        
        # Compute the loss
        train_targets = self.data['paper'].y[self.train_mask]
        
        # Main loss: citation prediction (computed on the training set only)
        if self.model.use_negative_binomial:
            mu = output['citation']['mu'][self.train_mask]
            alpha = output['citation']['alpha'][self.train_mask]
            main_loss = self.model.negative_binomial_loss(mu, alpha, train_targets)
        else:
            log_predictions = output['citation']['log_citation'][self.train_mask]
            main_loss = self.model.mse_log_loss(log_predictions, train_targets)
        
        # Auxiliary loss: quartile prediction (computed on the training set only)
        aux_loss = self.model.quartile_loss(
            output['quartile_prediction'][self.train_mask], 
            train_targets
        )
        
        # Adversarial loss
        adversarial_loss = torch.tensor(0.0, device=self.device)
        if self.model.use_adversarial and self.venue_labels is not None:
            adversarial_loss = self.model.adversarial_loss(
                output['adversarial'][self.train_mask] if output['adversarial'] is not None else None, 
                self.venue_labels[self.train_mask]
            )
        
        # Correlation constraint loss: encourages the early-exposure latent variable to correlate positively with citation counts
        correlation_loss = torch.tensor(0.0, device=self.device)
        if len(train_targets) > 1:  # need at least 2 samples to compute correlation
            early_exposure_train = output['early_exposure'][self.train_mask]
            targets_normalized = (train_targets.float() - train_targets.float().mean()) / (train_targets.float().std() + 1e-8)
            exposure_normalized = (early_exposure_train - early_exposure_train.mean()) / (early_exposure_train.std() + 1e-8)
            
            # Compute the Pearson correlation coefficient and encourage positive correlation
            correlation = torch.mean(targets_normalized * exposure_normalized)
            correlation_loss = -correlation  # negative sign: maximize positive correlation
        
        # Total loss
        #total_loss = main_loss + self.model.aux_loss_weight * aux_loss + 0.1 * adversarial_loss + 0.001 * correlation_loss
        #total_loss = main_loss + 0.1 * adversarial_loss + 0.001 * correlation_loss
        total_loss = main_loss + self.model.aux_loss_weight * aux_loss + 0.1 * adversarial_loss + correlation_loss
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'aux_loss': aux_loss.item(),
            'adversarial_loss': adversarial_loss.item(),
            'correlation_loss': correlation_loss.item()
        }
    
    def evaluate(self, split: str = 'val'):
        """Evaluate the model."""
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
            
            # Citation prediction evaluation (uses only the data of the corresponding split)
            if self.model.use_negative_binomial:
                predictions = output['citation']['mu'][mask].cpu().numpy()
                loss = self.model.negative_binomial_loss(
                    output['citation']['mu'][mask], 
                    output['citation']['alpha'][mask], 
                    self.data['paper'].y[mask]
                ).item()
            else:
                log_predictions = output['citation']['log_citation'][mask].cpu().numpy()
                predictions = np.expm1(log_predictions)
                loss = self.model.mse_log_loss(
                    output['citation']['log_citation'][mask],
                    self.data['paper'].y[mask]
                ).item()
            
            # Compute the full metrics (kept consistent with two_stage_hgt_counterfactual.py)
            metrics = calculate_metrics(targets, predictions)
            
            # Auxiliary-task evaluation: quartile prediction accuracy (uses only the data of the corresponding split)
            quartile_pred = output['quartile_prediction'][mask].cpu().numpy()
            quartile_pred_labels = np.argmax(quartile_pred, axis=1)
            
            # Compute the ground-truth quartile labels
            citation_targets = targets
            q1 = np.quantile(citation_targets, 0.25)
            q2 = np.quantile(citation_targets, 0.5)
            q3 = np.quantile(citation_targets, 0.75)
            
            quartile_true_labels = np.zeros_like(citation_targets, dtype=int)
            quartile_true_labels[citation_targets <= q1] = 0
            quartile_true_labels[(citation_targets > q1) & (citation_targets <= q2)] = 1
            quartile_true_labels[(citation_targets > q2) & (citation_targets <= q3)] = 2
            quartile_true_labels[citation_targets > q3] = 3
            
            quartile_accuracy = (quartile_pred_labels == quartile_true_labels).mean()
            
            # Debug information
            if split == 'val':
                print(f"Debug info - quartile prediction:")
                print(f"  Predicted label distribution: {np.bincount(quartile_pred_labels)}")
                print(f"  Ground-truth label distribution: {np.bincount(quartile_true_labels)}")
                print(f"  Quartile thresholds: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
                print(f"  Accuracy: {quartile_accuracy:.4f}")
            
            return {
                'loss': loss,
                'predictions': predictions,
                'targets': targets,
                'quartile_accuracy': quartile_accuracy,
                'early_exposure_pred': output['early_exposure'][mask].cpu().numpy(),
                **metrics  # includes all computed metrics
            }
    
    def train(self, num_epochs: int = 100, patience: int = 10, verbose: bool = True):
        """Train the model."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['total_loss'])
            self.aux_losses.append(train_metrics['aux_loss'])
            self.adversarial_losses.append(train_metrics['adversarial_loss'])
            self.correlation_losses.append(train_metrics['correlation_loss'])
            
            # Validate
            val_metrics = self.evaluate('val')
            val_loss = val_metrics['loss']
            val_mae = val_metrics['MAE']  # use MAE on the original scale, consistent with the counterfactual version
            
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Total Loss: {train_metrics['total_loss']:.4f} | "
                      f"Main Loss: {train_metrics['main_loss']:.4f} | "
                      f"Aux Loss: {train_metrics['aux_loss']:.4f} | "
                      f"Adv: {train_metrics['adversarial_loss']:.4f} | "
                      f"Corr: {train_metrics['correlation_loss']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_two_stage_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model
        self.model.load_state_dict(torch.load('best_two_stage_model.pth'))
        return self.model
    
    def plot_training_curves(self):
        """Plot the training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE
        axes[0, 1].plot(self.val_maes, label='Val MAE', color='orange')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('Validation MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Auxiliary loss
        axes[1, 0].plot(self.aux_losses, label='Auxiliary Loss', color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Auxiliary Loss (Quartile Prediction)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Adversarial loss
        axes[1, 1].plot(self.adversarial_losses, label='Adversarial Loss', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Adversarial Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def final_evaluation(self):
        """Final evaluation."""
        print("=" * 60)
        print("Final evaluation results of the two-stage HGT model")
        print("=" * 60)
        
        for split in ['val', 'test']:
            metrics = self.evaluate(split)
            print_detailed_metrics(metrics, split.upper())
            
            # Prediction statistics
            preds = metrics['predictions']
            targets = metrics['targets']
            print(f"  Citation Prediction range: [{preds.min():.2f}, {preds.max():.2f}]")
            print(f"  Citation Target range: [{targets.min():.2f}, {targets.max():.2f}]")
            print(f"  Mean Citation prediction: {preds.mean():.2f}")
            print(f"  Mean Citation target: {targets.mean():.2f}")
            
            if 'quartile_accuracy' in metrics:
                print(f"  Quartile Prediction Accuracy: {metrics['quartile_accuracy']:.4f}")


# Data loading function - kept consistent with CP_HGT_s1
def load_separated_data_modified(train_path: str, eval_path: str, test_path: str):
    """Load the separated data files and build the heterogeneous graph - uses the modified builder."""
    print("Building the heterogeneous graph (using the modified data builder)...")
    
    builder = ModifiedHeterographDataBuilder()
    
    # Load the training data
    print("Loading the training data...")
    with open(train_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing train data")):
            try:
                # Parse the JSON line
                line_data = json.loads(line.strip())
                # The data format is {"paper_id": {paper_data}}, the inner data needs to be extracted
                for paper_id, paper_data in line_data.items():
                    # Add an id field to paper_data
                    paper_data['id'] = paper_id
                    builder.add_paper(paper_data, 'train')
            except Exception as e:
                print(f"Error processing train line {line_num}: {e}")
                continue
    
    # Load the validation data
    print("Loading the validation data...")
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing eval data")):
            try:
                # Parse the JSON line
                line_data = json.loads(line.strip())
                # The data format is {"paper_id": {paper_data}}, the inner data needs to be extracted
                for paper_id, paper_data in line_data.items():
                    # Add an id field to paper_data
                    paper_data['id'] = paper_id
                    builder.add_paper(paper_data, 'eval')
            except Exception as e:
                print(f"Error processing eval line {line_num}: {e}")
                continue
    
    # Load the test data
    print("Loading the test data...")
    with open(test_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing test data")):
            try:
                # Parse the JSON line
                line_data = json.loads(line.strip())
                # The data format is {"paper_id": {paper_data}}, the inner data needs to be extracted
                for paper_id, paper_data in line_data.items():
                    # Add an id field to paper_data
                    paper_data['id'] = paper_id
                    builder.add_paper(paper_data, 'test')
            except Exception as e:
                print(f"Error processing test line {line_num}: {e}")
                continue
    
    return builder


def train_two_stage_model_standalone():
    """Main function for standalone training of the two-stage model."""
    print("="*60)
    print("Standalone training of the two-stage HGT citation prediction model")
    print("="*60)
    
    # Set the random seed
    config.set_random_seed()
    
    # Print the configuration summary
    config.print_config_summary()
    
    # Use the data loading function from this file
    
    # Get the data paths from the configuration file
    data_paths = config.get_data_paths()
    
    # 1. Load the data
    print("Loading the data...")
    builder = load_separated_data_modified(
        data_paths['train'],
        data_paths['eval'], 
        data_paths['test']
    )
    # Build the full graph (includes venue information)
    data = builder.build_hetero_data(include_venue=True)
    
    # Print the statistics
    stats = builder.get_statistics()
    print(f"Data statistics: {stats}")
    
    # 2. Create the model
    print("\nCreating the two-stage model...")
    model_params = config.get_model_params()
    
    # Create the two-stage model
    model = TwoStageCitationHGT(**model_params)
    
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Create venue labels (used for adversarial training)
    def create_venue_labels(data, device, threshold=None):
        """Create venue labels (used for adversarial training)."""
        if threshold is None:
            threshold = config.DATA_CONFIG['venue_threshold']
            
        venue_features = data['venue'].x
        venue_labels = (venue_features >= threshold).float().squeeze()
        
        # Map venue labels to papers
        paper_venue_edges = data['paper', 'published_in', 'venue'].edge_index
        paper_venue_labels = torch.zeros(data['paper'].x.size(0), device=device)
        
        for paper_idx, venue_idx in paper_venue_edges.t():
            paper_venue_labels[paper_idx] = venue_labels[venue_idx]
        
        print(f"Venue label statistics: high-tier fraction={paper_venue_labels.mean():.3f}")
        return paper_venue_labels
    
    # 4. Create the trainer
    print("\nCreating the trainer...")
    training_params = config.get_training_params()
    device = training_params['device']
    venue_labels = create_venue_labels(data, device) if model.use_adversarial else None
    
    trainer = TwoStageTrainer(
        model=model,
        data=data,
        device=device,
        venue_labels=venue_labels
    )
    
    # 5. Train the model
    print("\nStarting training...")
    trainer.train(
        num_epochs=training_params['num_epochs'], 
        patience=training_params['patience'], 
        verbose=training_params['verbose']
    )
    
    # 6. Evaluate the model
    print("\nEvaluating the model...")
    val_metrics = trainer.evaluate('val')
    test_metrics = trainer.evaluate('test')
    
    # 7. Print the detailed results
    print("\n" + "="*50)
    print("Final evaluation results")
    print("="*50)
    print_detailed_metrics(val_metrics, "Validation set")
    print_detailed_metrics(test_metrics, "Test set")
    
    # 8. Analyze the latent variable
    print("\n" + "="*50)
    print("Early-exposure latent-variable analysis")
    print("="*50)
    early_exposure_pred = val_metrics['early_exposure_pred']
    citation_targets = val_metrics['targets']
    quartile_accuracy = val_metrics['quartile_accuracy']
    
    correlation = np.corrcoef(early_exposure_pred, citation_targets)[0, 1]
    print(f"Correlation between the early-exposure latent variable and citation counts: {correlation:.4f}")
    print(f"Quartile prediction accuracy: {quartile_accuracy:.4f}")
    
    # 9. Save the results
    results = {
        'model': model,
        'trainer': trainer,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'correlation': correlation,
        'quartile_accuracy': quartile_accuracy
    }
    
    # Save the model
    model_save_path = config.SAVE_CONFIG['model_save_path']
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'correlation': correlation,
        'quartile_accuracy': quartile_accuracy
    }, model_save_path)
    
    print(f"\nModel saved to: {model_save_path}")
    print("Training complete!")
    
    return results


if __name__ == "__main__":
    print("Two-stage HGT citation prediction model (latent-variable version) is ready!")
    print("\nMain features:")
    print("1. Two-stage architecture: V->E->Y path")
    print("2. Early-exposure latent-variable learning (Stage A, no ground-truth labels needed)")
    print("3. Citation prediction (Stage B, does not use V directly)")
    print("4. Quartile prediction auxiliary head (regularization)")
    print("5. Optional adversarial debiasing module")
    print("6. Complete training and evaluation pipeline")
    print("\n" + "="*60)
    print("Starting standalone training...")
    print("="*60)
    
    # Run standalone training
    results = train_two_stage_model_standalone()
