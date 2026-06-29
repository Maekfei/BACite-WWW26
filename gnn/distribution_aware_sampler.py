#!/usr/bin/env python3
"""
Distribution-aware intelligent sampler based on the distribution characteristics of the test1 dataset.
Samples by mimicking the statistical properties of the test1 dataset.
"""

import json
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DistributionTarget:
    """Target distribution characteristics"""
    # Citation count distribution
    citation_mean: float = 51.58
    citation_median: float = 19.0
    citation_std: float = 367.62
    citation_max: int = 18816
    citation_95th: float = 137.05
    citation_99th: float = 529.23

    # Year distribution weights (2010-2018)
    year_weights: Dict[int, float] = None

    # Author-count distribution weights
    author_count_weights: Dict[int, float] = None

    # Venue distribution weights (top 10 major venues)
    venue_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.year_weights is None:
            # Year distribution based on test1
            self.year_weights = {
                2010: 0.082, 2011: 0.091, 2012: 0.105, 2013: 0.100,
                2014: 0.102, 2015: 0.108, 2016: 0.112, 2017: 0.135, 2018: 0.165
            }
        
        if self.author_count_weights is None:
            # Author-count distribution based on test1
            self.author_count_weights = {
                1: 0.063, 2: 0.233, 3: 0.286, 4: 0.213, 5: 0.112,
                6: 0.047, 7: 0.022, 8: 0.012, 9: 0.005, 10: 0.003,
                11: 0.001, 12: 0.001, 13: 0.0004, 14: 0.0002, 15: 0.0002,
                16: 0.0003, 18: 0.0002, 19: 0.0001, 20: 0.0002
            }
        
        if self.venue_weights is None:
            # Major venue distribution based on test1
            self.venue_weights = {
                'CoRR': 0.135, 'arXiv (Cornell University)': 0.127, 'Neurocomputing': 0.079,
                'Expert Systems with Applications': 0.055, 'IEEE Access': 0.051,
                'Information Sciences': 0.042, 'arXivorg': 0.042, 'Pattern Recognition': 0.042,
                'Applied Soft Computing': 0.038, 'Sensors': 0.037
            }

class DistributionAwareSampler:
    """Distribution-aware intelligent sampler"""

    def __init__(self, data_path: str, target_distribution: DistributionTarget = None):
        self.data_path = data_path
        self.target_distribution = target_distribution or DistributionTarget()
        self.all_papers = self._load_papers()
        logger.info(f"Data loading complete, {len(self.all_papers)} papers in total")

    def _load_papers(self) -> List[Dict[str, Any]]:
        """Load paper data"""
        papers = []
        print(f"Loading data file: {self.data_path}")

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    papers.append(data)

                    if line_num % 100000 == 0 and line_num > 0:
                        print(f"Loaded {line_num + 1} papers")

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num + 1}: {e}")
                except Exception as e:
                    print(f"Error processing line {line_num + 1}: {e}")

        print(f"Data loading complete, {len(papers)} papers in total")
        return papers

    def _calculate_paper_score(self, paper: Dict[str, Any]) -> float:
        """Compute the match score between a paper and the target distribution"""
        score = 0.0

        # 1. Citation-count match score (40% weight)
        citation = paper.get('n_citation', 0)
        if citation <= self.target_distribution.citation_95th:
            # Within the 95th percentile, give a high score
            citation_score = 1.0 - abs(citation - self.target_distribution.citation_median) / self.target_distribution.citation_median
        else:
            # Above the 95th percentile, give a low score
            citation_score = 0.1
        score += citation_score * 0.4

        # 2. Year match score (20% weight)
        year = paper.get('year', 2010)
        if year in self.target_distribution.year_weights:
            year_score = self.target_distribution.year_weights[year]
        else:
            year_score = 0.0
        score += year_score * 0.2

        # 3. Author-count match score (20% weight)
        author_count = len(paper.get('authors', []))
        if author_count in self.target_distribution.author_count_weights:
            author_score = self.target_distribution.author_count_weights[author_count]
        else:
            author_score = 0.0
        score += author_score * 0.2

        # 4. Venue match score (20% weight)
        venue = paper.get('venue', '')
        if venue in self.target_distribution.venue_weights:
            venue_score = self.target_distribution.venue_weights[venue]
        else:
            venue_score = 0.01  # Give other venues a small baseline score
        score += venue_score * 0.2

        return score

    # This method was removed - we do not generate new citation counts, we only sample existing data

    def sample_papers_by_distribution(self, n_samples: int, seed: int = 52) -> List[Dict[str, Any]]:
        """Sample papers based on distribution characteristics - only changes the sampling distribution, not any data labels"""
        np.random.seed(seed)

        print(f"Starting distribution-aware intelligent sampling, target sample count: {n_samples}")
        print("⚠️  Important: does not change any data labels, only changes the sampling distribution")

        # 1. Compute match scores for all papers
        print("Computing match scores between papers and target distribution...")
        paper_scores = []
        for i, paper in enumerate(self.all_papers):
            score = self._calculate_paper_score(paper)
            paper_scores.append((paper, score))  # Store the paper object directly

            if i % 100000 == 0 and i > 0:
                print(f"Computed scores for {i + 1} papers")

        # 2. Sort by score
        print("Sorting by match score...")
        paper_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Stratified sampling strategy
        print("Performing stratified sampling...")
        total_papers = len(paper_scores)
        top_ratio = 0.3  # Top 30% high-scoring papers
        n_top = int(n_samples * top_ratio)
        n_random = n_samples - n_top

        # Randomly sample from high-scoring papers
        top_papers = [paper for paper, score in paper_scores[:int(total_papers * top_ratio)]]
        if len(top_papers) >= n_top:
            sampled_top = np.random.choice(top_papers, n_top, replace=False).tolist()
        else:
            sampled_top = top_papers  # If there are not enough high-scoring papers, take all of them

        # Randomly sample from the remaining papers
        remaining_papers = [paper for paper, score in paper_scores[int(total_papers * top_ratio):]]
        if len(remaining_papers) >= n_random:
            sampled_random = np.random.choice(remaining_papers, n_random, replace=False).tolist()
        else:
            sampled_random = remaining_papers  # If there are not enough remaining papers, take all of them

        # Merge results
        sampled_papers = sampled_top + sampled_random

        print(f"Sampling complete:")
        print(f"  High-scoring papers: {len(sampled_top)} (from the top {top_ratio*100:.0f}% high-scoring papers)")
        print(f"  Random papers: {len(sampled_random)} (from the remaining papers)")
        print(f"  Total: {len(sampled_papers)}")

        return sampled_papers

    def analyze_sampled_distribution(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution characteristics after sampling"""
        citations = [p.get('n_citation', 0) for p in papers]
        years = [p.get('year', 2010) for p in papers]
        author_counts = [len(p.get('authors', [])) for p in papers]
        venues = [p.get('venue', '') for p in papers]
        
        analysis = {
            'citation_stats': {
                'mean': np.mean(citations),
                'median': np.median(citations),
                'std': np.std(citations),
                'min': np.min(citations),
                'max': np.max(citations),
                'p95': np.percentile(citations, 95),
                'p99': np.percentile(citations, 99)
            },
            'year_distribution': dict(Counter(years)),
            'author_count_distribution': dict(Counter(author_counts)),
            'venue_distribution': dict(Counter(venues)),
            'total_papers': len(papers)
        }
        
        return analysis
    
    def save_sampled_data(self, papers: List[Dict[str, Any]], output_path: str):
        """Save the sampled data"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')

        logger.info(f"Sampled data saved to: {output_path} ({len(papers)} papers)")

def perform_distribution_aware_sampling(data_path: str, output_dir: str,
                                      sample_sizes: Dict[str, int] = None):
    """Perform distribution-aware intelligent sampling"""
    
    if sample_sizes is None:
        sample_sizes = {
            'train': 10000,
            'eval': 1000,
            'test': 1000
        }
    
    print("🚀 Starting distribution-aware intelligent sampling")
    print("=" * 60)
    print(f"Source data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Sampling configuration: {sample_sizes}")
    print("=" * 60)

    # Initialize the sampler
    sampler = DistributionAwareSampler(data_path)

    # Split data by year
    train_papers = []
    eval_papers = []
    test_papers = []

    for paper in sampler.all_papers:
        year = paper.get('year', 2010)
        if year <= 2018:
            train_papers.append(paper)
        elif year == 2019:
            eval_papers.append(paper)
        elif year == 2020:
            test_papers.append(paper)

    print(f"Data splitting complete:")
    print(f"  Training-set candidates (≤2018): {len(train_papers)}")
    print(f"  Validation-set candidates (2019): {len(eval_papers)}")
    print(f"  Test-set candidates (2020): {len(test_papers)}")

    # Perform intelligent sampling
    datasets = {}

    if train_papers:
        print(f"\n📊 Sampling training set...")
        sampler_train = DistributionAwareSampler(data_path)
        sampler_train.all_papers = train_papers
        datasets['train'] = sampler_train.sample_papers_by_distribution(
            sample_sizes['train'], seed=62
        )

    if eval_papers:
        print(f"\n📊 Sampling validation set...")
        sampler_eval = DistributionAwareSampler(data_path)
        sampler_eval.all_papers = eval_papers
        datasets['eval'] = sampler_eval.sample_papers_by_distribution(
            sample_sizes['eval'], seed=63
        )

    if test_papers:
        print(f"\n📊 Sampling test set...")
        sampler_test = DistributionAwareSampler(data_path)
        sampler_test.all_papers = test_papers
        datasets['test'] = sampler_test.sample_papers_by_distribution(
            sample_sizes['test'], seed=64
        )

    # Save sampling results
    print(f"\n💾 Saving sampling results...")
    for dataset_name, papers in datasets.items():
        if papers:
            output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
            sampler.save_sampled_data(papers, output_path)

            # Analyze distribution characteristics
            analysis = sampler.analyze_sampled_distribution(papers)
            print(f"\n📈 {dataset_name.upper()} set distribution analysis:")
            print(f"  Citations: mean={analysis['citation_stats']['mean']:.2f}, "
                  f"median={analysis['citation_stats']['median']:.2f}, "
                  f"std={analysis['citation_stats']['std']:.2f}")
            print(f"  Year distribution: {len(analysis['year_distribution'])} years")
            print(f"  Author-count distribution: {len(analysis['author_count_distribution'])} cases")
            print(f"  Venue distribution: {len(analysis['venue_distribution'])} venues")

    print(f"\n✅ Distribution-aware intelligent sampling complete!")
    return datasets

if __name__ == "__main__":
    # Configuration parameters
    data_path = "feature_extraction/ai_papers_complete_all_years_20251003_011733_converted.jsonl"
    output_dir = "maindata/Openalex/test3_distribution_aware"

    sample_sizes = {
        'train': 10000,
        'eval': 1000,
        'test': 1000
    }

    # Perform intelligent sampling
    datasets = perform_distribution_aware_sampling(data_path, output_dir, sample_sizes)
