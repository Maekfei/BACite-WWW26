#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced author reputation Agent
Integrates the Google Scholar toolkit to obtain detailed metrics such as h-index
"""

import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Add the project path
project_root = Path(__file__).parent.parent  # Go back to the WWW_2025 directory
sys.path.append(str(project_root))

try:
    from camel.toolkits import GoogleScholarToolkit
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    print("✅ Successfully imported CAMEL modules")
except ImportError as e:
    print(f"❌ Failed to import CAMEL modules: {e}")
    print("Please make sure CAMEL is installed: pip install camel-ai")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AuthorMetrics:
    """Data class for author metrics"""
    name: str
    h_index: int = 0
    i10_index: int = 0
    total_citations: int = 0
    publication_count: int = 0
    affiliation: str = "Unknown"
    interests: List[str] = None
    recent_publications: List[Dict] = None
    source: str = "google_scholar"  # Data source
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.recent_publications is None:
            self.recent_publications = []

class EnhancedAuthorReputationAgent:
    """Enhanced author reputation Agent, integrating Google Scholar"""

    def __init__(self, model, author_stats: Dict[str, Dict] = None,
                 use_google_scholar: bool = True, cache_file: str = None):
        """
        Initialize the enhanced author reputation Agent

        Args:
            model: LLM model
            author_stats: Local author statistics
            use_google_scholar: Whether to use Google Scholar
            cache_file: Cache file path
        """
        self.agent_name = "Enhanced Author Reputation Agent"
        self.model = model
        self.author_stats = author_stats or {}
        self.use_google_scholar = use_google_scholar
        self.cache_file = cache_file or "maindata/author_metrics_cache.json"
        
        # Initialize the ChatAgent
        self.chat_agent = ChatAgent(
            system_message=self._get_system_prompt(),
            model=model,
            message_window_size=2
        )

        # Load the cache
        self.metrics_cache = self._load_cache()

        # Initialize the Google Scholar toolkit (if needed)
        self.gs_toolkit = None
        if self.use_google_scholar:
            try:
                # Don't create the instance here immediately; create it when needed
                logger.info("Google Scholar functionality is enabled")
            except Exception as e:
                logger.warning(f"Google Scholar initialization failed: {e}")
                self.use_google_scholar = False

    def _get_system_prompt(self) -> str:
        """Get the system prompt"""
        return """You are an author reputation assessment expert. Your task is to:
1. Assess author reputation based on metrics such as h-index, total citations, number of publications, and institutional prestige
2. Take both Google Scholar data and local statistics into account
3. Give an overall score from 1 to 5, where 5 is the highest reputation
4. Provide a detailed rationale for the assessment

Please output in JSON format:
{
    "score": score (1-5),
    "reasoning": "assessment rationale",
    "key_metrics": {
        "h_index": h-index,
        "total_citations": total citations,
        "publication_count": number of publications
    }
}
"""
    
    def _load_cache(self) -> Dict[str, AuthorMetrics]:
        """Load the author metrics cache"""
        cache = {}
        try:
            if Path(self.cache_file).exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for author_id, metrics_data in data.items():
                        cache[author_id] = AuthorMetrics(**metrics_data)
                logger.info(f"Loaded cached data for {len(cache)} authors")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
        return cache

    def _save_cache(self):
        """Save the author metrics cache"""
        try:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            cache_data = {}
            for author_id, metrics in self.metrics_cache.items():
                cache_data[author_id] = {
                    'name': metrics.name,
                    'h_index': metrics.h_index,
                    'i10_index': metrics.i10_index,
                    'total_citations': metrics.total_citations,
                    'publication_count': metrics.publication_count,
                    'affiliation': metrics.affiliation,
                    'interests': metrics.interests,
                    'recent_publications': metrics.recent_publications,
                    'source': metrics.source
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved cached data for {len(cache_data)} authors")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _get_google_scholar_metrics(self, author_name: str) -> Optional[AuthorMetrics]:
        """Get author metrics from Google Scholar"""
        if not self.use_google_scholar:
            return None

        try:
            logger.info(f"🔍 Searching Google Scholar: {author_name}")

            # Create a Google Scholar toolkit instance
            toolkit = GoogleScholarToolkit(
                author_identifier=author_name,
                is_author_name=True,
                use_free_proxies=False,
                timeout=30.0
            )
            
            # Get detailed author information
            author_info = toolkit.get_author_detailed_info()
            
            if author_info:
                metrics = AuthorMetrics(
                    name=author_info.get('name', author_name),
                    h_index=author_info.get('hindex', 0),
                    i10_index=author_info.get('i10index', 0),
                    total_citations=author_info.get('citedby', 0),
                    publication_count=len(author_info.get('publications', [])),
                    affiliation=author_info.get('affiliation', 'Unknown'),
                    interests=author_info.get('interests', []),
                    recent_publications=author_info.get('publications', [])[:5],
                    source="google_scholar"
                )
                
                logger.info(f"✅ Retrieved Google Scholar data for {author_name}: h-index={metrics.h_index}")
                return metrics
            else:
                logger.warning(f"⚠️ Could not find {author_name} on Google Scholar")
                return None

        except Exception as e:
            logger.error(f"❌ Google Scholar search for {author_name} failed: {e}")
            return None

    def _get_local_metrics(self, author_id: str, author_name: str) -> AuthorMetrics:
        """Get local statistical metrics"""
        stats = self.author_stats.get(author_id, {})

        return AuthorMetrics(
            name=author_name,
            h_index=0,  # Local data usually has no h-index
            i10_index=0,
            total_citations=stats.get('citation_count', 0),
            publication_count=stats.get('paper_count', 0),
            affiliation=stats.get('org', 'Unknown'),
            source="local_stats"
        )
    
    def get_author_metrics(self, author_id: str, author_name: str) -> AuthorMetrics:
        """Get author metrics (prefer the cache, then Google Scholar, then local data)"""

        # 1. Check the cache
        if author_id in self.metrics_cache:
            logger.info(f"📋 Using cached data: {author_name}")
            return self.metrics_cache[author_id]

        # 2. Try Google Scholar
        gs_metrics = self._get_google_scholar_metrics(author_name)
        if gs_metrics:
            # Save to the cache
            self.metrics_cache[author_id] = gs_metrics
            self._save_cache()
            return gs_metrics

        # 3. Use local data
        logger.info(f"📊 Using local data: {author_name}")
        local_metrics = self._get_local_metrics(author_id, author_name)
        self.metrics_cache[author_id] = local_metrics
        self._save_cache()
        return local_metrics
    
    def extract_features(self, paper, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract author reputation features"""
        if context is None:
            context = {}

        authors = paper.authors
        if not authors:
            return {
                'agent_name': self.agent_name,
                'features': {'author_reputation': 1.0},
                'evidence': {'error': 'No author information'},
                'confidence': 0.1
            }

        # Get metrics for the first author and the last author (usually the corresponding author)
        first_author = authors[0]
        last_author = authors[-1] if len(authors) > 1 else first_author
        
        first_author_metrics = self.get_author_metrics(
            first_author['id'], 
            first_author['name']
        )
        
        last_author_metrics = self.get_author_metrics(
            last_author['id'], 
            last_author['name']
        )
        
        # Build the LLM analysis prompt
        prompt = f"""
        Assess the academic reputation of the following authors:

        First author: {first_author_metrics.name}
        - Affiliation: {first_author_metrics.affiliation}
        - h-index: {first_author_metrics.h_index}
        - Total citations: {first_author_metrics.total_citations}
        - Number of publications: {first_author_metrics.publication_count}
        - Data source: {first_author_metrics.source}

        Last author: {last_author_metrics.name}
        - Affiliation: {last_author_metrics.affiliation}
        - h-index: {last_author_metrics.h_index}
        - Total citations: {last_author_metrics.total_citations}
        - Number of publications: {last_author_metrics.publication_count}
        - Data source: {last_author_metrics.source}

        Based on the information above, assess the overall author reputation (score 1-5).
        """
        
        try:
            response = self.chat_agent.step(prompt)
            
            # Try to parse the JSON response
            try:
                result = json.loads(response.msg.content)
                score = result.get('score', 3.0)
                reasoning = result.get('reasoning', response.msg.content[:200])
            except:
                # If it is not JSON, try to extract a number
                import re
                score_match = re.search(r'(\d+(?:\.\d+)?)', response.msg.content)
                score = float(score_match.group(1)) if score_match else 3.0
                reasoning = response.msg.content[:200]

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            score = 3.0
            reasoning = "Analysis failed, using the default score"
        
        features = {
            'author1_reputation': score,
            'author1_h_index': first_author_metrics.h_index,
            'author1_citations': first_author_metrics.total_citations,
            'author2_reputation': score,  # Simplified handling
            'author2_h_index': last_author_metrics.h_index,
            'author2_citations': last_author_metrics.total_citations,
        }
        
        evidence = {
            'author1_metrics': f"{first_author_metrics.name}: h-index={first_author_metrics.h_index}, citations={first_author_metrics.total_citations}",
            'author2_metrics': f"{last_author_metrics.name}: h-index={last_author_metrics.h_index}, citations={last_author_metrics.total_citations}",
            'llm_reasoning': reasoning,
            'data_sources': f"First author: {first_author_metrics.source}, Last author: {last_author_metrics.source}"
        }
        
        return {
            'agent_name': self.agent_name,
            'features': features,
            'evidence': evidence,
            'confidence': 0.9 if first_author_metrics.source == "google_scholar" else 0.7
        }

def test_enhanced_agent():
    """Test the enhanced Agent"""
    print("🧪 Testing the enhanced author reputation Agent")
    print("-" * 50)

    try:
        # Initialize the model
        model = ModelFactory.create(
            model_platform=ModelPlatformType.OLLAMA,
            model_type="llama3.1:8b",
            url="http://localhost:11434/v1",
            model_config_dict={"temperature": 0.0, "max_tokens": 1024}
        )
        
        # Create the enhanced Agent
        agent = EnhancedAuthorReputationAgent(
            model=model,
            author_stats={},  # Empty local statistics
            use_google_scholar=True,
            cache_file="maindata/test_author_cache.json"
        )

        # Create test paper data
        class MockPaper:
            def __init__(self):
                self.authors = [
                    {'id': 'hinton_geoffrey', 'name': 'Geoffrey Hinton', 'org': 'University of Toronto'},
                    {'id': 'lecun_yann', 'name': 'Yann LeCun', 'org': 'NYU'}
                ]
        
        test_paper = MockPaper()

        # Test feature extraction
        print("🔍 Testing author reputation feature extraction...")
        result = agent.extract_features(test_paper)

        print(f"\n📊 Extraction result:")
        print(f"  Agent name: {result['agent_name']}")
        print(f"  Features: {result['features']}")
        print(f"  Evidence: {result['evidence']}")
        print(f"  Confidence: {result['confidence']}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    test_enhanced_agent()
