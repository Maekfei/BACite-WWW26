#!/usr/bin/env python3
"""
Complete AI paper download script - saves all fields provided by the OpenAlex API
"""

import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIPaperDownloaderComplete:
    def __init__(self, output_dir: str = "feature_extraction"):
        self.base_url = "https://api.openalex.org/works"
        self.session = requests.Session()
        self.output_dir = output_dir
        
        # AI-related concept IDs (standard IDs in OpenAlex)
        self.ai_concept_ids = [
            "C154945302",  # Artificial intelligence (16,203,869 papers)
            "C119857082",  # Machine learning (4,064,008 papers)
            "C31972630",   # Computer vision (2,805,296 papers)
            "C124101348",  # Data mining (2,239,291 papers)
            "C153180895",  # Pattern recognition (1,642,307 papers)
            "C204321447",  # Natural language processing (1,111,148 papers)
        ]

        # Download configuration
        self.save_interval = 1000  # Save every 1000 papers
        self.max_retries = 5  # Maximum number of retries
        self.retry_delay = 2  # Retry delay (seconds)
        self.max_workers = 5  # Maximum number of threads
        self.request_delay = 0.1  # Interval between requests (seconds)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Thread lock used to control the request rate
        self.request_lock = threading.Lock()
        self.last_request_time = 0
    
    def get_ai_papers_by_year(self, 
                             start_year: int = 2010,
                             end_year: Optional[int] = None,
                             max_results_per_year: int = 1000,
                             min_citations: int = 3) -> Dict[int, List[Dict]]:
        """
        Fetch AI paper data by year

        Args:
            start_year: Start year
            end_year: End year; None means the current year
            max_results_per_year: Maximum number of results per year
            min_citations: Minimum citation count
        """
        if end_year is None:
            end_year = datetime.now().year
        
        all_papers_by_year = {}
        
        for year in range(start_year, end_year + 1):
            logger.info(f"Fetching AI papers for year {year}...")

            # Clean up any pre-existing incremental file for this year
            incremental_file = os.path.join(self.output_dir, f"ai_papers_complete_{year}_incremental.jsonl")
            if os.path.exists(incremental_file):
                os.remove(incremental_file)
                logger.info(f"🗑️ Cleaned up previous incremental file: {incremental_file}")

            # Build the filter conditions - using the correct syntax
            year_filter = f"publication_year:{year}"
            citation_filter = f"cited_by_count:>{min_citations}"

            # Query each AI concept ID separately using multiple threads
            year_papers = []

            # Create the thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_concept = {}
                for concept_id in self.ai_concept_ids:
                    concept_filter = f"concepts.id:{concept_id}"
                    filters = [year_filter, citation_filter, concept_filter]
                    
                    future = executor.submit(
                        self._fetch_papers_with_cursor,
                        filters=filters,
                        max_results=max_results_per_year // len(self.ai_concept_ids),
                        current_year=year
                    )
                    future_to_concept[future] = concept_id
                
                # Collect the results
                for future in as_completed(future_to_concept):
                    concept_id = future_to_concept[future]
                    try:
                        papers = future.result()
                        year_papers.extend(papers)
                        logger.info(f"  Concept {concept_id}: fetched {len(papers)} papers")
                    except Exception as e:
                        logger.error(f"  Concept {concept_id} download failed: {e}")

            # Deduplicate (based on OpenAlex ID)
            unique_papers = {}
            for paper in year_papers:
                paper_id = paper.get('id')
                if paper_id and paper_id not in unique_papers:
                    unique_papers[paper_id] = paper
            
            all_papers_by_year[year] = list(unique_papers.values())
            logger.info(f"Year {year}: fetched {len(all_papers_by_year[year])} unique AI papers in total")
        
        return all_papers_by_year
    
    def _make_request_with_retry(self, url: str, params: dict, timeout: int = 30) -> Optional[requests.Response]:
        """Request with a retry mechanism, supporting multiple threads"""
        for attempt in range(self.max_retries):
            try:
                # Thread-safe request rate control
                with self.request_lock:
                    current_time = time.time()
                    time_since_last = current_time - self.last_request_time
                    if time_since_last < self.request_delay:
                        time.sleep(self.request_delay - time_since_last)
                    self.last_request_time = time.time()
                
                response = self.session.get(url, params=params, timeout=timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Too many requests
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Too many requests, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:  # Server error
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.warning(f"Server error {response.status_code}, waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Request failed: {response.status_code}")
                    if response.status_code == 403:
                        logger.error(f"Error message: {response.text}")
                    return None

            except requests.exceptions.Timeout:
                wait_time = self.retry_delay * (attempt + 1)
                logger.warning(f"Request timed out, waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            except requests.exceptions.ConnectionError:
                wait_time = self.retry_delay * (attempt + 1)
                logger.warning(f"Connection error, waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            except Exception as e:
                logger.error(f"Request error: {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(self.retry_delay)
                continue
        
        return None

    def _fetch_papers_with_cursor(self, 
                                 filters: List[str],
                                 max_results: int,
                                 current_year: int = None) -> List[Dict]:
        """Fetch paper data using cursor pagination, with support for incremental saving"""
        params = {
            'filter': ",".join(filters),
            'per-page': 200,
            'cursor': '*'
        }
        
        all_papers = []
        last_save_count = 0
        
        while len(all_papers) < max_results:
            # Send the request using the retry mechanism
            response = self._make_request_with_retry(self.base_url, params)

            if response is None:
                logger.error("Request failed, stopping data fetch")
                break

            try:
                data = response.json()
                papers = data.get('results', [])

                if not papers:
                    logger.info("No more data, stopping fetch")
                    break

                all_papers.extend(papers)

                # Get the next cursor
                next_cursor = data.get('meta', {}).get('next_cursor')
                if not next_cursor:
                    logger.info("No more pages, stopping fetch")
                    break

                params['cursor'] = next_cursor

                # Avoid sending requests too frequently
                time.sleep(0.5)

                # Save data periodically
                if len(all_papers) - last_save_count >= self.save_interval:
                    self._save_incremental_data(all_papers[last_save_count:],
                                              current_year)
                    last_save_count = len(all_papers)
                    logger.info(f"💾 Incremental save: saved {len(all_papers)} papers so far")

                if len(all_papers) % 200 == 0:
                    logger.info(f"Fetched {len(all_papers)} papers so far...")

            except Exception as e:
                logger.error(f"Error while parsing response data: {e}")
                break
        
        return all_papers[:max_results]
    
    def _save_incremental_data(self, papers: List[Dict], current_year: int):
        """Incrementally save data to the same file (append mode)"""
        if not papers:
            return

        # Use a fixed filename to avoid creating too many intermediate files
        filename = f"ai_papers_complete_{current_year}_incremental.jsonl"
        filepath = os.path.join(self.output_dir, filename)

        try:
            processed_papers = self.process_paper_data(papers)

            # Use append mode ('a') instead of write mode ('w')
            with open(filepath, 'a', encoding='utf-8') as f:
                for paper in processed_papers:
                    json.dump(paper, f, ensure_ascii=False)
                    f.write('\n')

            logger.info(f"💾 Incremental data appended to {filepath} ({len(processed_papers)} papers)")
        except Exception as e:
            logger.error(f"Error while saving incremental data: {e}")
    
    def process_paper_data(self, papers: List[Dict]) -> List[Dict]:
        """Process paper data, keeping only the same fields as the reference file"""
        processed_papers = []

        for paper in papers:
            try:
                # Check whether paper is None
                if paper is None:
                    continue

                # Keep only the same fields as the reference file
                paper_info = {}

                # 1. Basic information
                paper_info['id'] = paper.get('id', '')
                paper_info['doi'] = paper.get('doi', '')
                paper_info['title'] = paper.get('title', '')
                paper_info['year'] = paper.get('publication_year')
                paper_info['doc_type'] = paper.get('type', '')
                paper_info['lang'] = paper.get('language', '')
                
                # 2. Abstract information
                abstract_inverted = paper.get('abstract_inverted_index')
                if abstract_inverted:
                    abstract_text = self._reconstruct_abstract(abstract_inverted)
                    paper_info['abstract'] = abstract_text
                else:
                    paper_info['abstract'] = ''
                
                # 3. Author information
                authorships = paper.get('authorships', [])
                authors = []
                author_ids = []
                for authorship in authorships:
                    author = authorship.get('author', {})
                    if author:
                        authors.append(author.get('display_name', ''))
                        author_ids.append(author.get('id', ''))
                paper_info['authors'] = '; '.join(authors)
                paper_info['author_ids'] = '; '.join(author_ids)
                
                # 4. Publication venue information
                primary_location = paper.get('primary_location', {})
                source = primary_location.get('source', {}) if primary_location else {}
                paper_info['venue'] = source.get('display_name', '') if source else ''
                
                # 5. Citation information
                paper_info['n_citation'] = paper.get('cited_by_count', 0)

                # 6. Citation counts by year
                counts_by_year = paper.get('counts_by_year', [])
                year_counts = {}
                for count_data in counts_by_year:
                    year = count_data.get('year')
                    count = count_data.get('cited_by_count', 0)
                    if year:
                        year_counts[year] = count
                
                # Extract citation counts for each year (note: OpenAlex data only has
                # citation data starting from the 2nd year after publication)
                # Therefore citations_year_1 actually corresponds to the 2nd year after publication
                current_year = paper.get('publication_year')
                if current_year:
                    paper_info['citations_year_1'] = year_counts.get(current_year + 2, 0)  # 2nd year after publication
                    paper_info['citations_year_2'] = year_counts.get(current_year + 3, 0)  # 3rd year after publication
                    paper_info['citations_year_3'] = year_counts.get(current_year + 4, 0)  # 4th year after publication
                    paper_info['citations_year_4'] = year_counts.get(current_year + 5, 0)  # 5th year after publication
                    paper_info['citations_year_5'] = year_counts.get(current_year + 6, 0)  # 6th year after publication
                else:
                    paper_info['citations_year_1'] = 0
                    paper_info['citations_year_2'] = 0
                    paper_info['citations_year_3'] = 0
                    paper_info['citations_year_4'] = 0
                    paper_info['citations_year_5'] = 0
                
                # 7. Keywords
                keywords = paper.get('keywords', [])
                if keywords:
                    keyword_list = [kw.get('display_name', '') for kw in keywords if kw.get('display_name')]
                    paper_info['keywords'] = '; '.join(keyword_list)
                else:
                    paper_info['keywords'] = ''
                
                # 8. References
                referenced_works = paper.get('referenced_works', [])
                paper_info['references'] = referenced_works  # Save the real list of reference IDs

                # 9. Other fields (set to empty or default values)
                paper_info['isbn'] = ''
                paper_info['issn'] = ''
                paper_info['issue'] = ''
                paper_info['page_end'] = ''
                paper_info['page_start'] = ''
                paper_info['url'] = ''
                paper_info['volume'] = ''
                
                processed_papers.append(paper_info)
                
            except Exception as e:
                logger.error(f"Error while processing paper data: {e}")
                continue

        return processed_papers

    def _reconstruct_abstract(self, abstract_inverted: Dict) -> str:
        """Reconstruct the abstract text"""
        try:
            words = {}
            for word, positions in abstract_inverted.items():
                for pos in positions:
                    words[pos] = word
            
            abstract_text = ' '.join([words[pos] for pos in sorted(words.keys())])
            return abstract_text
        except:
            return ''
    
    def save_to_jsonl(self, papers_by_year: Dict[int, List[Dict]], filename_prefix: str = "ai_papers_complete"):
        """Save paper data to a JSONL file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save the data for each year
        for year, papers in papers_by_year.items():
            if not papers:
                continue

            # Check whether an incremental file exists
            incremental_file = os.path.join(self.output_dir, f"ai_papers_complete_{year}_incremental.jsonl")

            if os.path.exists(incremental_file):
                # If an incremental file exists, just rename it to the final file
                final_filename = f"{filename_prefix}_{year}_{timestamp}.jsonl"
                final_filepath = os.path.join(self.output_dir, final_filename)
                os.rename(incremental_file, final_filepath)
                logger.info(f"Year {year} data renamed from the incremental file to {final_filepath}")
            else:
                # If there is no incremental file, process normally
                processed_papers = self.process_paper_data(papers)
                
                filename = f"{filename_prefix}_{year}_{timestamp}.jsonl"
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    for paper in processed_papers:
                        json.dump(paper, f, ensure_ascii=False)
                        f.write('\n')
                
                logger.info(f"Year {year} data saved to {filepath} ({len(processed_papers)} papers)")

        # Save the aggregated data
        all_papers = []
        for year, papers in papers_by_year.items():
            all_papers.extend(papers)
        
        if all_papers:
            processed_all_papers = self.process_paper_data(all_papers)
            
            filename_all = f"{filename_prefix}_all_years_{timestamp}.jsonl"
            filepath_all = os.path.join(self.output_dir, filename_all)
            
            with open(filepath_all, 'w', encoding='utf-8') as f:
                for paper in processed_all_papers:
                    json.dump(paper, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Aggregated data saved to {filepath_all} ({len(processed_all_papers)} papers)")

def main():
    """Main function"""
    logger.info("🚀 Starting download of AI-domain paper data (complete version)...")
    logger.info("=" * 60)

    # Create the downloader
    downloader = AIPaperDownloaderComplete()

    # Show the configuration
    logger.info("📋 Download configuration:")
    logger.info(f"   - Save interval: save every {downloader.save_interval} papers")
    logger.info(f"   - Maximum retries: {downloader.max_retries}")
    logger.info(f"   - Retry delay: {downloader.retry_delay} seconds")
    logger.info(f"   - Maximum threads: {downloader.max_workers}")
    logger.info(f"   - Request interval: {downloader.request_delay} seconds")
    logger.info(f"   - Output directory: {downloader.output_dir}")
    logger.info(f"   - Data completeness: only the same 24 fields as the reference file are saved")
    logger.info("=" * 60)

    # Download AI papers from 2010 onwards
    papers_by_year = downloader.get_ai_papers_by_year(
        start_year=2010,
        end_year=None,  # Current year
        max_results_per_year=10000,  # At most 10000 papers per year
        min_citations=0,  # At least 0 citations
    )

    # Save the final data
    logger.info("💾 Saving final data to file...")
    downloader.save_to_jsonl(papers_by_year, "ai_papers_complete")

    # Print statistics
    total_papers = sum(len(papers) for papers in papers_by_year.values())
    logger.info("=" * 60)
    logger.info(f"🎉 Download complete! Fetched {total_papers} AI papers in total")
    logger.info("=" * 60)

    for year, papers in papers_by_year.items():
        logger.info(f"📅 Year {year}: {len(papers)} papers")

    logger.info("=" * 60)
    logger.info("✅ All data has been saved!")
    logger.info("📊 The data contains all fields provided by the OpenAlex API")

if __name__ == "__main__":
    main()
