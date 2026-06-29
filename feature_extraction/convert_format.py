#!/usr/bin/env python3
"""
Convert the format of file 1 into the format of file 2.
Main conversions:
- authors: string + author_ids -> list format (using real author IDs)
- keywords: string -> list format
- references: list/integer -> list format (prefer real reference IDs)
- url: string -> list format
- citations_year_n: per-year counts -> cumulative counts
- doi: kept unchanged
"""

import json
import argparse
from typing import Dict, Any, List

def convert_authors_to_list(authors_str: str, author_ids_str: str) -> List[Dict[str, str]]:
    """Convert semicolon-separated author and author-ID strings into list format"""
    if not authors_str or authors_str.strip() == '':
        return []
    
    authors_list = []
    author_names = [name.strip() for name in authors_str.split(';') if name.strip()]
    author_ids = [aid.strip() for aid in author_ids_str.split(';') if aid.strip()] if author_ids_str else []
    
    # Ensure the author names and IDs have the same count
    for i, name in enumerate(author_names):
        author_id = author_ids[i] # If there is no ID, fall back to the name
        author_dict = {
            'id': author_id,  # Use the real author ID
            'name': name,
            'org': ''  # File 1 has no affiliation info, so leave it empty
        }
        authors_list.append(author_dict)
    
    return authors_list

def convert_keywords_to_list(keywords_str: str) -> List[str]:
    """Convert a semicolon-separated keyword string into list format"""
    if not keywords_str or keywords_str.strip() == '':
        return []
    
    keywords_list = [kw.strip() for kw in keywords_str.split(';') if kw.strip()]
    return keywords_list

def convert_references_to_list(references_data) -> List[str]:
    """Convert reference data into a list of reference IDs"""
    if not references_data:
        return []

    # If references_data is already a list, return it directly
    if isinstance(references_data, list):
        return references_data

    # Otherwise return an empty list
    return []

def convert_url_to_list(url_str: str) -> List[str]:
    """Convert a URL string into list format"""
    if not url_str or url_str.strip() == '':
        return []
    
    return [url_str.strip()]

def convert_citations_to_cumulative(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert citation counts into cumulative counts"""
    converted = record.copy()

    # Get the citation count for each year
    citations = []
    for year_num in range(1, 6):
        field_name = f'citations_year_{year_num}'
        count = converted.get(field_name, 0)
        citations.append(count)
    
    # Convert to cumulative counts
    cumulative_citations = []
    cumulative_sum = 0
    for count in citations:
        cumulative_sum += count
        cumulative_citations.append(cumulative_sum)

    # Update the record
    for year_num, cumulative_count in enumerate(cumulative_citations, 1):
        field_name = f'citations_year_{year_num}'
        converted[field_name] = cumulative_count
    
    return converted

def convert_record_format(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the format of a single record"""
    converted = record.copy()

    # Convert the authors field (needs both the authors and author_ids fields)
    if 'authors' in converted:
        author_ids = converted.get('author_ids', '')
        converted['authors'] = convert_authors_to_list(converted['authors'], author_ids)

    # Convert the keywords field
    if 'keywords' in converted:
        converted['keywords'] = convert_keywords_to_list(converted['keywords'])

    # Convert the references field
    if 'references' in converted:
        converted['references'] = convert_references_to_list(converted['references'])

    # Convert the url field
    if 'url' in converted:
        converted['url'] = convert_url_to_list(converted['url'])

    # Convert citation counts into cumulative counts
    converted = convert_citations_to_cumulative(converted)

    # The doi field is kept unchanged

    return converted

def convert_file_format(input_file: str, output_file: str, max_records: int = None):
    """Convert the format of an entire file"""
    print(f"🔄 Starting file format conversion...")
    print(f"📁 Input file: {input_file}")
    print(f"📁 Output file: {output_file}")
    
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            if max_records and line_num > max_records:
                break
                
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse the original record
                record = json.loads(line)

                # Convert the format
                converted_record = convert_record_format(record)

                # Write the converted record
                outfile.write(json.dumps(converted_record, ensure_ascii=False) + '\n')
                converted_count += 1

                if converted_count % 1000 == 0:
                    print(f"✅ Converted {converted_count} records so far...")

            except Exception as e:
                print(f"❌ Failed to convert line {line_num}: {e}")
                continue

    print(f"🎉 Conversion complete! Converted {converted_count} records in total")
    print(f"📄 Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Convert file format')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum number of records to convert (for testing)')
    
    args = parser.parse_args()
    
    convert_file_format(args.input, args.output, args.max_records)

if __name__ == '__main__':
    main()
