import os
import requests
import json
import pandas as pd
import re
import time
import argparse
import csv
from typing import Dict, List, Any

def parse_number_to_format(number_text):
    """Parse number information into required format"""
    if not number_text or str(number_text).strip().lower() in ['unspecified', 'none', '', 'number not specified']:
        return 'unknown'
    
    text = str(number_text).strip()
    
    # Look for density patterns first
    density_patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:ind(?:ividuals?)?|specimens?|animals?)\s*(?:per|/)\s*(?:m[²2]|square\s*m(?:eter)?s?|ha|hectare)',
        r'(\d+(?:\.\d+)?)\s*/\s*(?:m[²2]|ha)',
        r'density\s*(?:of\s*)?(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:per|/)\s*(?:m[²2]|ha|hectare)',
    ]
    
    for pattern in density_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1)
            if 'm' in text.lower():
                return f"{num} indv/m2"
            elif 'ha' in text.lower():
                return f"{num} indv/ha"
            else:
                return f"{num} indv/m2"
    
    # Look for simple numbers
    number_patterns = [
        r'n\s*=\s*(\d+)',
        r'N\s*=\s*(\d+)',
        r'(\d+)\s*(?:individuals?|specimens?|animals?|subjects?|fish|birds?|mammals?)',
        r'sample\s*size\s*(?:of\s*)?(\d+)',
        r'total\s*(?:of\s*)?(\d+)',
        r'(\d+)\s*(?:were|was)\s*(?:collected|captured|studied)',
        r'(?:using|with|from)\s*(\d+)\s*(?:individuals?|specimens?)',
    ]
    
    numbers = []
    for pattern in number_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        numbers.extend(matches)
    
    if numbers:
        unique_numbers = []
        for num in numbers:
            if num not in unique_numbers:
                unique_numbers.append(num)
        
        if len(unique_numbers) > 1:
            return '\n'.join(unique_numbers)
        else:
            return unique_numbers[0]
    
    # Look for any numbers as fallback
    all_numbers = re.findall(r'\b(\d+)\b', text)
    if all_numbers:
        filtered_numbers = [n for n in all_numbers if 1 <= int(n) <= 100000 and int(n) not in range(1900, 2030)]
        if filtered_numbers:
            if len(filtered_numbers) > 1:
                return '\n'.join(filtered_numbers[:3])
            else:
                return filtered_numbers[0]
    
    return 'unknown'

def extract_clean_value(value):
    """Extract clean string value from various input types"""
    if value is None:
        return "UNSPECIFIED"
    
    if isinstance(value, dict):
        for key in ['value', 'text', '$', 'content']:
            if key in value:
                return extract_clean_value(value[key])
        
        if len(value) == 1:
            return extract_clean_value(list(value.values())[0])
        
        return str(value)
    
    elif isinstance(value, list):
        if not value:
            return "UNSPECIFIED"
        elif len(value) == 1:
            return extract_clean_value(value[0])
        else:
            clean_values = []
            for item in value:
                clean_item = extract_clean_value(item)
                if clean_item not in clean_values and clean_item != "UNSPECIFIED":
                    clean_values.append(clean_item)
            if clean_values:
                return "; ".join(clean_values)
            else:
                return "UNSPECIFIED"
    
    else:
        result = str(value).strip() if value else "UNSPECIFIED"
        return result if result else "UNSPECIFIED"

def clean_claude_response(claude_response):
    """Clean and parse Claude's response"""
    try:
        claude_response = re.sub(r'```(?:json)?\n', '', claude_response)
        claude_response = re.sub(r'\n```', '', claude_response)
        
        json_match = re.search(r'(\[.*\]|\{.*\})', claude_response, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            json_text = claude_response
        
        result = json.loads(json_text)
        
        if isinstance(result, dict):
            result = [result]
        
        return result
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        objects = re.findall(r'\{[^{}]*\}', claude_response)
        parsed_objects = []
        
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                parsed_objects.append(obj)
            except:
                continue
        
        if parsed_objects:
            return parsed_objects
        else:
            print(f"Could not parse Claude response: {claude_response[:200]}...")
            return []

def extract_species_data(paper_text: str, paper_info: Dict = None) -> Dict[str, List[Dict[str, Any]]]:
    """Extract species information using Claude API"""
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    paper_title = paper_info.get('title', "UNSPECIFIED") if paper_info else "UNSPECIFIED"
    paper_doi = paper_info.get('doi', "UNSPECIFIED") if paper_info else "UNSPECIFIED"
    
    prompt = f"""
    Extract species information from this research paper. Return ONLY a JSON array of objects.

    For each species mentioned in the study (not just examples or background), extract:
    - species: scientific name (Genus species format)
    - number: specimen count or "number not specified"
    - study_type: "Laboratory", "Field", or "Field+Laboratory"
    - location: study location/site
    The location is very important, so, if it possible obtain the geological place, or a place that can be easily locate at a map with folium. 

    Return format (use simple strings only, no nested objects):
    [
      {{
        "species": "Genus species",
        "number": "count or number not specified",
        "study_type": "Laboratory/Field/Field+Laboratory",
        "location": "location description"
      }}
    ]

    Paper: {paper_title}
    DOI: {paper_doi}

    Text to analyze:
    {paper_text[:50000]}
    """
    
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1500,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    # Retry logic for rate limits
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload, 
                timeout=60
            )
            
            if response.status_code == 429:
                wait_time = min(2 ** attempt, 60)
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
                
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.text}")
                
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt, 60)
                print(f"ERROR: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
    else:
        raise Exception("Max retries exceeded")
    
    response_data = response.json()
    claude_response = response_data["content"][0]["text"]
    
    # Clean and parse Claude's response
    try:
        result = clean_claude_response(claude_response)
        
        if not result:
            print("No valid JSON found in Claude's response")
            return {"species_data": []}
        
        # Process each result - CREATE EXACTLY THE RIGHT OUTPUT FORMAT
        cleaned_results = []
        for item in result:
            if not isinstance(item, dict):
                continue
            
            # Extract and clean each field
            species_value = extract_clean_value(item.get('species'))
            number_value = extract_clean_value(item.get('number'))
            study_type_value = extract_clean_value(item.get('study_type'))
            location_value = extract_clean_value(item.get('location'))
            
            # CRITICAL: Create the EXACT structure expected by the bash script
            clean_item = {
                'paper_link': paper_doi,      # This becomes the main DOI link
                'species': species_value,     # Scientific name
                'study_type': study_type_value,  # Study type
                'location': location_value,   # Location
                'paper_title': paper_title    # Title LAST
            }
            
            # Process the number field with our custom formatter
            formatted_number = parse_number_to_format(number_value)
            
            # Handle multiple numbers by creating separate rows
            if '\n' in formatted_number:
                numbers = formatted_number.split('\n')
                for num in numbers:
                    if num.strip():
                        row_item = clean_item.copy()
                        row_item['number'] = num.strip()
                        cleaned_results.append(row_item)
            else:
                clean_item['number'] = formatted_number
                cleaned_results.append(clean_item)
        
        return {"species_data": cleaned_results}
    
    except Exception as e:
        print(f"ERROR extracting JSON: {e}")
        print(f"Claude's response: {claude_response[:500]}...")
        return {"species_data": []}

def safe_read_papers_csv(file_path):
    """Read papers CSV safely - expects: authors,journal,year,abstract,doi,pmid,url,database,title"""
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL, encoding='utf-8')
        print(f" Read {len(df)} papers from {file_path}")
        print(f" Columns found: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        # Try alternative methods
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"Read with fallback method: {len(df)} papers")
            return df
        except Exception as e2:
            print(f"ALL methods failed: {e2}")
            raise

def safe_write_species_csv(species_data, file_path):
    """Write species CSV with EXACT format: paper_link,species,number,study_type,location,paper_title"""
    if not species_data:
        print(f"No species data to write to {file_path}")
        return
    
    df = pd.DataFrame(species_data)
    
    # EXACT column order expected by bash script
    desired_columns = ['paper_link', 'species', 'number', 'study_type', 'location', 'paper_title']
    
    # Ensure all columns exist
    for col in desired_columns:
        if col not in df.columns:
            df[col] = "UNSPECIFIED"
    
    # Reorder to exact specification
    df = df[desired_columns]
    
    # Write with proper quoting
    df.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f"OK Wrote {len(df)} species entries to {file_path}")
    print(f"OK Column order: {list(df.columns)}")

def process_dois_from_csv(input_csv, output_csv, claude_api_key, scopus_api_key=None, inst_token=None, max_papers=None):
    """
    Process papers CSV and extract species information
    INPUT: CSV with columns: authors,journal,year,abstract,doi,pmid,url,database,title
    OUTPUT: CSV with columns: paper_link,species,number,study_type,location,paper_title
    """
    os.environ["ANTHROPIC_API_KEY"] = claude_api_key
    
    try:
        # Read the papers CSV
        df = safe_read_papers_csv(input_csv)
        
        # We need title and abstract columns - they should be there from search script
        if 'title' not in df.columns:
            print("ERROR: 'title' column not found in input CSV")
            print(f"Available columns: {list(df.columns)}")
            return
        
        if 'abstract' not in df.columns:
            print("  Warning: 'abstract' column not found - will use title only")
        
        # Get DOI/URL column
        doi_column = None
        for col in ['doi', 'url', 'pmid']:
            if col in df.columns and df[col].notna().any():
                doi_column = col
                break
        
        if not doi_column:
            print("ERROR: No valid DOI/URL column found")
            return
        
        print(f"Using '{doi_column}' as identifier column")
        
        # Process papers
        papers_to_process = []
        for idx, row in df.iterrows():
            identifier = row[doi_column]
            if pd.notna(identifier) and str(identifier).strip():
                papers_to_process.append((idx, str(identifier).strip()))
        
        if max_papers:
            papers_to_process = papers_to_process[:max_papers]
        
        print(f"Processing {len(papers_to_process)} papers...")
        
        all_species_data = []
        
        for i, (row_idx, identifier) in enumerate(papers_to_process):
            try:
                print(f"\n Processing paper {i+1}/{len(papers_to_process)}: {identifier}")
                
                paper_row = df.iloc[row_idx]
                
                # Get paper info
                paper_info = {
                    'doi': identifier,
                    'title': str(paper_row['title']).strip() if pd.notna(paper_row['title']) else "UNSPECIFIED"
                }
                
                # Create paper text
                text_parts = []
                
                # Add title
                if paper_info['title'] != "UNSPECIFIED":
                    text_parts.append(f"Title: {paper_info['title']}")
                
                # Add abstract if available
                if 'abstract' in df.columns and pd.notna(paper_row['abstract']):
                    abstract = str(paper_row['abstract']).strip()
                    if abstract:
                        text_parts.append(f"Abstract: {abstract}")
                
                # Add other metadata
                for col in ['authors', 'journal', 'year']:
                    if col in df.columns and pd.notna(paper_row[col]):
                        val = str(paper_row[col]).strip()
                        if val:
                            text_parts.append(f"{col.title()}: {val}")
                
                if not text_parts:
                    print(f"  No text available for {identifier}. Skipping.")
                    continue
                
                paper_text = "\n\n".join(text_parts)
                
                # Extract species data
                print("Extracting species data using Claude API...")
                species_data = extract_species_data(paper_text, paper_info)
                
                # Add extracted data to the list
                if species_data and species_data.get("species_data"):
                    print(f"Found {len(species_data['species_data'])} species in this paper")
                    all_species_data.extend(species_data["species_data"])
                else:
                    print("No species data extracted from this paper")
                
                # Rate limiting
                if i < len(papers_to_process) - 1:
                    wait_time = 5
                    print(f"Waiting {wait_time} seconds before next paper...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"ERROR processing {identifier}: {e}")
        
        # Save species data
        if all_species_data:
            safe_write_species_csv(all_species_data, output_csv)
            print(f"\n Successfully saved {len(all_species_data)} species entries!")
        else:
            print("\n  No species data was extracted from any papers.")
    
    except Exception as e:
        print(f" Overall ERROR in processing: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Species data extraction from papers using Claude API')
    
    parser.add_argument('--input-csv', '-i', type=str, required=True, help='Input CSV file with papers')
    parser.add_argument('--output-csv', '-o', type=str, required=True, help='Output CSV file for species data')
    parser.add_argument('--claude-key', '-c', type=str, required=True, help='Claude API key')
    parser.add_argument('--scopus-key', '-s', type=str, default=None, help='Scopus API key (unused)')
    parser.add_argument('--inst-token', '-t', type=str, default=None, help='Institutional token (unused)')
    parser.add_argument('--max-papers', '-m', type=int, default=None, help='Maximum papers to process')
    
    args = parser.parse_args()
    
    print("SPECIES EXTRACTION - FINAL CORRECTED VERSION")
    print("=" * 60)
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_csv}")
    print(f"Max papers: {args.max_papers or 'ALL'}")
    print("=" * 60)
    
    process_dois_from_csv(
        args.input_csv,
        args.output_csv,
        args.claude_key,
        args.scopus_key,
        args.inst_token,
        args.max_papers
    )

if __name__ == "__main__":
    main()
