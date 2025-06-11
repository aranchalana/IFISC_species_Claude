#!/usr/bin/env python3
"""
Multi-database animal research paper search script.
OUTPUT: CSV with columns in this exact order: authors,journal,year,abstract,doi,pmid,url,database,title
Title is LAST to prevent comma issues.
"""

import requests
import pandas as pd
import time
import argparse
import csv
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional

def clean_text_for_csv(text):
    """Clean text to prevent CSV parsing issues"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('\x00', '')
    
    return text.strip()

def safe_write_papers_csv(papers: List[Dict], output_path: str):
    """Write papers CSV with EXACT column order: authors,journal,year,abstract,doi,pmid,url,database,title"""
    if not papers:
        print(f"No papers to write to {output_path}")
        return
    
    # Clean all text fields
    cleaned_papers = []
    for paper in papers:
        cleaned = {}
        for key, value in paper.items():
            if key in ['title', 'abstract', 'authors', 'journal']:
                cleaned[key] = clean_text_for_csv(value)
            else:
                cleaned[key] = str(value).strip() if value else ""
        cleaned_papers.append(cleaned)
    
    # Create DataFrame with EXACT column order
    df = pd.DataFrame(cleaned_papers)
    
    # FIXED COLUMN ORDER - title is LAST
    column_order = ['authors', 'journal', 'year', 'abstract', 'doi', 'pmid', 'url', 'database', 'title']
    
    # Ensure all columns exist
    for col in column_order:
        if col not in df.columns:
            df[col] = ""
    
    # Reorder to exact specification
    df = df[column_order]
    
    # Write with proper quoting
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f"Wrote {len(df)} papers to {output_path}")
    print(f"Column order: {list(df.columns)}")

# Database search functions (unchanged logic, but return consistent format)

def search_pubmed(species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
    """Search PubMed"""
    print(f"Searching PubMed for '{species}'...")
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    query = f'("{species}"[Title/Abstract]) AND ("{start_year}"[PDAT] : "{end_year}"[PDAT])'
    
    try:
        # Search for PMIDs
        search_response = requests.get(f"{base_url}/esearch.fcgi", params={
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }, timeout=30)
        search_response.raise_for_status()
        
        pmids = search_response.json().get('esearchresult', {}).get('idlist', [])
        if not pmids:
            print(f"No PubMed results found")
            return []
        
        print(f"Found {len(pmids)} PubMed results")
        
        # Fetch details in batches
        results = []
        batch_size = 10
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            fetch_response = requests.get(f"{base_url}/efetch.fcgi", params={
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'rettype': 'abstract'
            }, timeout=30)
            fetch_response.raise_for_status()
            
            root = ET.fromstring(fetch_response.content)
            
            for article in root.findall('.//PubmedArticle'):
                paper_data = parse_pubmed_article(article)
                if paper_data:
                    results.append(paper_data)
            
            time.sleep(0.5)
        
        print(f"Successfully parsed {len(results)} PubMed papers")
        return results
        
    except Exception as e:
        print(f"ERROR searching PubMed: {e}")
        return []

def parse_pubmed_article(article_element) -> Optional[Dict]:
    """Parse PubMed article XML"""
    try:
        # Extract PMID
        pmid_elem = article_element.find('.//PMID')
        pmid = pmid_elem.text if pmid_elem is not None else ""
        
        # Extract title
        title_elem = article_element.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ""
        
        # Extract authors
        authors = []
        for author in article_element.findall('.//Author'):
            lastname = author.find('LastName')
            forename = author.find('ForeName')
            if lastname is not None and forename is not None:
                authors.append(f"{lastname.text}, {forename.text}")
            elif lastname is not None:
                authors.append(lastname.text)
        authors_str = "; ".join(authors)
        
        # Extract journal
        journal_elem = article_element.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Extract year
        year_elem = article_element.find('.//PubDate/Year')
        year = year_elem.text if year_elem is not None else ""
        
        # Extract abstract
        abstract_parts = []
        for abstract_text in article_element.findall('.//AbstractText'):
            if abstract_text.text:
                abstract_parts.append(abstract_text.text)
        abstract = " ".join(abstract_parts)
        
        # Extract DOI
        doi = ""
        for article_id in article_element.findall('.//ArticleId'):
            if article_id.get('IdType') == 'doi':
                doi = article_id.text
                break
        
        return {
            'authors': authors_str,
            'journal': journal,
            'year': year,
            'abstract': abstract,
            'doi': doi,
            'pmid': pmid,
            'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            'database': 'PubMed',
            'title': title
        }
        
    except Exception as e:
        print(f"ERROR parsing PubMed article: {e}")
        return None

def search_crossref(species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
    """Search CrossRef"""
    print(f"Searching CrossRef for '{species}'...")
    
    try:
        response = requests.get("https://api.crossref.org/works", params={
            'query': species,
            'rows': max_results,
            'filter': f'from-pub-date:{start_year},until-pub-date:{end_year}',
            'sort': 'relevance',
            'select': 'DOI,title,author,published-print,published-online,container-title,abstract,URL'
        }, headers={
            'User-Agent': 'Academic Research Tool (mailto:researcher@example.com)'
        }, timeout=30)
        response.raise_for_status()
        
        items = response.json().get('message', {}).get('items', [])
        if not items:
            print(f"    No CrossRef results found")
            return []
        
        print(f"    Found {len(items)} CrossRef results")
        
        results = []
        for item in items:
            paper_data = parse_crossref_item(item)
            if paper_data:
                results.append(paper_data)
        
        print(f"    Successfully parsed {len(results)} CrossRef papers")
        return results
        
    except Exception as e:
        print(f"ERROR searching CrossRef: {e}")
        return []

def parse_crossref_item(item: Dict) -> Optional[Dict]:
    """Parse CrossRef item"""
    try:
        title_list = item.get('title', [])
        title = title_list[0] if title_list else ""
        
        authors = []
        for author in item.get('author', []):
            given = author.get('given', '')
            family = author.get('family', '')
            if family:
                if given:
                    authors.append(f"{family}, {given}")
                else:
                    authors.append(family)
        authors_str = "; ".join(authors)
        
        container_title = item.get('container-title', [])
        journal = container_title[0] if container_title else ""
        
        year = ""
        pub_date = item.get('published-print') or item.get('published-online')
        if pub_date and 'date-parts' in pub_date:
            date_parts = pub_date['date-parts'][0]
            if date_parts:
                year = str(date_parts[0])
        
        return {
            'authors': authors_str,
            'journal': journal,
            'year': year,
            'abstract': item.get('abstract', ''),
            'doi': item.get('DOI', ''),
            'pmid': '',
            'url': item.get('URL', ''),
            'database': 'CrossRef',
            'title': title
        }
        
    except Exception as e:
        print(f"ERROR parsing CrossRef item: {e}")
        return None

def search_biorxiv(species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
    """Search bioRxiv"""
    print(f"  Searching bioRxiv for '{species}'...")
    
    try:
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        response = requests.get(f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}", timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if 'collection' not in data:
            print(f"    No bioRxiv results found")
            return []
        
        # Filter by species name
        all_papers = data['collection']
        filtered_papers = []
        species_lower = species.lower()
        
        for paper in all_papers:
            title = paper.get('title', '').lower()
            abstract = paper.get('abstract', '').lower()
            
            if species_lower in title or species_lower in abstract:
                filtered_papers.append(paper)
                
            if len(filtered_papers) >= max_results:
                break
        
        if not filtered_papers:
            print(f"No relevant bioRxiv results found")
            return []
        
        print(f"Found {len(filtered_papers)} relevant bioRxiv results")
        
        results = []
        for paper in filtered_papers:
            results.append({
                'authors': paper.get('authors', ''),
                'journal': 'bioRxiv (preprint)',
                'year': paper.get('date', '')[:4] if paper.get('date') else '',
                'abstract': paper.get('abstract', ''),
                'doi': paper.get('doi', ''),
                'pmid': '',
                'url': f"https://www.biorxiv.org/content/{paper.get('doi', '')}v1",
                'database': 'bioRxiv',
                'title': paper.get('title', '')
            })
        
        print(f"Successfully parsed {len(results)} bioRxiv papers")
        return results
        
    except Exception as e:
        print(f"ERROR searching bioRxiv: {e}")
        return []

def search_arxiv(species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
    """Search arXiv"""
    print(f"Searching arXiv for '{species}'...")
    
    try:
        response = requests.get("http://export.arxiv.org/api/query", params={
            'search_query': f'all:"{species}"',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        
        if not entries:
            print(f"No arXiv results found")
            return []
        
        print(f"Found {len(entries)} arXiv results")
        
        results = []
        for entry in entries:
            paper_data = parse_arxiv_entry(entry, namespace, start_year, end_year)
            if paper_data:
                results.append(paper_data)
        
        print(f"Successfully parsed {len(results)} arXiv papers")
        return results
        
    except Exception as e:
        print(f"ERROR searching arXiv: {e}")
        return []

def parse_arxiv_entry(entry, namespace: Dict, start_year: int, end_year: int) -> Optional[Dict]:
    """Parse arXiv entry"""
    try:
        title_elem = entry.find('atom:title', namespace)
        title = title_elem.text.strip() if title_elem is not None else ""
        
        authors = []
        for author in entry.findall('atom:author', namespace):
            name_elem = author.find('atom:name', namespace)
            if name_elem is not None:
                authors.append(name_elem.text)
        authors_str = "; ".join(authors)
        
        published_elem = entry.find('atom:published', namespace)
        published = published_elem.text if published_elem is not None else ""
        year = published[:4] if len(published) >= 4 else ""
        
        if year and (int(year) < start_year or int(year) > end_year):
            return None
        
        summary_elem = entry.find('atom:summary', namespace)
        abstract = summary_elem.text.strip() if summary_elem is not None else ""
        
        id_elem = entry.find('atom:id', namespace)
        arxiv_url = id_elem.text if id_elem is not None else ""
        
        return {
            'authors': authors_str,
            'journal': 'arXiv (preprint)',
            'year': year,
            'abstract': abstract,
            'doi': '',
            'pmid': '',
            'url': arxiv_url,
            'database': 'arXiv',
            'title': title
        }
        
    except Exception as e:
        print(f"ERROR parsing arXiv entry: {e}")
        return None

def search_scopus(species: str, api_key: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025, inst_token: str = None) -> List[Dict]:
    """Search Scopus"""
    if not api_key:
        print(f"Skipping Scopus search (no API key)")
        return []
    
    print(f"Searching Scopus for '{species}'...")
    
    try:
        headers = {
            'X-ELS-APIKey': api_key,
            'Accept': 'application/json'
        }
        
        if inst_token:
            headers['X-ELS-Insttoken'] = inst_token
        
        response = requests.get("https://api.elsevier.com/content/search/scopus", headers=headers, params={
            'query': f'TITLE-ABS-KEY("{species}") AND PUBYEAR > {start_year-1} AND PUBYEAR < {end_year+1}',
            'count': max_results,
            'sort': 'relevancy',
            'field': 'dc:title,dc:creator,prism:publicationName,prism:coverDate,dc:description,prism:doi,dc:identifier,prism:url'
        }, timeout=30)
        response.raise_for_status()
        
        entries = response.json().get('search-results', {}).get('entry', [])
        if not entries:
            print(f"No Scopus results found")
            return []
        
        print(f"Found {len(entries)} Scopus results")
        
        results = []
        for entry in entries:
            results.append({
                'authors': entry.get('dc:creator', ''),
                'journal': entry.get('prism:publicationName', ''),
                'year': entry.get('prism:coverDate', '')[:4] if entry.get('prism:coverDate') else '',
                'abstract': entry.get('dc:description', ''),
                'doi': entry.get('prism:doi', ''),
                'pmid': '',
                'url': entry.get('prism:url', ''),
                'database': 'Scopus',
                'title': entry.get('dc:title', '')
            })
        
        print(f"Successfully parsed {len(results)} Scopus papers")
        return results
        
    except Exception as e:
        print(f"ERROR searching Scopus: {e}")
        return []

def remove_duplicates(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers"""
    if not papers:
        return []
    
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in papers:
        doi = paper.get('doi', '').strip()
        title = paper.get('title', '').strip().lower()
        
        # Check DOI duplicates
        if doi and doi in seen_dois:
            continue
        
        # Check title similarity
        title_words = set(title.split())
        is_duplicate = False
        
        for seen_title in seen_titles:
            seen_words = set(seen_title.split())
            if title_words and seen_words:
                overlap = len(title_words & seen_words) / len(title_words | seen_words)
                if overlap > 0.8:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_papers.append(paper)
            if doi:
                seen_dois.add(doi)
            if title:
                seen_titles.add(title)
    
    return unique_papers

def main():
    """Main search function"""
    parser = argparse.ArgumentParser(description='Multi-database animal research paper search')
    
    parser.add_argument('--species', '-s', type=str, required=True, help='Animal species to search for')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--start-year', '-y1', type=int, default=2015, help='Start year')
    parser.add_argument('--end-year', '-y2', type=int, default=2025, help='End year')
    parser.add_argument('--max-results', '-m', type=int, default=25, help='Maximum results per database')
    parser.add_argument('--scopus-key', '-sk', type=str, default=None, help='Scopus API key')
    parser.add_argument('--scopus-token', '-st', type=str, default=None, help='Scopus institutional token')
    
    args = parser.parse_args()
    
    print(f" Multi-database search for: {args.species}")
    print(f" Year range: {args.start_year} - {args.end_year}")
    print(f" Max results per database: {args.max_results}")
    print(f" Output: {args.output}")
    print("=" * 60)
    
    all_papers = []
    
    # Search databases
    databases = [
        ("PubMed", lambda: search_pubmed(args.species, args.max_results, args.start_year, args.end_year)),
        ("CrossRef", lambda: search_crossref(args.species, args.max_results, args.start_year, args.end_year)),
        ("bioRxiv", lambda: search_biorxiv(args.species, args.max_results, args.start_year, args.end_year)),
        ("arXiv", lambda: search_arxiv(args.species, args.max_results, args.start_year, args.end_year)),
    ]
    
    if args.scopus_key:
        databases.append(("Scopus", lambda: search_scopus(args.species, args.scopus_key, args.max_results, args.start_year, args.end_year, args.scopus_token)))
    
    for db_name, search_func in databases:
        print(f"\n Searching {db_name}...")
        try:
            results = search_func()
            all_papers.extend(results)
            print(f"* {db_name}: {len(results)} papers found")
        except Exception as e:
            print(f"*** {db_name}: ERROR - {e}")
        
        time.sleep(1)
    
    print(f"\n Total papers before deduplication: {len(all_papers)}")
    
    # Remove duplicates
    unique_papers = remove_duplicates(all_papers)
    print(f"Unique papers after deduplication: {len(unique_papers)}")
    
    # Write results
    safe_write_papers_csv(unique_papers, args.output)
    
    if unique_papers:
        print(f"\n Successfully saved {len(unique_papers)} papers")
        
        # Show database breakdown
        db_counts = {}
        for paper in unique_papers:
            db = paper.get('database', 'Unknown')
            db_counts[db] = db_counts.get(db, 0) + 1
        
        print("\n PAPERS by database:")
        for db, count in sorted(db_counts.items()):
            print(f"  {db}: {count}")
    else:
        print(f"\n  NO PAPERS found for '{args.species}'")

if __name__ == "__main__":
    main()
