#!/usr/bin/env python3
"""
construct_final_csv.py - Construct final CSV from temporary species data files
Merges all temporary species data files into the final output format.
"""

import os
import pandas as pd
import csv
import glob
import argparse
from pathlib import Path

def find_temp_directories(output_dir):
    """Find all temporary directories that match the pattern temp_*"""
    temp_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.startswith('temp_'):
            temp_dirs.append(item_path)
    return temp_dirs

def find_species_data_files(search_dirs):
    """Find all *_species_data.csv files in the given directories"""
    csv_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Look for files ending with _species_data.csv
            pattern = os.path.join(search_dir, "*_species_data.csv")
            files = glob.glob(pattern)
            csv_files.extend(files)
            print(f"Found {len(files)} species data files in {search_dir}")
    
    return csv_files

def read_and_validate_csv(file_path):
    """Read a CSV file and validate its structure"""
    try:
        df = pd.read_csv(file_path, quoting=csv.QUOTE_ALL, encoding='utf-8')
        print(f"  ✓ Read {len(df)} entries from {os.path.basename(file_path)}")
        print(f"    Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"  ✗ Error reading {file_path}: {e}")
        try:
            # Fallback method
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"  ✓ Read with fallback method: {len(df)} entries")
            return df
        except Exception as e2:
            print(f"  ✗ All methods failed for {file_path}: {e2}")
            return None

def standardize_dataframe(df):
    """Standardize dataframe to expected format"""
    # Expected final columns in exact order
    final_columns = [
        'query_species',    # Species that was queried
        'paper_link',       # DOI/URL of paper
        'species',          # Scientific name of species found
        'number',           # Number of specimens (formatted)
        'study_type',       # Type of study
        'location',         # Study location
        'doi',              # DOI (backup/duplicate of paper_link)
        'paper_title'       # Title LAST to prevent CSV parsing issues
    ]
    
    # Ensure all required columns exist
    for col in final_columns:
        if col not in df.columns:
            if col == 'doi' and 'paper_link' in df.columns:
                df['doi'] = df['paper_link']
            elif col == 'paper_link' and 'doi' in df.columns:
                df['paper_link'] = df['doi']
            else:
                df[col] = "UNSPECIFIED"
    
    # Reorder columns to exact specification
    df = df[final_columns]
    
    return df

def merge_species_data_files(csv_files, output_file):
    """Merge all species data CSV files into final format"""
    if not csv_files:
        print("No CSV files found to merge")
        return False
    
    print(f"\nMerging {len(csv_files)} CSV files...")
    
    dfs = []
    total_entries = 0
    
    for file_path in csv_files:
        df = read_and_validate_csv(file_path)
        if df is not None and len(df) > 0:
            # Standardize the dataframe
            df = standardize_dataframe(df)
            dfs.append(df)
            total_entries += len(df)
        else:
            print(f"  Skipping {os.path.basename(file_path)} (no valid data)")
    
    if not dfs:
        print("No valid data found in any file")
        return False
    
    print(f"\nCombining data from {len(dfs)} files...")
    
    # Combine all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total entries after merging: {len(merged_df)}")
    print(f"Final column order: {list(merged_df.columns)}")
    
    # Save the merged data
    try:
        merged_df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
        print(f"\n✓ Successfully created final merged file: {output_file}")
        print(f"  Total species entries: {len(merged_df)}")
        
        # Show some statistics
        if 'query_species' in merged_df.columns:
            unique_species = merged_df['query_species'].nunique()
            print(f"  Unique queried species: {unique_species}")
        
        if 'species' in merged_df.columns:
            found_species = merged_df['species'].nunique()
            print(f"  Unique found species: {found_species}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving merged file: {e}")
        return False

def create_summary_file(csv_files, output_dir):
    """Create a summary file showing processing results"""
    summary_data = []
    
    for file_path in csv_files:
        df = read_and_validate_csv(file_path)
        if df is not None:
            filename = os.path.basename(file_path)
            # Extract species name from filename (remove _species_data.csv)
            species_name = filename.replace('_species_data.csv', '').replace('_', ' ')
            
            entry_count = len(df)
            summary_data.append({
                'query_species': species_name,
                'species_entries': entry_count,
                'file': filename
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(output_dir, 'reconstruction_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Created summary file: {summary_file}")
        
        # Print summary table
        print("\nProcessing Summary:")
        print("=" * 50)
        for _, row in summary_df.iterrows():
            print(f"{row['query_species']:<30} {row['species_entries']:>5} entries")
        print("=" * 50)
        print(f"Total files: {len(summary_data)}")
        print(f"Total entries: {summary_df['species_entries'].sum()}")

def main():
    parser = argparse.ArgumentParser(description='Construct final CSV from temporary species data files')
    parser.add_argument('--output-dir', '-o', type=str, default='./species_data', 
                       help='Output directory containing temp files (default: ./species_data)')
    parser.add_argument('--temp-dir', '-t', type=str, default=None, 
                       help='Specific temp directory to process (optional)')
    parser.add_argument('--output-file', '-f', type=str, default=None, 
                       help='Output CSV file (default: output_dir/all_species_data.csv)')
    
    args = parser.parse_args()
    
    print("SPECIES DATA CSV RECONSTRUCTION")
    print("=" * 60)
    
    # Determine output directory and file
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        return
    
    output_file = args.output_file or os.path.join(output_dir, 'all_species_data.csv')
    
    print(f"Output directory: {output_dir}")
    print(f"Final CSV file: {output_file}")
    
    # Find directories to search
    if args.temp_dir:
        # Use specific temp directory
        search_dirs = [os.path.abspath(args.temp_dir)]
        print(f"Using specific temp directory: {args.temp_dir}")
    else:
        # Find all temp directories
        search_dirs = find_temp_directories(output_dir)
        if not search_dirs:
            # Also check the output directory itself
            search_dirs = [output_dir]
        print(f"Found {len(search_dirs)} directories to search")
    
    # Find all species data CSV files
    csv_files = find_species_data_files(search_dirs)
    
    if not csv_files:
        print("No species data CSV files found!")
        print("Looking for files with pattern: *_species_data.csv")
        print(f"In directories: {search_dirs}")
        return
    
    print(f"\nFound {len(csv_files)} species data files:")
    for file_path in csv_files:
        print(f"  {os.path.relpath(file_path, output_dir)}")
    
    # Merge all files
    success = merge_species_data_files(csv_files, output_file)
    
    if success:
        # Create summary
        create_summary_file(csv_files, output_dir)
        
        print(f"\n✓ RECONSTRUCTION COMPLETED SUCCESSFULLY!")
        print(f"Final merged file: {output_file}")
        
        # Show final format
        print("\nFinal CSV format:")
        print("query_species, paper_link, species, number, study_type, location, doi, paper_title")
        
    else:
        print("\n✗ Reconstruction failed")

if __name__ == "__main__":
    main()
