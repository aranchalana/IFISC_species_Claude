#!/bin/bash
# batch_multi_database_pipeline.sh - FINAL CORRECTED VERSION
# EXACT column order: query_species, paper_link, species, number, study_type, location, doi, title

show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Multi-database batch pipeline - FINAL CORRECTED VERSION"
    echo "EXACT final format: query_species, paper_link, species, number, study_type, location, doi, title"
    echo
    echo "Options:"
    echo "  -f, --species-file FILE    Text file containing animal species (one per line)"
    echo "  -ck, --claude-key KEY      Claude API key (required)"
    echo "  -sk, --scopus-key KEY      Scopus API key (optional)"
    echo "  -st, --scopus-token TOKEN  Scopus institutional token (optional)"
    echo "  -o, --output-dir DIR       Output directory (default: ./species_data)"
    echo "  -y1, --start-year YEAR     Start year for search (default: 2015)"
    echo "  -y2, --end-year YEAR       End year for search (default: 2025)"
    echo "  -m, --max-results NUM      Maximum number of papers per database per species (default: 25)"
    echo "  -h, --help                 Show this help message"
    echo
}

# Set default values
SPECIES_FILE=""
CLAUDE_KEY=""
SCOPUS_KEY=""
SCOPUS_TOKEN=""
OUTPUT_DIR="./species_data"
START_YEAR="2015"
END_YEAR="2025"
MAX_RESULTS=25

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -f|--species-file)
            SPECIES_FILE="$2"
            shift 2
            ;;
        -ck|--claude-key)
            CLAUDE_KEY="$2"
            shift 2
            ;;
        -sk|--scopus-key)
            SCOPUS_KEY="$2"
            shift 2
            ;;
        -st|--scopus-token)
            SCOPUS_TOKEN="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -y1|--start-year)
            START_YEAR="$2"
            shift 2
            ;;
        -y2|--end-year)
            END_YEAR="$2"
            shift 2
            ;;
        -m|--max-results)
            MAX_RESULTS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$SPECIES_FILE" ]; then
    echo " ERROR: Species file is required"
    show_help
    exit 1
fi

if [ ! -f "$SPECIES_FILE" ]; then
    echo " ERROR: Species file '$SPECIES_FILE' does not exist"
    exit 1
fi

if [ -z "$CLAUDE_KEY" ]; then
    echo " ERROR: Claude API key is required"
    show_help
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create temporary directory
TEMP_DIR="$OUTPUT_DIR/temp_$$"
mkdir -p "$TEMP_DIR"

# Create summary file
SUMMARY_FILE="$OUTPUT_DIR/batch_summary.csv"
echo "query_species,papers_found,species_entries" > "$SUMMARY_FILE"

# Read species count
SPECIES_COUNT=$(wc -l < "$SPECIES_FILE")
echo "Found $SPECIES_COUNT species in the file"

# Show configuration
echo
echo "** DATABASES TO BE SEARCHED: **"
echo "PubMed (free, life sciences)"
echo "CrossRef (free, broad coverage)"
echo "bioRxiv (free, biology preprints)"
echo "arXiv (free, preprints)"
if [ ! -z "$SCOPUS_KEY" ]; then
    echo " Scopus (with API key)"
else
    echo "ERROR Scopus (no API key provided)"
fi
echo
echo "FINAL OUTPUT FORMAT:"
echo "query_species, paper_link, species, number, study_type, location, doi, title"
echo

# Initialize array for species data files
declare -a SPECIES_DATA_FILES

# Process each species
CURRENT_SPECIES=0
while IFS= read -r SPECIES || [ -n "$SPECIES" ]; do
    # Skip empty lines
    if [ -z "$SPECIES" ]; then
        continue
    fi

    # Increment counter
    CURRENT_SPECIES=$((CURRENT_SPECIES + 1))
    
    # Trim whitespace
    SPECIES=$(echo "$SPECIES" | xargs)
    
    echo
    echo "============================================================"
    echo " PROCESSING SPECIES $CURRENT_SPECIES/$SPECIES_COUNT: '$SPECIES'"
    echo "============================================================"
    
    # Set file paths
    SPECIES_SAFE=$(echo "$SPECIES" | tr ' ' '_')
    SEARCH_CSV="$TEMP_DIR/${SPECIES_SAFE}_papers.csv"
    SPECIES_DATA_CSV="$TEMP_DIR/${SPECIES_SAFE}_species_data.csv"

    echo "STEP 1: Multi-database search for papers about '$SPECIES'"

    # Build search arguments
    SEARCH_ARGS="--species \"$SPECIES\" --output \"$SEARCH_CSV\" --start-year $START_YEAR --end-year $END_YEAR --max-results $MAX_RESULTS"
    
    if [ ! -z "$SCOPUS_KEY" ]; then
        SEARCH_ARGS="$SEARCH_ARGS --scopus-key $SCOPUS_KEY"
    fi
    
    if [ ! -z "$SCOPUS_TOKEN" ]; then
        SEARCH_ARGS="$SEARCH_ARGS --scopus-token $SCOPUS_TOKEN"
    fi

    # Run search script
    eval "python 1_multi_database_animal_search_v2.py $SEARCH_ARGS"

    # Check if search was successful
    if [ ! -f "$SEARCH_CSV" ]; then
        echo "  Warning: Multi-database search failed for '$SPECIES'. Skipping."
        echo "\"$SPECIES\",0,0" >> "$SUMMARY_FILE"
        continue
    fi

    # Count papers found (subtract 1 for header)
    PAPERS_FOUND=$(($(wc -l < "$SEARCH_CSV") - 1))
    echo " Found $PAPERS_FOUND papers for '$SPECIES'"
    
    echo
    echo " STEP 2: Extracting species information using Claude"

    # Build extractor arguments
    EXTRACTOR_ARGS="--input-csv \"$SEARCH_CSV\" --output-csv \"$SPECIES_DATA_CSV\" --claude-key \"$CLAUDE_KEY\""
    
    if [ ! -z "$SCOPUS_KEY" ]; then
        EXTRACTOR_ARGS="$EXTRACTOR_ARGS --scopus-key \"$SCOPUS_KEY\""
    fi
    
    if [ ! -z "$SCOPUS_TOKEN" ]; then
        EXTRACTOR_ARGS="$EXTRACTOR_ARGS --inst-token \"$SCOPUS_TOKEN\""
    fi

    # Run species extractor
    eval "python 2_simple_species_extractor_v2.py $EXTRACTOR_ARGS"
    
    # Check if extraction was successful
    if [ ! -f "$SPECIES_DATA_CSV" ]; then
        echo "Warning: Species extraction failed for '$SPECIES'."
        echo "\"$SPECIES\",$PAPERS_FOUND,0" >> "$SUMMARY_FILE"
        continue
    fi

    # Add query_species column and process with proper CSV handling
    python - <<EOF
import pandas as pd
import csv

try:
    # Read species data CSV
    df = pd.read_csv("$SPECIES_DATA_CSV", quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f" Read {len(df)} species entries")
    
    # Add query_species as FIRST column
    df.insert(0, 'query_species', "$SPECIES")
    
    # Ensure we have doi column (copy from paper_link)
    if 'doi' not in df.columns and 'paper_link' in df.columns:
        df['doi'] = df['paper_link']
    elif 'paper_link' not in df.columns and 'doi' in df.columns:
        df['paper_link'] = df['doi']
    
    # EXACT final column order as specified
    final_columns = [
        'query_species',    # Species queried
        'paper_link',       # DOI/URL of paper  
        'species',          # Scientific name found
        'number',           # Number of specimens
        'study_type',       # Laboratory/Field/Field+Laboratory
        'location',         # Study location
        'doi',              # DOI (duplicate of paper_link for compatibility)
        'paper_title'       # Title LAST to prevent CSV issues
    ]
    
    # Ensure all columns exist
    for col in final_columns:
        if col not in df.columns:
            df[col] = "UNSPECIFIED"
    
    # Reorder columns exactly
    df = df[final_columns]
    
    # Save with proper quoting
    df.to_csv("$SPECIES_DATA_CSV", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    
    species_entries = len(df)
    print(f" Final format: {list(df.columns)}")
    print(f" Saved {species_entries} entries for '$SPECIES'")
    
    # Update summary
    with open("$SUMMARY_FILE", "a") as f:
        f.write(f'"$SPECIES",$PAPERS_FOUND,{species_entries}\\n')
    
except Exception as e:
    print(f" Error processing species data: {e}")
    import traceback
    traceback.print_exc()
    with open("$SUMMARY_FILE", "a") as f:
        f.write(f'"$SPECIES",$PAPERS_FOUND,0\\n')
EOF

    # Add to merge list
    if [ -f "$SPECIES_DATA_CSV" ]; then
        SPECIES_DATA_FILES+=("$SPECIES_DATA_CSV")
    fi

    echo "Finished processing '$SPECIES'"

done < "$SPECIES_FILE"

# Create final merged file
ALL_SPECIES_CSV="$OUTPUT_DIR/all_species_data.csv"
echo
echo "Creating final merged data file..."

python - <<EOF
import pandas as pd
import os
import csv

try:
    # List of species data files
    csv_files = [$(printf '"%s",' "${SPECIES_DATA_FILES[@]}" | sed 's/,$//')]
    
    if not csv_files:
        print("  No species data files found to merge")
        exit(0)
    
    # Read and concatenate all files
    dfs = []
    for file in csv_files:
        try:
            if os.path.exists(file):
                df = pd.read_csv(file, quoting=csv.QUOTE_ALL, encoding='utf-8')
                dfs.append(df)
                print(f" Added {len(df)} entries from {os.path.basename(file)}")
        except Exception as e:
            print(f" Error reading {file}: {e}")
    
    if not dfs:
        print(" No valid data found in any file")
        exit(0)
    
    # Combine all data
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # EXACT final column order
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
    
    # Ensure all columns exist
    for col in final_columns:
        if col not in merged_df.columns:
            merged_df[col] = "UNSPECIFIED"
    
    # Reorder to exact specification
    merged_df = merged_df[final_columns]
    
    # Save final merged data
    merged_df.to_csv("$ALL_SPECIES_CSV", index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f" Successfully created final merged file with {len(merged_df)} total species entries")
    print(f" Final column order: {list(merged_df.columns)}")
    
except Exception as e:
    print(f" Error creating merged file: {e}")
    import traceback
    traceback.print_exc()
EOF

# Clean up temporary files
echo " Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# Print final summary
echo
echo "============================================================"
echo " MULTI-DATABASE BATCH PROCESSING COMPLETE - FINAL VERSION"
echo "============================================================"
echo
echo "SUMMARY:"
echo "- Processed species: $CURRENT_SPECIES"
echo "- Output directory: $OUTPUT_DIR"
echo "- Summary file: $SUMMARY_FILE"
echo "- All species data: $ALL_SPECIES_CSV"
echo
echo "FINAL OUTPUT FORMAT:"
echo "   query_species, paper_link, species, number, study_type, location, doi, title"
echo
echo "Databases searched:"
echo "   PubMed (life sciences)"
echo "   CrossRef (broad coverage)"
echo "   bioRxiv (biology preprints)"
echo "   arXiv (preprints)"
if [ ! -z "$SCOPUS_KEY" ]; then
    echo "Scopus (with API key)"
else
    echo "Scopus (no API key)"
fi
echo
echo "Species processing results:"
echo "----------------------------------------"
cat "$SUMMARY_FILE"
echo "----------------------------------------"
echo
echo "Files created:"
echo "1. $ALL_SPECIES_CSV - All species data with correct column ordering"
echo "2. $SUMMARY_FILE - Processing summary"
echo
echo "Number formats used:"
echo "- Simple counts: 20, 230, etc."
echo "- Density: 20 indv/m2, 15 indv/ha"
echo "- Multiple groups: separate lines"
echo "- Unknown counts: 'unknown'"
echo
echo "All files located in: $(cd "$OUTPUT_DIR" && pwd)"
echo
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "  No more column misalignment issues - title is safely positioned last."
echo
