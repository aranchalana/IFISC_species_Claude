# IFISC_species_Claude
A repository to create a data base based in species names. It looks for the species names, find with multiple searchers the research papers, and look for the information using API Claude.

**************** SEARCH SPECIES FROM RESEARCH PAPERS

1. cd SCOPUS_SPECIES_LIST_v3
2. Activate deepseek
3. Create txt with the names of the species.
4. ./batch_multi_database_pipeline_v2.sh --species-file species.txt --claude-key ***** --output-dir ./results

En este caso se utiliza Claude, y SCOPUS opcional: 

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
    
    
STEP 1: Multi-database search for papers about **** (va seleccionando las especies)
	Prueba con varias database y genera especies_papers.csv
STEP 2: Extracting species information using Claude:
	Selecciona los csv temporales que se van formando para cada especies, de especies_papers.csv y crea especies_data.csv, con los datos.

De esos dos archivos temporales, saca un archivo csv con: query_species, paper_link, species, number, study_type, location, doi, paper_title, 
en all_species_data.csv y luego un summary: batch_summary.csv.

