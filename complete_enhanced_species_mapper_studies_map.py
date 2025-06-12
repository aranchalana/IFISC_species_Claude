#!/usr/bin/env python3
"""
Enhanced Species Density Mapping Tool - Complete Version with Satellite Maps & Species Filtering

This tool analyzes species occurrence data and creates organized outputs focused on both 
density analysis and species abundance distributions. Features intelligent extraction of 
geographic references, satellite imagery, and species-specific filtering.

FEATURES:
- Satellite imagery options (Esri, Google, Ocean basemaps)
- Species filtering by query_species or species columns
- Intelligent location extraction from complex descriptions
- Interactive maps with area/point distinction
- Comprehensive biodiversity analysis
- Single species global distribution maps
- Study density mapping (research effort visualization)
- Abundance density mapping (population distribution)

Author: Enhanced Biodiversity Research Tool
Version: 4.0 (Complete with Study Density)
Date: 2024

Requirements:
    pip install pandas folium geopy matplotlib seaborn numpy
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os
import time
import warnings
import re
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

warnings.filterwarnings('ignore')


class GeographicAreaHandler:
    """Handles detection and processing of broad geographic areas."""
    
    def __init__(self):
        # Keywords that indicate broad geographic areas
        self.area_keywords = [
            'western', 'eastern', 'northern', 'southern', 'central', 'southwest', 'southeast', 
            'northwest', 'northeast', 'west', 'east', 'north', 'south', 'sw', 'se', 'nw', 'ne',
            'sea', 'ocean', 'waters', 'basin', 'gulf', 'bay', 'strait', 'channel',
            'mediterranean', 'atlantic', 'pacific', 'indian', 'arctic', 'antarctic',
            'caribbean', 'north sea', 'baltic', 'red sea', 'black sea',
            'coast', 'coastal', 'offshore', 'continental shelf', 'slope', 'rise',
            'tropical', 'subtropical', 'temperate', 'polar', 'equatorial',
            'region', 'area', 'zone', 'sector', 'part', 'portion', 'vicinity'
        ]
        
        # Pre-defined coordinates for major geographic areas
        self.known_areas = {
            'mediterranean': {'lat': 35.0, 'lon': 18.0, 'type': 'sea'},
            'western mediterranean': {'lat': 38.0, 'lon': 3.0, 'type': 'sea_region'},
            'eastern mediterranean': {'lat': 34.0, 'lon': 30.0, 'type': 'sea_region'},
            'southern mediterranean': {'lat': 32.0, 'lon': 15.0, 'type': 'sea_region'},
            'northern mediterranean': {'lat': 42.0, 'lon': 15.0, 'type': 'sea_region'},
            'western atlantic': {'lat': 30.0, 'lon': -60.0, 'type': 'ocean_region'},
            'eastern atlantic': {'lat': 30.0, 'lon': -15.0, 'type': 'ocean_region'},
            'north atlantic': {'lat': 50.0, 'lon': -30.0, 'type': 'ocean_region'},
            'south atlantic': {'lat': -20.0, 'lon': -20.0, 'type': 'ocean_region'},
            'indian ocean': {'lat': -20.0, 'lon': 75.0, 'type': 'ocean'},
            'western indian ocean': {'lat': -10.0, 'lon': 60.0, 'type': 'ocean_region'},
            'eastern indian ocean': {'lat': -15.0, 'lon': 100.0, 'type': 'ocean_region'},
            'southern indian ocean': {'lat': -35.0, 'lon': 80.0, 'type': 'ocean_region'},
            'northern indian ocean': {'lat': 10.0, 'lon': 75.0, 'type': 'ocean_region'},
            'pacific': {'lat': 0.0, 'lon': -160.0, 'type': 'ocean'},
            'north pacific': {'lat': 35.0, 'lon': -150.0, 'type': 'ocean_region'},
            'south pacific': {'lat': -20.0, 'lon': -140.0, 'type': 'ocean_region'},
            'western pacific': {'lat': 10.0, 'lon': 140.0, 'type': 'ocean_region'},
            'eastern pacific': {'lat': 10.0, 'lon': -100.0, 'type': 'ocean_region'},
            'caribbean sea': {'lat': 15.0, 'lon': -75.0, 'type': 'sea'},
            'north sea': {'lat': 56.0, 'lon': 3.0, 'type': 'sea'},
            'baltic sea': {'lat': 58.0, 'lon': 20.0, 'type': 'sea'},
            'red sea': {'lat': 22.0, 'lon': 38.0, 'type': 'sea'},
            'black sea': {'lat': 43.0, 'lon': 35.0, 'type': 'sea'},
            'gulf of naples': {'lat': 40.8, 'lon': 14.2, 'type': 'gulf'},
            'gulf of mexico': {'lat': 25.0, 'lon': -90.0, 'type': 'gulf'},
            'san francisco bay': {'lat': 37.8, 'lon': -122.3, 'type': 'bay'}
        }
    
    def is_broad_area(self, location_text):
        """Determine if a location string refers to a broad geographic area."""
        if not location_text or pd.isna(location_text):
            return False
        
        location_lower = location_text.lower().strip()
        
        # Check against known broad areas
        if location_lower in self.known_areas:
            return True
        
        # Check for area keywords
        for keyword in self.area_keywords:
            if keyword in location_lower:
                return True
        
        # Check for patterns that suggest broad areas
        area_patterns = [
            r'\b(waters?)\s+(of|off|near)\b',
            r'\b(coast|coastal)\s+(of|off|waters?)\b',
            r'\b(region|area|zone|sector)\b',
            r'\b\w+(ern|ern\s+part)\s+(sea|ocean|mediterranean|atlantic|pacific|indian)\b',
            r'\b(tropical|subtropical|temperate)\s+(waters?|region|zone)\b'
        ]
        
        for pattern in area_patterns:
            if re.search(pattern, location_lower):
                return True
        
        return False
    
    def get_area_coordinates(self, location_text):
        """Get representative coordinates for a broad geographic area."""
        location_lower = location_text.lower().strip()
        
        # Check known areas first
        if location_lower in self.known_areas:
            area_data = self.known_areas[location_lower].copy()
            area_data['is_area'] = True
            area_data['area_size'] = 'large'
            area_data['confidence'] = 'high'
            return area_data
        
        # Try to parse directional areas
        directional_areas = self.parse_directional_area(location_lower)
        if directional_areas:
            return directional_areas
        
        # Default broad area
        return {
            'lat': None,
            'lon': None,
            'is_area': True,
            'area_size': 'unknown',
            'confidence': 'low',
            'type': 'unknown_area'
        }
    
    def parse_directional_area(self, location_text):
        """Parse directional areas like 'western mediterranean'."""
        
        base_coords = {
            'mediterranean': {'lat': 35.0, 'lon': 18.0},
            'atlantic': {'lat': 30.0, 'lon': -30.0},
            'pacific': {'lat': 0.0, 'lon': -160.0},
            'indian': {'lat': -20.0, 'lon': 75.0},
            'caribbean': {'lat': 15.0, 'lon': -75.0}
        }
        
        direction_offsets = {
            'western': {'lat': 0, 'lon': -15},
            'eastern': {'lat': 0, 'lon': 15},
            'northern': {'lat': 10, 'lon': 0},
            'southern': {'lat': -10, 'lon': 0},
            'northwest': {'lat': 8, 'lon': -12},
            'northeast': {'lat': 8, 'lon': 12},
            'southwest': {'lat': -8, 'lon': -12},
            'southeast': {'lat': -8, 'lon': 12}
        }
        
        for water_body, coords in base_coords.items():
            if water_body in location_text:
                for direction, offset in direction_offsets.items():
                    if direction in location_text:
                        return {
                            'lat': coords['lat'] + offset['lat'],
                            'lon': coords['lon'] + offset['lon'],
                            'is_area': True,
                            'area_size': 'large',
                            'confidence': 'medium',
                            'type': f'{direction}_{water_body}_region',
                            'base_body': water_body,
                            'direction': direction
                        }
        
        return None


class LocationExtractor:
    """Handles intelligent extraction of geographic references from complex descriptions."""
    
    def extract_geographic_reference(self, location_text):
        """Extract the most geographically meaningful part from complex descriptions."""
        text_lower = location_text.lower()
        
        # Define extraction patterns in order of priority
        extraction_patterns = [
            # "X in the Y" patterns - prioritize geographic features
            (r'\b(?:in|within)\s+the\s+([^,]+(?:gulf|bay|sea|ocean|strait|channel|sound|lagoon|harbor|harbour)[^,]*)', 1),
            (r'\b(?:in|within)\s+([^,]+(?:gulf|bay|sea|ocean|strait|channel|sound|lagoon|harbor|harbour)[^,]*)', 1),
            
            # "X off Y" patterns
            (r'\boff\s+(?:the\s+)?(?:coast\s+of\s+)?([^,]+)', 1),
            
            # "X near Y" patterns
            (r'\b(?:near|close\s+to|adjacent\s+to)\s+([^,]+)', 1),
            
            # "X at Y" patterns
            (r'\b(?:at|around)\s+([^,]+)', 1),
            
            # Geographic features with descriptors
            (r'\b(?:in|within|at|near)\s+(?:the\s+)?([^,]*(?:island|peninsula|cape|point|headland|archipelago)[^,]*)', 1),
            
            # Coastal references
            (r'\b(?:along|off)\s+(?:the\s+)?(?:coast\s+of\s+)?([^,]+)', 1),
            
            # Water body references
            (r'\b(?:in|within)\s+(?:the\s+)?([^,]*(?:waters?|basin|region|area)[^,]*)', 1)
        ]
        
        for pattern, group_idx in extraction_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                extracted = match.group(group_idx).strip()
                cleaned = self.clean_extracted_location(extracted)
                
                if self.is_valid_geographic_reference(cleaned):
                    return cleaned
        
        # Try extracting standalone geographic features
        standalone_patterns = [
            r'\b([A-Z][a-zA-Z\s]+(?:Gulf|Bay|Sea|Ocean|Strait|Channel|Sound|Island|Islands|Peninsula|Cape|Harbor|Harbour))\b',
            r'\b(Gulf\s+of\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Bay\s+of\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b(Sea\s+of\s+[A-Z][a-zA-Z\s]+)\b',
            r'\b([A-Z][a-zA-Z\s]+\s+Ocean)\b'
        ]
        
        for pattern in standalone_patterns:
            matches = re.findall(pattern, location_text, re.IGNORECASE)
            if matches:
                best_match = max(matches, key=len)
                if self.is_valid_geographic_reference(best_match):
                    return best_match.strip()
        
        return None
    
    def clean_extracted_location(self, extracted_text):
        """Clean up extracted location text."""
        cleanup_patterns = [
            r'\b(?:underwater|submerged|benthic|pelagic|coastal|offshore|deep|shallow)\s+',
            r'\b(?:rocky|coral|sandy|muddy|volcanic)\s+',
            r'\b(?:outcrop|reef|formation|station|site|area|zone|region)\s*',
            r'\b(?:an?|the)\s+',
            r'\s+(?:area|zone|region|waters?)$'
        ]
        
        cleaned = extracted_text
        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        cleaned = ' '.join(cleaned.split())
        
        if len(cleaned.strip()) < 3:
            return extracted_text.strip()
        
        return cleaned.strip()
    
    def is_valid_geographic_reference(self, text):
        """Check if the extracted text is a valid geographic reference."""
        if not text or len(text.strip()) < 3:
            return False
        
        geographic_keywords = [
            'gulf', 'bay', 'sea', 'ocean', 'strait', 'channel', 'sound', 'lagoon',
            'harbor', 'harbour', 'island', 'islands', 'peninsula', 'cape', 'point',
            'coast', 'shore', 'beach', 'reef', 'atoll', 'archipelago', 'basin',
            'mediterranean', 'atlantic', 'pacific', 'indian', 'arctic', 'caribbean',
            'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern', 'western'
        ]
        
        text_lower = text.lower()
        has_geographic_keyword = any(keyword in text_lower for keyword in geographic_keywords)
        
        looks_like_place = (text[0].isupper() and 3 <= len(text) <= 50 and 
                           not text.lower().startswith(('a ', 'an ', 'the ')))
        
        return has_geographic_keyword or looks_like_place


class EnhancedSpeciesMapper:
    """Enhanced Species Density Mapper with satellite maps and species filtering."""
    
    def __init__(self, csv_file, output_dir="./", prefix="species_analysis", use_satellite=True, 
                 species_filter=None, species_column='species'):
        self.csv_file = csv_file
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.use_satellite = use_satellite
        self.species_filter = species_filter
        self.species_column = species_column  # 'species', 'query_species', or 'both'
        
        # Data containers
        self.df_original = None
        self.df_clean = None
        self.location_coords = {}
        self.area_coords = {}
        self.density_stats = None
        self.abundance_stats = None
        self.top_species_data = None
        self.area_analysis = None
        
        # Initialize handlers
        self.area_handler = GeographicAreaHandler()
        self.location_extractor = LocationExtractor()
        self.geolocator = Nominatim(user_agent="enhanced_species_mapper_v4.0")
        
        self.setup_directories()
        
    def setup_directories(self):
        """Create organized output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.density_dir = self.output_dir / "density_analysis"
        self.abundance_dir = self.output_dir / "abundance_analysis"
        self.areas_dir = self.output_dir / "geographic_areas"
        self.species_dir = self.output_dir / "species_distribution"
        self.general_dir = self.output_dir / "general_analysis"
        
        for directory in [self.density_dir, self.abundance_dir, self.areas_dir, 
                         self.species_dir, self.general_dir]:
            directory.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"   ├── density_analysis/     (🎯 Species diversity)")
        print(f"   ├── abundance_analysis/   (📊 Species abundance)")
        print(f"   ├── geographic_areas/     (🌍 Area-based analysis)")
        print(f"   ├── species_distribution/ (🐠 Single species maps)")
        print(f"   └── general_analysis/")
    
    def load_and_validate_data(self):
        """Load CSV data and validate required columns."""
        print(f"\n📊 Loading data from {self.csv_file}...")
        
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        try:
            self.df_original = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df_original)} records with {len(self.df_original.columns)} columns")
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        # Validate required columns
        required_cols = ['location']
        missing_cols = [col for col in required_cols if col not in self.df_original.columns]
        
        if missing_cols:
            print(f"Available columns: {list(self.df_original.columns)}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"📋 Available columns: {list(self.df_original.columns)}")
        
        # Check species columns
        species_cols = []
        if 'species' in self.df_original.columns:
            species_cols.append('species')
        if 'query_species' in self.df_original.columns:
            species_cols.append('query_species')
        
        if not species_cols:
            raise ValueError("No species columns found. Need 'species' or 'query_species' column.")
        
        print(f"🐠 Species columns found: {species_cols}")
        
        # Check for abundance column
        abundance_cols = ['number', 'count', 'abundance', 'individuals', 'quantity']
        self.abundance_column = None
        for col in abundance_cols:
            if col in self.df_original.columns:
                self.abundance_column = col
                print(f"🔢 Found abundance column: '{col}'")
                break
        
        if self.abundance_column is None:
            print("ℹ️ No abundance column found. Will treat each record as 1 individual.")
            self.df_original['record_count'] = 1
            self.abundance_column = 'record_count'
        
        return self.df_original
    
    def apply_species_filter(self):
        """Apply species filtering based on command line arguments."""
        if self.species_filter is None:
            print("ℹ️ No species filter applied - using all data")
            return self.df_original.copy()
        
        print(f"\n🔍 Applying species filter: '{self.species_filter}'")
        print(f"🎯 Searching in column(s): {self.species_column}")
        
        filtered_df = pd.DataFrame()
        
        if self.species_column == 'both':
            # Search in both species and query_species columns
            species_mask = self.df_original['species'].str.contains(
                self.species_filter, case=False, na=False
            ) if 'species' in self.df_original.columns else pd.Series([False] * len(self.df_original))
            
            query_mask = self.df_original['query_species'].str.contains(
                self.species_filter, case=False, na=False
            ) if 'query_species' in self.df_original.columns else pd.Series([False] * len(self.df_original))
            
            combined_mask = species_mask | query_mask
            filtered_df = self.df_original[combined_mask].copy()
            
            print(f"   📖 Matches in 'species': {species_mask.sum()}")
            print(f"   🔍 Matches in 'query_species': {query_mask.sum()}")
            print(f"   🎯 Total unique matches: {len(filtered_df)}")
            
        else:
            # Search in specific column
            if self.species_column not in self.df_original.columns:
                raise ValueError(f"Column '{self.species_column}' not found in data")
            
            mask = self.df_original[self.species_column].str.contains(
                self.species_filter, case=False, na=False
            )
            filtered_df = self.df_original[mask].copy()
            print(f"   🎯 Matches in '{self.species_column}': {len(filtered_df)}")
        
        if len(filtered_df) == 0:
            print(f"❌ No records found matching '{self.species_filter}'")
            self.show_available_species()
            sys.exit(1)
        
        # Show what was found
        if self.species_column == 'both':
            unique_species = set()
            if 'species' in filtered_df.columns:
                unique_species.update(filtered_df['species'].dropna().unique())
            if 'query_species' in filtered_df.columns:
                unique_species.update(filtered_df['query_species'].dropna().unique())
            unique_species = list(unique_species)
        else:
            unique_species = filtered_df[self.species_column].dropna().unique()
        
        print(f"✅ Found {len(unique_species)} unique species matching the filter:")
        for species in sorted(unique_species)[:10]:  # Show first 10
            count = len(filtered_df[
                (filtered_df.get('species', '') == species) | 
                (filtered_df.get('query_species', '') == species)
            ]) if self.species_column == 'both' else len(filtered_df[filtered_df[self.species_column] == species])
            print(f"   • {species} ({count} records)")
        
        if len(unique_species) > 10:
            print(f"   ... and {len(unique_species) - 10} more species")
        
        return filtered_df
    
    def show_available_species(self):
        """Show available species for filtering."""
        print("\n🐠 AVAILABLE SPECIES FOR FILTERING:")
        print("=" * 50)
        
        if 'species' in self.df_original.columns:
            species_counts = self.df_original['species'].value_counts().head(15)
            print(f"\n📖 Top 15 species in 'species' column:")
            for i, (species, count) in enumerate(species_counts.items(), 1):
                print(f"  {i:2d}. {species[:60]}... ({count} records)")
        
        if 'query_species' in self.df_original.columns:
            query_counts = self.df_original['query_species'].value_counts().head(15)
            print(f"\n🔍 Top 15 species in 'query_species' column:")
            for i, (species, count) in enumerate(query_counts.items(), 1):
                print(f"  {i:2d}. {species[:60]}... ({count} records)")
        
        print(f"\n💡 Use --species 'partial_name' to filter")
        print(f"   Example: --species 'Octopus' or --species 'vulgaris'")
    
    def determine_working_species_column(self):
        """Determine which species column to use for analysis."""
        if self.species_column == 'both':
            # Combine both columns into a new working column
            working_species = []
            for _, row in self.df_clean.iterrows():
                species_val = row.get('species', '')
                query_val = row.get('query_species', '')
                
                # Prefer 'species' if available, otherwise use 'query_species'
                if pd.notna(species_val) and species_val.strip():
                    working_species.append(species_val)
                elif pd.notna(query_val) and query_val.strip():
                    working_species.append(query_val)
                else:
                    working_species.append('Unknown')
            
            self.df_clean['working_species'] = working_species
            return 'working_species'
        else:
            return self.species_column
    
    def clean_data(self):
        """Clean and prepare data for mapping."""
        print("🧹 Cleaning data for analysis...")
        
        # Apply species filter first
        self.df_clean = self.apply_species_filter()
        
        # Clean location column
        self.df_clean['location'] = self.df_clean['location'].fillna('unknown')
        
        # Remove unusable locations
        excluded_terms = [
            'not specified', 'unknown', 'natural habitat', 
            'number not specified', 'Collection', 'Museum'
        ]
        
        mask = ~self.df_clean['location'].str.contains('|'.join(excluded_terms), 
                                                       case=False, na=False)
        self.df_clean = self.df_clean[mask].copy()
        
        # Clean abundance data
        if self.abundance_column in self.df_clean.columns:
            # Convert to numeric, handling text values
            self.df_clean[self.abundance_column] = pd.to_numeric(
                self.df_clean[self.abundance_column], errors='coerce'
            ).fillna(1)
            self.df_clean[self.abundance_column] = self.df_clean[self.abundance_column].abs()
            self.df_clean[self.abundance_column] = self.df_clean[self.abundance_column].replace(0, 1)
        
        # Determine working species column
        self.working_species_col = self.determine_working_species_column()
        
        # Classify locations as areas vs points
        self.df_clean['is_broad_area'] = self.df_clean['location'].apply(
            self.area_handler.is_broad_area
        )
        
        areas_count = self.df_clean['is_broad_area'].sum()
        points_count = len(self.df_clean) - areas_count
        total_individuals = self.df_clean[self.abundance_column].sum()
        unique_species = self.df_clean[self.working_species_col].nunique()
        
        print(f"✅ Cleaned data: {len(self.df_clean)} mappable records")
        print(f"🐠 Unique species: {unique_species}")
        print(f"👥 Total individuals: {total_individuals:.0f}")
        print(f"🌍 Geographic areas: {areas_count} records")
        print(f"📍 Specific locations: {points_count} records")
        
        return self.df_clean
    
    def add_enhanced_tile_layers(self, map_object):
        """Add comprehensive tile layers including satellite imagery."""
        
        # Standard layers
        folium.TileLayer('OpenStreetMap', name='🗺️ Street Map').add_to(map_object)
        
        if self.use_satellite:
            # Satellite options
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri Satellite', 
                name='🛰️ Satellite Imagery'
            ).add_to(map_object)
            
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite', 
                name='🌍 Google Satellite'
            ).add_to(map_object)
        
        # Specialized layers for marine data
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/Ocean_Basemap/MapServer/tile/{z}/{y}/{x}',
            attr='Esri Ocean', 
            name='🌊 Ocean Basemap'
        ).add_to(map_object)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}',
            attr='Esri Terrain', 
            name='🏔️ Terrain'
        ).add_to(map_object)
        
        return map_object
    
    def parse_location_name(self, location_text):
        """Parse and clean location text with intelligent extraction."""
        if pd.isna(location_text) or not location_text:
            return None
        
        original_text = location_text.strip()
        
        # Try intelligent extraction first
        extracted_location = self.location_extractor.extract_geographic_reference(original_text)
        if extracted_location and extracted_location != original_text:
            return extracted_location
        
        # Handle comma-separated locations
        if ',' in location_text:
            parts = [part.strip() for part in location_text.split(',')]
            for part in parts:
                if any(keyword in part.lower() for keyword in 
                      ['island', 'sea', 'ocean', 'coast', 'bay', 'gulf', 'peninsula']):
                    return part
            return parts[0] if len(parts[0]) > 2 else parts[-1]
        
        # Clean parenthetical information
        if '(' in location_text and ')' in location_text:
            main_part = location_text.split('(')[0].strip()
            paren_part = location_text.split('(')[1].split(')')[0].strip()
            return main_part if len(main_part) > 3 else paren_part
        
        return location_text.strip()
    
    def geocode_locations(self):
        """Geocode all unique locations with intelligent extraction and fallback."""
        print("🌍 Geocoding locations with intelligent extraction...")
        
        unique_locations = self.df_clean['location'].unique()
        print(f"🔍 Found {len(unique_locations)} unique locations to geocode")
        
        successful_geocodes = 0
        failed_geocodes = 0
        area_geocodes = 0
        point_geocodes = 0
        extraction_successes = 0
        
        for i, location in enumerate(unique_locations):
            if location in self.location_coords or location in self.area_coords:
                continue
            
            print(f"🔄 [{i+1}/{len(unique_locations)}] Processing: {location}")
            
            # Try intelligent location extraction
            extracted_location = self.parse_location_name(location)
            
            if not extracted_location:
                failed_geocodes += 1
                print(f"  ❌ Could not extract meaningful location")
                continue
            
            # Show extraction result
            if extracted_location != location:
                print(f"  🎯 Extracted: '{extracted_location}' from '{location}'")
                extraction_successes += 1
            
            # Check if this is a broad area
            is_area = self.area_handler.is_broad_area(location)
            
            if is_area:
                # Handle as geographic area
                area_data = self.area_handler.get_area_coordinates(location)
                
                if area_data['lat'] is not None and area_data['lon'] is not None:
                    area_data.update({
                        'display_name': f"Geographic Area: {location}",
                        'clean_name': extracted_location,
                        'original_location': location,
                        'extracted_location': extracted_location
                    })
                    self.area_coords[location] = area_data
                    successful_geocodes += 1
                    area_geocodes += 1
                    lat_val = area_data['lat']
                    lon_val = area_data['lon']
                    print(f"  🌊 Area mapped: {lat_val:.2f}, {lon_val:.2f}")
                else:
                    # Try geocoding the extracted location
                    success = self.try_geocode_with_fallback(location, extracted_location, is_area=True)
                    if success:
                        successful_geocodes += 1
                        area_geocodes += 1
                    else:
                        failed_geocodes += 1
                        
            else:
                # Handle as specific location
                success = self.try_geocode_with_fallback(location, extracted_location, is_area=False)
                if success:
                    successful_geocodes += 1
                    point_geocodes += 1
                else:
                    failed_geocodes += 1
            
            time.sleep(1)  # Rate limiting
        
        print(f"✅ Geocoding complete:")
        print(f"   📍 Specific locations: {point_geocodes} successful")
        print(f"   🌊 Geographic areas: {area_geocodes} successful")
        print(f"   🎯 Successful extractions: {extraction_successes}")
        print(f"   ❌ Failed: {failed_geocodes}")
        if successful_geocodes + failed_geocodes > 0:
            success_rate = (successful_geocodes/(successful_geocodes+failed_geocodes)*100)
            print(f"   📊 Success rate: {success_rate:.1f}%")
        
        return self.location_coords, self.area_coords
    
    def try_geocode_with_fallback(self, original_location, extracted_location, is_area=False):
        """Try geocoding with multiple fallback strategies."""
        # Strategy 1: Try the extracted location first
        result = self.attempt_geocode(extracted_location)
        if result:
            self.store_geocoding_result(original_location, result, extracted_location, is_area, "extracted")
            return True
        
        # Strategy 2: Try the original if different
        if extracted_location != original_location:
            print(f"  🔄 Trying original location as fallback...")
            result = self.attempt_geocode(original_location)
            if result:
                self.store_geocoding_result(original_location, result, original_location, is_area, "original")
                return True
        
        print(f"  ❌ All geocoding strategies failed")
        return False
    
    def attempt_geocode(self, location_text):
        """Attempt to geocode a single location string."""
        try:
            result = self.geolocator.geocode(location_text, timeout=10)
            if result:
                lat_val = result.latitude
                lon_val = result.longitude
                print(f"    ✅ Found: {lat_val:.4f}, {lon_val:.4f}")
                return result
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            print(f"    ⚠️ Geocoding error: {e}")
        
        return None
    
    def store_geocoding_result(self, original_location, geocoding_result, used_location, is_area, method):
        """Store the geocoding result in the appropriate dictionary."""
        coord_data = {
            'lat': geocoding_result.latitude,
            'lon': geocoding_result.longitude,
            'display_name': geocoding_result.address,
            'clean_name': used_location,
            'original_location': original_location,
            'geocoding_method': method
        }
        
        if is_area:
            coord_data.update({
                'is_area': True,
                'area_size': 'medium',
                'confidence': 'geocoded',
                'type': 'geocoded_area'
            })
            self.area_coords[original_location] = coord_data
            print(f"    🌊 Stored as area using {method} method")
        else:
            coord_data['is_area'] = False
            self.location_coords[original_location] = coord_data
            print(f"    📍 Stored as point using {method} method")
    
    def calculate_density_metrics(self):
        """Calculate comprehensive species density metrics."""
        print("📊 Calculating species density metrics...")
        
        density_data = []
        all_coords = {**self.location_coords, **self.area_coords}
        
        for location in self.df_clean['location'].unique():
            if location not in all_coords:
                continue
            
            location_records = self.df_clean[self.df_clean['location'] == location]
            coords = all_coords[location]
            
            is_area = coords.get('is_area', False)
            unique_species_count = location_records[self.working_species_col].nunique()
            total_records = len(location_records)
            total_individuals = location_records[self.abundance_column].sum()
            
            metrics = {
                'location': location,
                'clean_location': coords['clean_name'],
                'latitude': coords['lat'],
                'longitude': coords['lon'],
                'is_geographic_area': is_area,
                'area_type': coords.get('type', 'unknown') if is_area else 'specific_location',
                'confidence': coords.get('confidence', 'high') if is_area else 'high',
                'total_records': total_records,
                'total_individuals': total_individuals,
                'unique_species': unique_species_count,
                'density_ratio': unique_species_count / total_records if total_records > 0 else 0,
                'abundance_ratio': total_individuals / total_records if total_records > 0 else 0,
                'species_list': list(location_records[self.working_species_col].unique()),
                'most_common_species': location_records[self.working_species_col].mode().iloc[0] if len(location_records) > 0 else 'Unknown'
            }
            
            density_data.append(metrics)
        
        self.density_stats = pd.DataFrame(density_data)
        
        if len(self.density_stats) > 0:
            self.density_stats['diversity_rank'] = self.density_stats['unique_species'].rank(ascending=False)
            self.density_stats['abundance_rank'] = self.density_stats['total_individuals'].rank(ascending=False)
            self.density_stats = self.density_stats.sort_values('unique_species', ascending=False)
            
            areas_data = self.density_stats[self.density_stats['is_geographic_area'] == True]
            points_data = self.density_stats[self.density_stats['is_geographic_area'] == False]
            
            print(f"✅ Calculated density metrics for {len(self.density_stats)} locations")
            print(f"   🌊 Geographic areas: {len(areas_data)} locations")
            print(f"   📍 Specific points: {len(points_data)} locations")
        
        return self.density_stats
    
    def create_density_heatmap(self):
        """Create interactive density heatmap with satellite imagery."""
        print("🗺️ Creating species diversity density heatmap...")
        
        if self.density_stats is None or len(self.density_stats) == 0:
            print("❌ No density data available for heatmap")
            return None
        
        center_lat = self.density_stats['latitude'].mean()
        center_lon = self.density_stats['longitude'].mean()
        
        # Create base map (no default tiles)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)
        
        # Add enhanced tile layers
        self.add_enhanced_tile_layers(m)
        
        # Prepare heatmap data
        heat_data = []
        max_species = self.density_stats['unique_species'].max()
        
        for _, row in self.density_stats.iterrows():
            weight = (row['unique_species'] / max_species) * 100
            heat_data.append([row['latitude'], row['longitude'], weight])
        
        # Add density heatmap
        HeatMap(heat_data, radius=30, blur=25, max_zoom=18).add_to(m)
        
        # Add markers
        for _, row in self.density_stats.iterrows():
            is_area = row.get('is_geographic_area', False)
            total_individuals = row['total_individuals']
            
            popup_html = f"""
            <div style="width: 350px; font-family: Arial; line-height: 1.4;">
                <h3>{'🌊' if is_area else '📍'} {row['location'][:50]}</h3>
                <p><strong>Type:</strong> {'Geographic Area' if is_area else 'Specific Location'}</p>
                <p><strong>Species:</strong> {row['unique_species']} unique</p>
                <p><strong>Individuals:</strong> {total_individuals:.0f}</p>
                <p><strong>Records:</strong> {row['total_records']}</p>
            </div>
            """
            
            # Color and size based on species count
            density_percentile = row['unique_species'] / max_species
            if density_percentile > 0.7:
                color, size = 'red', 12
            elif density_percentile > 0.4:
                color, size = 'orange', 10
            elif density_percentile > 0.2:
                color, size = 'blue', 8
            else:
                color, size = 'gray', 6
            
            # Different styling for areas vs points
            if is_area:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size * 1.2,
                    popup=folium.Popup(popup_html, max_width=380),
                    color='white', fillColor=color, fillOpacity=0.6,
                    weight=3, dashArray='5, 5'
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size,
                    popup=folium.Popup(popup_html, max_width=380),
                    color='white', fillColor=color, fillOpacity=0.8, weight=2
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        heatmap_path = self.density_dir / f"{self.prefix}_species_diversity_density_map.html"
        m.save(str(heatmap_path))
        print(f"✅ Species diversity density heatmap saved: {heatmap_path}")
        
        return m
    
    def create_study_density_heatmap(self):
        """Create interactive heatmap showing research effort (number of studies per location)."""
        print("🔬 Creating study density heatmap...")
        
        if self.density_stats is None or len(self.density_stats) == 0:
            print("❌ No density data available for study heatmap")
            return None
        
        center_lat = self.density_stats['latitude'].mean()
        center_lon = self.density_stats['longitude'].mean()
        
        # Create base map (no default tiles)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)
        
        # Add enhanced tile layers
        self.add_enhanced_tile_layers(m)
        
        # Prepare heatmap data based on study count (total_records)
        heat_data = []
        max_studies = self.density_stats['total_records'].max()
        
        for _, row in self.density_stats.iterrows():
            # Weight based on number of studies, not species diversity
            weight = (row['total_records'] / max_studies) * 100
            heat_data.append([row['latitude'], row['longitude'], weight])
        
        # Add study density heatmap
        HeatMap(heat_data, radius=30, blur=25, max_zoom=18).add_to(m)
        
        # Add markers sized and colored by study count
        for _, row in self.density_stats.iterrows():
            is_area = row.get('is_geographic_area', False)
            study_count = row['total_records']
            total_individuals = row['total_individuals']
            
            popup_html = f"""
            <div style="width: 350px; font-family: Arial; line-height: 1.4;">
                <h3>🔬 {'🌊' if is_area else '📍'} {row['location'][:50]}</h3>
                <div style="background: #e8f4fd; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <p><strong>🔬 Studies:</strong> {study_count} research records</p>
                    <p><strong>🐠 Species:</strong> {row['unique_species']} unique</p>
                    <p><strong>👥 Individuals:</strong> {total_individuals:.0f} total</p>
                </div>
                <p><strong>Type:</strong> {'Geographic Area' if is_area else 'Specific Location'}</p>
                <p><strong>Research Intensity:</strong> {(study_count/max_studies*100):.1f}% of max</p>
                <p><strong>Most Studied Species:</strong> {row['most_common_species']}</p>
            </div>
            """
            
            # Color and size based on study count
            study_percentile = row['total_records'] / max_studies
            if study_percentile > 0.7:
                color, size = 'purple', 14  # Highest research activity
            elif study_percentile > 0.4:
                color, size = 'red', 12     # High research activity
            elif study_percentile > 0.2:
                color, size = 'orange', 10  # Moderate research activity
            else:
                color, size = 'blue', 8     # Low research activity
            
            # Different styling for areas vs points
            if is_area:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size * 1.2,
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"🔬 {study_count} studies at {row['location'][:30]}",
                    color='white', fillColor=color, fillOpacity=0.7,
                    weight=3, dashArray='5, 5'
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size,
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"🔬 {study_count} studies at {row['location'][:30]}",
                    color='white', fillColor=color, fillOpacity=0.8, weight=2
                ).add_to(m)
        
        # Add research effort legend
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 250px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 15px;">
            <h4>🔬 Research Effort Map</h4>
            <p><strong>Total Studies:</strong> {self.density_stats['total_records'].sum()}</p>
            <p><strong>Max at Single Location:</strong> {max_studies}</p>
            <hr>
            <h5>Study Density Scale:</h5>
            <p><span style="color: purple; font-size: 16px;">●</span> Very High (70%+ of max)</p>
            <p><span style="color: red; font-size: 16px;">●</span> High (40-70%)</p>
            <p><span style="color: orange; font-size: 16px;">●</span> Moderate (20-40%)</p>
            <p><span style="color: blue; font-size: 16px;">●</span> Low (<20%)</p>
            <hr>
            <p style="font-size: 10px;">🌊 Dashed = Geographic areas<br>
            📍 Solid = Specific locations<br>
            Size = Relative study count<br>
            Heat = Research intensity</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        
        # Save study density map
        study_map_path = self.density_dir / f"{self.prefix}_study_density_map.html"
        m.save(str(study_map_path))
        print(f"✅ Study density heatmap saved: {study_map_path}")
        
        return m
    
    def create_abundance_density_heatmap(self):
        """Create interactive heatmap showing total abundance (number of individuals per location)."""
        print("👥 Creating abundance density heatmap...")
        
        if self.density_stats is None or len(self.density_stats) == 0:
            print("❌ No density data available for abundance heatmap")
            return None
        
        center_lat = self.density_stats['latitude'].mean()
        center_lon = self.density_stats['longitude'].mean()
        
        # Create base map (no default tiles)
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)
        
        # Add enhanced tile layers
        self.add_enhanced_tile_layers(m)
        
        # Prepare heatmap data based on total individuals
        heat_data = []
        max_individuals = self.density_stats['total_individuals'].max()
        
        for _, row in self.density_stats.iterrows():
            # Weight based on number of individuals
            weight = (row['total_individuals'] / max_individuals) * 100
            heat_data.append([row['latitude'], row['longitude'], weight])
        
        # Add abundance heatmap
        HeatMap(heat_data, radius=30, blur=25, max_zoom=18).add_to(m)
        
        # Add markers sized and colored by abundance
        for _, row in self.density_stats.iterrows():
            is_area = row.get('is_geographic_area', False)
            total_individuals = row['total_individuals']
            study_count = row['total_records']
            
            popup_html = f"""
            <div style="width: 350px; font-family: Arial; line-height: 1.4;">
                <h3>👥 {'🌊' if is_area else '📍'} {row['location'][:50]}</h3>
                <div style="background: #f0fff0; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <p><strong>👥 Total Individuals:</strong> {total_individuals:.0f}</p>
                    <p><strong>🐠 Species:</strong> {row['unique_species']} unique</p>
                    <p><strong>🔬 Studies:</strong> {study_count} records</p>
                </div>
                <p><strong>Type:</strong> {'Geographic Area' if is_area else 'Specific Location'}</p>
                <p><strong>Population Density:</strong> {(total_individuals/max_individuals*100):.1f}% of max</p>
                <p><strong>Avg per Study:</strong> {(total_individuals/study_count):.1f} individuals</p>
            </div>
            """
            
            # Color and size based on abundance
            abundance_percentile = row['total_individuals'] / max_individuals
            if abundance_percentile > 0.7:
                color, size = 'darkgreen', 14  # Highest abundance
            elif abundance_percentile > 0.4:
                color, size = 'green', 12      # High abundance
            elif abundance_percentile > 0.2:
                color, size = 'yellow', 10     # Moderate abundance
            else:
                color, size = 'lightblue', 8   # Low abundance
            
            # Different styling for areas vs points
            if is_area:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size * 1.2,
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"👥 {total_individuals:.0f} individuals at {row['location'][:30]}",
                    color='white', fillColor=color, fillOpacity=0.7,
                    weight=3, dashArray='5, 5'
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=size,
                    popup=folium.Popup(popup_html, max_width=380),
                    tooltip=f"👥 {total_individuals:.0f} individuals at {row['location'][:30]}",
                    color='white', fillColor=color, fillOpacity=0.8, weight=2
                ).add_to(m)
        
        # Add abundance legend
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 250px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 15px;">
            <h4>👥 Population Abundance Map</h4>
            <p><strong>Total Individuals:</strong> {self.density_stats['total_individuals'].sum():.0f}</p>
            <p><strong>Max at Single Location:</strong> {max_individuals:.0f}</p>
            <hr>
            <h5>Abundance Scale:</h5>
            <p><span style="color: darkgreen; font-size: 16px;">●</span> Very High (70%+ of max)</p>
            <p><span style="color: green; font-size: 16px;">●</span> High (40-70%)</p>
            <p><span style="color: yellow; font-size: 16px;">●</span> Moderate (20-40%)</p>
            <p><span style="color: lightblue; font-size: 16px;">●</span> Low (<20%)</p>
            <hr>
            <p style="font-size: 10px;">🌊 Dashed = Geographic areas<br>
            📍 Solid = Specific locations<br>
            Size = Relative abundance<br>
            Heat = Population density</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        folium.LayerControl().add_to(m)
        
        # Save abundance map
        abundance_map_path = self.abundance_dir / f"{self.prefix}_abundance_density_map.html"
        m.save(str(abundance_map_path))
        print(f"✅ Abundance density heatmap saved: {abundance_map_path}")
        
        return m
    
    def create_single_species_distribution_map(self, species_name):
        """Create a map showing global distribution of a specific species."""
        print(f"🐠 Creating distribution map for: {species_name}")
        
        # Filter data for specific species
        species_data = self.df_clean[
            self.df_clean[self.working_species_col].str.contains(species_name, case=False, na=False)
        ].copy()
        
        if len(species_data) == 0:
            print(f"❌ No data found for species: {species_name}")
            return None
        
        print(f"📊 Found {len(species_data)} records for {species_name}")
        
        # Get all coordinates
        all_coords = {**self.location_coords, **self.area_coords}
        if not all_coords:
            print("❌ No coordinates available")
            return None
        
        # Calculate map center from species locations only
        species_lats, species_lons = [], []
        for _, record in species_data.iterrows():
            location = record['location']
            if location in all_coords:
                coords = all_coords[location]
                if coords['lat'] is not None and coords['lon'] is not None:
                    species_lats.append(coords['lat'])
                    species_lons.append(coords['lon'])
        
        if not species_lats:
            print(f"❌ No mappable locations for {species_name}")
            return None
        
        center_lat = np.mean(species_lats)
        center_lon = np.mean(species_lons)
        
        # Create enhanced map with satellite imagery
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles=None)
        self.add_enhanced_tile_layers(m)
        
        # Calculate abundance statistics for this species
        total_individuals = species_data[self.abundance_column].sum()
        total_locations = species_data['location'].nunique()
        max_abundance = species_data[self.abundance_column].max()
        
        # Add species markers
        for _, record in species_data.iterrows():
            location = record['location']
            abundance = record[self.abundance_column]
            
            if location in all_coords:
                coords = all_coords[location]
                is_area = coords.get('is_area', False)
                
                # Marker size based on abundance (relative to max for this species)
                size_factor = (abundance / max_abundance) if max_abundance > 0 else 0.5
                marker_size = 5 + (size_factor * 15)  # Size between 5-20
                
                # Color intensity based on abundance
                if size_factor > 0.7:
                    color, opacity = 'red', 0.9
                elif size_factor > 0.4:
                    color, opacity = 'orange', 0.8
                elif size_factor > 0.1:
                    color, opacity = 'blue', 0.7
                else:
                    color, opacity = 'green', 0.6
                
                popup_html = f"""
                <div style="width: 320px; font-family: Arial; line-height: 1.4;">
                    <h3>🐠 {species_name}</h3>
                    <div style="background: #f0f8ff; padding: 10px; border-radius: 5px;">
                        <p><strong>📍 Location:</strong> {location}</p>
                        <p><strong>🔢 Abundance:</strong> {abundance:.0f} individuals</p>
                        <p><strong>🌍 Type:</strong> {'Geographic Area' if is_area else 'Specific Location'}</p>
                    </div>
                    <div style="background: #f8f9fa; padding: 8px; border-radius: 5px; margin-top: 5px;">
                        <p><strong>Species Total:</strong> {total_individuals:.0f} individuals across {total_locations} locations</p>
                    </div>
                </div>
                """
                
                # Different marker styles for areas vs points
                if is_area:
                    folium.CircleMarker(
                        location=[coords['lat'], coords['lon']],
                        radius=marker_size * 1.2,
                        popup=folium.Popup(popup_html, max_width=350),
                        tooltip=f"🌊 {location}: {abundance:.0f} {species_name}",
                        color='white', fillColor=color, fillOpacity=opacity,
                        weight=3, dashArray='5, 5'
                    ).add_to(m)
                else:
                    folium.CircleMarker(
                        location=[coords['lat'], coords['lon']],
                        radius=marker_size,
                        popup=folium.Popup(popup_html, max_width=350),
                        tooltip=f"📍 {location}: {abundance:.0f} {species_name}",
                        color='white', fillColor=color, fillOpacity=opacity,
                        weight=2
                    ).add_to(m)
        
        # Add species-specific legend
        legend_html = f'''
        <div style="position: fixed; top: 10px; right: 10px; width: 280px; 
                    background-color: white; border: 2px solid grey; z-index: 9999; 
                    font-size: 12px; padding: 15px;">
            <h4>🐠 {species_name[:30]}{"..." if len(species_name) > 30 else ""}</h4>
            <p><strong>Total Records:</strong> {len(species_data)}</p>
            <p><strong>Total Individuals:</strong> {total_individuals:.0f}</p>
            <p><strong>Locations:</strong> {total_locations}</p>
            <hr>
            <h5>Abundance Scale:</h5>
            <p><span style="color: red; font-size: 16px;">●</span> High (70%+ of max)</p>
            <p><span style="color: orange; font-size: 16px;">●</span> Medium (40-70%)</p>
            <p><span style="color: blue; font-size: 16px;">●</span> Low (10-40%)</p>
            <p><span style="color: green; font-size: 16px;">●</span> Minimal (<10%)</p>
            <hr>
            <p style="font-size: 10px;">🌊 Dashed = Geographic areas<br>
            📍 Solid = Specific locations<br>
            Size = Relative abundance</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        safe_species_name = "".join(c for c in species_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
        map_path = self.species_dir / f"{self.prefix}_distribution_{safe_species_name.replace(' ', '_')}.html"
        m.save(str(map_path))
        print(f"✅ Species distribution map saved: {map_path}")
        
        return m
    
    def save_data(self):
        """Save all analysis data."""
        print("💾 Saving analysis data...")
        
        # Save density data
        if self.density_stats is not None:
            density_path = self.density_dir / f"{self.prefix}_density_statistics.csv"
            self.density_stats.to_csv(density_path, index=False)
            print(f"✅ Density statistics saved: {density_path}")
        
        # Save cleaned data
        if self.df_clean is not None:
            clean_path = self.general_dir / f"{self.prefix}_cleaned_species_data.csv"
            self.df_clean.to_csv(clean_path, index=False)
            print(f"✅ Cleaned data saved: {clean_path}")
    
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline."""
        print("🚀 STARTING ENHANCED SPECIES ANALYSIS")
        print("="*60)
        
        try:
            # Data loading and preparation
            self.load_and_validate_data()
            self.clean_data()
            
            # Enhanced geocoding with intelligent extraction
            self.geocode_locations()
            self.calculate_density_metrics()
            
            # Create all density maps
            print(f"\n🗺️ CREATING COMPREHENSIVE MAPS...")
            print("   🐠 Creating species diversity density map...")
            self.create_density_heatmap()
            
            print("   🔬 Creating study density map...")
            self.create_study_density_heatmap()
            
            print("   👥 Creating abundance density map...")
            self.create_abundance_density_heatmap()
            
            # Save data
            print(f"\n💾 SAVING DATA...")
            self.save_data()
            
            # If single species analysis requested
            if self.species_filter:
                print(f"\n🐠 CREATING SPECIES-SPECIFIC DISTRIBUTION MAP...")
                self.create_single_species_distribution_map(self.species_filter)
            
            # Final summary
            self.print_final_summary()
            
        except Exception as e:
            print(f"❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        print("\n" + "="*70)
        print("🎉 ENHANCED SPECIES ANALYSIS COMPLETE")
        print("="*70)
        
        print(f"📁 Input Dataset: {self.csv_file}")
        print(f"📊 Original Records: {len(self.df_original) if self.df_original is not None else 0}")
        print(f"🗺️ Mappable Records: {len(self.df_clean) if self.df_clean is not None else 0}")
        
        if self.species_filter:
            print(f"🔍 Species Filter: '{self.species_filter}' in column '{self.species_column}'")
        
        if self.df_clean is not None:
            area_records = self.df_clean['is_broad_area'].sum()
            point_records = len(self.df_clean) - area_records
            print(f"   🌊 Geographic areas: {area_records} records")
            print(f"   📍 Specific locations: {point_records} records")
        
        print(f"📍 Geocoded Specific Locations: {len(self.location_coords)}")
        print(f"🌊 Geocoded Geographic Areas: {len(self.area_coords)}")
        
        if self.df_clean is not None:
            unique_species = self.df_clean[self.working_species_col].nunique()
            total_individuals = self.df_clean[self.abundance_column].sum()
            total_studies = len(self.df_clean)
            print(f"🐠 Unique Species: {unique_species}")
            print(f"👥 Total Individuals: {total_individuals:.0f}")
            print(f"🔬 Total Studies: {total_studies}")
        
        print(f"🛰️ Satellite Imagery: {'Enabled' if self.use_satellite else 'Disabled'}")
        
        if self.density_stats is not None and len(self.density_stats) > 0:
            most_diverse = self.density_stats.iloc[0]
            location_type = "Geographic Area" if most_diverse.get('is_geographic_area', False) else "Specific Location"
            print(f"🏆 Top Biodiversity Hotspot: {most_diverse['location']} ({most_diverse['unique_species']} species, {location_type})")
            
            # Find location with most studies
            most_studied_idx = self.density_stats['total_records'].idxmax()
            most_studied = self.density_stats.loc[most_studied_idx]
            most_studied_type = "Geographic Area" if most_studied.get('is_geographic_area', False) else "Specific Location"
            print(f"🔬 Most Studied Location: {most_studied['location']} ({most_studied['total_records']} studies, {most_studied_type})")
            
            # Find location with highest abundance
            most_abundant_idx = self.density_stats['total_individuals'].idxmax()
            most_abundant = self.density_stats.loc[most_abundant_idx]
            most_abundant_type = "Geographic Area" if most_abundant.get('is_geographic_area', False) else "Specific Location"
            print(f"👥 Highest Abundance Location: {most_abundant['location']} ({most_abundant['total_individuals']:.0f} individuals, {most_abundant_type})")
        
        print("\n🗺️ MAPS CREATED:")
        print("   🐠 Species Diversity Density Map - Shows biodiversity hotspots")
        print("   🔬 Study Density Map - Shows research effort distribution") 
        print("   👥 Abundance Density Map - Shows population abundance distribution")
        if self.species_filter:
            print(f"   🎯 {self.species_filter} Distribution Map - Species-specific distribution")
        
        print("\n🗺️ Open HTML files in your browser to view interactive maps")
        print("📈 Check output directories for analysis results")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='🛰️ Enhanced Species Mapping Tool with Satellite Maps & Species Filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🌟 EXAMPLES:
  # Basic analysis with all data
  python enhanced_mapper.py all_species_data.csv
  
  # Analysis with satellite maps disabled
  python enhanced_mapper.py all_species_data.csv --no-satellite
  
  # Filter by species found in papers
  python enhanced_mapper.py all_species_data.csv --species "Octopus" --species-column species
  
  # Filter by query species (search terms)
  python enhanced_mapper.py all_species_data.csv --species "vulgaris" --species-column query_species
  
  # Search in both columns
  python enhanced_mapper.py all_species_data.csv --species "Octopus" --species-column both
  
  # Show available species for filtering
  python enhanced_mapper.py all_species_data.csv --list-species
  
  # Complete example with custom output
  python enhanced_mapper.py data.csv -o results/ --prefix marine_study --species "Octopus"

📋 CSV COLUMNS EXPECTED:
  - location (required): Where species was found
  - species (optional): Species found in papers
  - query_species (optional): Species used for search
  - number (optional): Abundance/count data
  
🛰️ SATELLITE MAPS: Include multiple tile layers (Esri, Google, Ocean basemaps)
🐠 SPECIES FILTERING: Focus analysis on specific species or groups
🌍 GEOGRAPHIC AREAS: Automatic detection of broad vs specific locations
🔬 STUDY DENSITY: Visualize research effort distribution
👥 ABUNDANCE DENSITY: Show population abundance patterns
        """
    )
    
    parser.add_argument('csv_file', help='📊 Path to CSV file with species data')
    parser.add_argument('-o', '--output', default='.', help='📁 Output directory (default: current)')
    parser.add_argument('--prefix', default='species_analysis', help='📝 Output filename prefix')
    
    # Satellite map option
    parser.add_argument('--no-satellite', action='store_true', 
                       help='🗺️ Disable satellite imagery layers (faster loading)')
    
    # Species filtering options
    parser.add_argument('--species', help='🐠 Filter by species name (partial match)')
    parser.add_argument('--species-column', choices=['species', 'query_species', 'both'], 
                       default='species', help='📖 Which column to search for species filter')
    parser.add_argument('--list-species', action='store_true', 
                       help='📋 List available species and exit (no analysis)')
    
    return parser.parse_args()


def validate_input_file(csv_path):
    """Validate the input CSV file."""
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: CSV file '{csv_path}' not found!")
        return False
    
    try:
        test_df = pd.read_csv(csv_path, nrows=1)
        print(f"✅ Input file validated: {csv_path}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Cannot read CSV file - {e}")
        return False


def main():
    """Main execution function."""
    print("🛰️" + "="*75)
    print("   ENHANCED SPECIES MAPPING TOOL WITH SATELLITE MAPS & FILTERING - v4.0")
    print("   Complete Biodiversity Analysis + Satellite Imagery + Study Density")
    print("="*77)
    
    args = parse_arguments()
    
    if not validate_input_file(args.csv_file):
        sys.exit(1)
    
    # Create mapper instance
    mapper = EnhancedSpeciesMapper(
        csv_file=args.csv_file,
        output_dir=args.output,
        prefix=args.prefix,
        use_satellite=not args.no_satellite,
        species_filter=args.species,
        species_column=args.species_column
    )
    
    # Handle list species request
    if args.list_species:
        print("\n📋 LISTING AVAILABLE SPECIES...")
        mapper.load_and_validate_data()
        mapper.show_available_species()
        return
    
    print(f"\n⚙️ ANALYSIS CONFIGURATION:")
    print(f"   📊 Input File: {args.csv_file}")
    print(f"   📁 Output Directory: {args.output}")
    print(f"   📝 File Prefix: {args.prefix}")
    print(f"   🛰️ Satellite Imagery: {'Enabled' if not args.no_satellite else 'Disabled'}")
    
    if args.species:
        print(f"   🔍 Species Filter: '{args.species}'")
        print(f"   📖 Search Column: {args.species_column}")
    else:
        print(f"   🌍 Analysis Scope: All species")
    
    print("-" * 77)
    
    try:
        mapper.run_complete_analysis()
        
        print(f"\n🎉 SUCCESS! Enhanced analysis completed!")
        print(f"🛰️ Satellite imagery: {'Available' if not args.no_satellite else 'Disabled'}")
        print(f"🔬 Study density map: Available")
        print(f"👥 Abundance density map: Available")
        if args.species:
            print(f"🐠 Species-specific analysis for: {args.species}")
        print(f"📊 Check output directories for results!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
