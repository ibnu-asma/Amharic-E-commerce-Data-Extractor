import os
import sys
import json
import pandas as pd
import glob
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocessing.preprocess import preprocess_amharic_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_raw_file(raw_file_path):
    """Process a single raw JSON file and save as CSV."""
    try:
        # Load raw data
        with open(raw_file_path, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        if not messages:
            logging.warning(f"No messages found in {raw_file_path}")
            return
        
        # Create DataFrame and process text
        df = pd.DataFrame(messages)
        df['processed_text'] = df['text'].apply(preprocess_amharic_text)
        
        # Generate output path
        os.makedirs('data/processed', exist_ok=True)
        base_name = os.path.basename(raw_file_path).replace('.json', '.csv')
        processed_path = f"data/processed/{base_name}"
        
        # Save processed CSV
        df.to_csv(processed_path, index=False, encoding='utf-8')
        logging.info(f"Processed {raw_file_path} -> {processed_path}")
        
    except Exception as e:
        logging.error(f"Error processing {raw_file_path}: {e}")

def process_all_raw_files():
    """Process all raw JSON files in data/raw/ directory."""
    raw_files = glob.glob('data/raw/*.json')
    
    if not raw_files:
        logging.warning("No raw JSON files found in data/raw/")
        return
    
    logging.info(f"Found {len(raw_files)} raw files to process")
    
    for raw_file in raw_files:
        process_raw_file(raw_file)

def main():
    """Main function to process raw data files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process raw Telegram data')
    parser.add_argument('--file', help='Process specific raw JSON file')
    parser.add_argument('--all', action='store_true', help='Process all raw JSON files')
    
    args = parser.parse_args()
    
    if args.file:
        if os.path.exists(args.file):
            process_raw_file(args.file)
        else:
            logging.error(f"File not found: {args.file}")
    elif args.all:
        process_all_raw_files()
    else:
        # Default: process all files
        process_all_raw_files()

if __name__ == "__main__":
    main()