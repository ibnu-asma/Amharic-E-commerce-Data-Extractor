import os
import pandas as pd
import re
from transformers import AutoTokenizer
import logging
import random
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocessing.preprocess import preprocess_for_conll

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_messages(input_dir='data/processed', num_messages=50):
    """Load and preprocess messages for CoNLL labeling."""
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found in {input_dir}")
        return []
    
    df = pd.concat([pd.read_csv(os.path.join(input_dir, f), encoding='utf-8') for f in csv_files])
    # Use preprocessed text if available, otherwise preprocess raw text
    if 'conll_text' in df.columns:
        df_valid = df[df['conll_text'].notna() & df['conll_text'].str.strip().astype(bool)]
        messages = df_valid['conll_text'].tolist()
    else:
        df_valid = df[df['text'].notna() & df['text'].str.strip().astype(bool)]
        messages = [preprocess_for_conll(text) for text in df_valid['text'].tolist()]
    
    # Sample messages for labeling
    if len(messages) > num_messages:
        messages = random.sample(messages, num_messages)
    
    logging.info(f"Loaded {len(messages)} preprocessed messages for labeling")
    return messages

def tokenize_message(text, tokenizer):
    """Tokenize text using XLM-RoBERTa tokenizer with proper space handling."""
    try:
        if not isinstance(text, str) or not text.strip():
            logging.warning(f"Invalid input: {text}. Returning empty tokens.")
            return []
        tokens = tokenizer.tokenize(text)
        # Join subword tokens and clean
        cleaned_tokens = []
        for t in tokens:
            if t.startswith('##'):
                if cleaned_tokens:
                    cleaned_tokens[-1] += t.replace('##', '')
            else:
                cleaned_tokens.append(t.replace('▁', ' ').strip())
        return [t for t in cleaned_tokens if t]
    except Exception as e:
        logging.error(f"Tokenization failed for text: {text}, Error: {e}")
        return text.split()

def label_tokens(tokens):
    """Enhanced CoNLL labeling with better entity recognition."""
    labels = []
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        token_lower = token.lower()
        
        # PRICE entities - numbers followed by currency
        if re.match(r'^\d+([,.]\d+)*$', token):
            next_token = tokens[i + 1].lower() if i + 1 < len(tokens) else ''
            if next_token in ['etb', 'ብር', 'birr']:
                labels.extend(['B-PRICE', 'I-PRICE'])
                i += 2
                continue
            elif re.search(r'(ዋጋ|price)', ' '.join(tokens[max(0, i-2):i])):
                labels.append('B-PRICE')
                i += 1
                continue
        
        # LOCATION entities
        location_patterns = [
            r'(አዲስ\s*አበባ|addis\s*ababa)',
            r'(ቦሌ|bole)', r'(መርካቶ|mercato)', r'(ፒያሳ|piassa)',
            r'(ካዛንቺስ|kazanchis)', r'(ሜክሲኮ|mexico)', r'(ሰሚት|summit)'
        ]
        
        for pattern in location_patterns:
            if re.search(pattern, token_lower, re.IGNORECASE):
                labels.append('B-LOC')
                i += 1
                break
        else:
            # PRODUCT entities - common e-commerce products
            product_keywords = [
                'ጫማ', 'ልብስ', 'ሻንጣ', 'ሰዓት', 'ስልክ', 'ኮምፒውተር',
                'ጠርሙስ', 'አሻንጉሊት', 'ህጻን', 'ኩሽና', 'መጽሐፍ'
            ]
            
            if any(keyword in token_lower for keyword in product_keywords):
                labels.append('B-PRODUCT')
                i += 1
            else:
                labels.append('O')
                i += 1
    
    return labels

def save_conll(messages, output_path='data/labeled/conll_labeled.txt'):
    """Save messages in CoNLL format with BIO tagging."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labeled_count = 0
    entity_stats = {'PRICE': 0, 'LOC': 0, 'PRODUCT': 0}
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, message in enumerate(messages):
            if not message or not isinstance(message, str):
                continue
                
            # Simple tokenization for Amharic text
            tokens = message.split()
            if not tokens:
                continue
                
            labels = label_tokens(tokens)
            
            # Ensure token-label alignment
            min_len = min(len(tokens), len(labels))
            
            for token, label in zip(tokens[:min_len], labels[:min_len]):
                f.write(f"{token}\t{label}\n")
                if label.startswith('B-'):
                    entity_type = label.split('-')[1]
                    entity_stats[entity_type] = entity_stats.get(entity_type, 0) + 1
            
            f.write("\n")  # Sentence separator
            labeled_count += 1
    
    logging.info(f"CoNLL labeling complete: {labeled_count} messages")
    logging.info(f"Entity statistics: {entity_stats}")
    logging.info(f"Saved to {output_path}")

def main():
    """Main function for CoNLL labeling."""
    print("=== TASK 2: CoNLL FORMAT LABELING ===")
    
    messages = load_messages()
    if not messages:
        logging.error("No messages loaded. Run data processing first.")
        return
    
    print(f"Processing {len(messages)} messages for CoNLL labeling...")
    save_conll(messages)
    print("✅ CoNLL labeling completed!")
    
    # Validate output
    output_path = 'data/labeled/conll_labeled.txt'
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✅ Generated {len([l for l in lines if l.strip()])} labeled tokens")
    
if __name__ == "__main__":
    main()