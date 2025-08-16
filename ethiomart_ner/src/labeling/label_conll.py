import os
import pandas as pd
import re
from transformers import AutoTokenizer
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_messages(input_dir='data/processed', num_messages=30):
    """Load messages from processed CSV files, filtering out nan values."""
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found in {input_dir}")
        return []
    
    df = pd.concat([pd.read_csv(os.path.join(input_dir, f), encoding='utf-8') for f in csv_files])
    # Filter messages with non-null text
    df_valid = df[df['text'].notna() & df['text'].str.strip().astype(bool)]
    if len(df_valid) < num_messages:
        logging.warning(f"Only {len(df_valid)} valid messages available. Using all.")
        messages = df_valid['text'].tolist()
    else:
        messages = df_valid['text'].sample(num_messages, random_state=42).tolist()
    logging.info(f"Loaded {len(messages)} valid messages for labeling")
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
    """Assign CoNLL labels to tokens with improved multi-word entity detection."""
    labels = []
    i = 0
    while i < len(tokens):
        token = tokens[i].lower()  # Case-insensitive matching
        
        # Price: Matches "X,YYY ETB" or "X,YYY ብር" with multi-token handling
        if re.match(r'^\d+(,\d+)?$', token):
            next_token = tokens[i + 1] if i + 1 < len(tokens) else ''
            if next_token in ['birr', 'ብር', 'etb']:
                labels.append('B-PRICE')
                labels.append('I-PRICE')
                i += 2
                continue
            elif i > 0 and tokens[i - 1] in ['ዋጋ', 'price', '💰']:
                labels.append('B-PRICE')
                i += 1
                continue
        
        # Location: Matches known multi-word locations
        if token in ['ቦሌ'] and i + 1 < len(tokens) and tokens[i + 1] == 'መድሐኔዓለም':
            labels.append('B-LOC')
            labels.append('I-LOC')
            i += 2
            continue
        elif token in ['megenga', 'ሜክሲኮ']:
            labels.append('B-LOC')
            i += 1
            continue
        
        # Product: Heuristic for multi-word products (before ዋጋ, price, or after emojis)
        product_keywords = ['shoe', 'ጫማ', 'high', 'pressure', 'water', 'gun', 'turmeric', 'whitening', 'oil', 'la', 'roche', 'eucerine', 'baby', 'feeding', 'set', 'sponge', 'clay']
        if (token in product_keywords or 
            (i > 0 and tokens[i - 1] in ['❇️', '🌟', '🔰', '👉']) or 
            (i + 1 < len(tokens) and tokens[i + 1] in ['ዋጋ', 'price'])):
            if labels and labels[-1] in ['B-Product', 'I-Product']:
                labels.append('I-Product')
            else:
                labels.append('B-Product')
            i += 1
            continue
        
        # Default: Outside any entity
        labels.append('O')
        i += 1
    
    return labels

def save_conll(messages, output_path='data/labeled/conll_labeled.txt', num_messages=30):
    """Save labeled messages in CoNLL format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained('Davlan/xlm-roberta-base-finetuned-amharic')
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}. Using whitespace splitting.")
        tokenizer = None
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, message in enumerate(messages[:num_messages]):
            if not message or not isinstance(message, str):
                logging.warning(f"Skipping invalid message {idx + 1}")
                continue
            tokens = tokenize_message(message, tokenizer) if tokenizer else message.split()
            labels = label_tokens(tokens)
            
            # Write tokens and labels, ensuring alignment
            for token, label in zip(tokens, labels[:len(tokens)]):
                f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line between messages
            logging.info(f"Labeled message {idx + 1}/{num_messages}")
    
    logging.info(f"Saved CoNLL data to {output_path}")

if __name__ == "__main__":
    messages = load_messages()
    if messages:
        save_conll(messages)
    else:
        logging.error("No messages loaded. Run scraper.py and process_data.py first.")