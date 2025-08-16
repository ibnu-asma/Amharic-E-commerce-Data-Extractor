import os
import pandas as pd
import re
from transformers import AutoTokenizer
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_messages(input_dir='data/processed', num_messages=30):
    """Load messages from processed CSV files."""
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.error(f"No CSV files found in {input_dir}")
        return []
    
    df = pd.concat([pd.read_csv(os.path.join(input_dir, f), encoding='utf-8') for f in csv_files])
    # Filter messages with Amharic text
    amharic_pattern = r'[\u1200-\u137F]'
    df_amharic = df[df['text'].str.contains(amharic_pattern, na=False)]
    # Sample messages
    if len(df_amharic) < num_messages:
        logging.warning(f"Only {len(df_amharic)} Amharic messages available. Using all.")
        messages = df_amharic['text'].tolist()
    else:
        messages = df_amharic['text'].sample(num_messages, random_state=42).tolist()
    logging.info(f"Loaded {len(messages)} messages for labeling")
    return messages

def tokenize_message(text, tokenizer):
    """Tokenize text using XLM-RoBERTa tokenizer."""
    try:
        if not isinstance(text, str):
            logging.warning(f"Non-string input: {text}. Returning empty tokens.")
            return []
        tokens = tokenizer.tokenize(text)
        # Clean subword prefixes and underscores
        cleaned_tokens = [t.replace('##', '').replace('▁', '') for t in tokens]
        return cleaned_tokens
    except Exception as e:
        logging.error(f"Tokenization failed for text: {text}, Error: {e}")
        # Fallback: split on whitespace
        return text.split()

def label_tokens(tokens):
    """Assign CoNLL labels to tokens using rule-based and heuristic methods."""
    labels = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Price: Matches "1000 ETB" or "ዋጋ 1000"
        if re.match(r'^\d+(,\d+)?$', token) and i + 1 < len(tokens) and tokens[i + 1] in ['ETB', 'ብር']:
            labels.append('B-PRICE')
            labels.append('I-PRICE')
            i += 2
            continue
        elif token in ['ዋጋ', 'በ'] and i + 1 < len(tokens) and re.match(r'^\d+(,\d+)?$', tokens[i + 1]):
            labels.append('O')
            i += 1
            continue
        
        # Location: Matches known locations from EDA (e.g., ቦሌ, አዲስ አበባ)
        if token in ['ቦሌ', 'መድሐኔዓለም', 'አዲስ'] and i + 1 < len(tokens) and tokens[i + 1] == 'አበባ':
            labels.append('B-LOC')
            labels.append('I-LOC')
            i += 2
            continue
        elif token in ['ቦሌ', 'መድሐኔዓለም']:
            labels.append('B-LOC')
            i += 1
            continue
        
        # Product: Heuristic for product names (e.g., before ዋጋ, or specific keywords)
        if token in ['Baby', 'bottle', 'feeding', 'bank', 'ማሽን', 'ጫማ', 'ቦርሳ', 'ቲሸርት'] or (i > 0 and tokens[i - 1] == '❇️'):
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
            
            # Write tokens and labels
            for token, label in zip(tokens, labels):
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