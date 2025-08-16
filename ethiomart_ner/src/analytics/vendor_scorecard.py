import os
import pandas as pd
import numpy as np
from datetime import datetime
from model_training.ner_trainer import FinalNERPredictor

# Load NER model
MODEL_DIR = '../models/distilbert_ner'
predictor = FinalNERPredictor(MODEL_DIR)

def extract_price(entities):
    for ent in entities:
        if ent['label'] == 'PRICE':
            try:
                return float(''.join(filter(str.isdigit, ent['text'])))
            except:
                continue
    return None

def extract_product(entities):
    for ent in entities:
        if ent['label'] == 'PRODUCT':
            return ent['text']
    return None

def process_vendor(csv_path):
    df = pd.read_csv(csv_path)
    # Try to infer timestamp column name
    timestamp_col = None
    for col in df.columns:
        if 'time' in col:
            timestamp_col = col
            break
    if not timestamp_col:
        raise ValueError(f'No timestamp column found in {csv_path}')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['week'] = df[timestamp_col].dt.isocalendar().week

    # NER extraction
    df['entities'] = df['text'].apply(predictor.predict_simple)
    df['price'] = df['entities'].apply(extract_price)
    df['product'] = df['entities'].apply(extract_product)

    total_weeks = df['week'].nunique()
    posts_per_week = len(df) / total_weeks if total_weeks else 0
    avg_views = df['views'].mean() if 'views' in df.columns else 0
    avg_price = df['price'].dropna().mean() if not df['price'].dropna().empty else 0

    # Top post
    if 'views' in df.columns and not df.empty:
        top_idx = df['views'].idxmax()
        top_post = df.loc[top_idx]
        top_product = top_post['product']
        top_price = top_post['price']
        top_views = top_post['views']
    else:
        top_product = None
        top_price = None
        top_views = None

    return {
        'vendor': os.path.basename(csv_path).replace('.csv', ''),
        'avg_views': avg_views,
        'posts_per_week': posts_per_week,
        'avg_price': avg_price,
        'top_product': top_product,
        'top_price': top_price,
        'top_views': top_views
    }

def lending_score(row):
    return (row['avg_views'] * 0.5) + (row['posts_per_week'] * 0.5)

if __name__ == '__main__':
    data_dir = '../data/processed/'
    vendor_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    results = [process_vendor(f) for f in vendor_files]

    scorecard = pd.DataFrame(results)
    scorecard['lending_score'] = scorecard.apply(lending_score, axis=1)

    scorecard_table = scorecard[['vendor', 'avg_views', 'posts_per_week', 'avg_price', 'lending_score']]
    scorecard_table = scorecard_table.sort_values('lending_score', ascending=False)

    print('Vendor Scorecard:')
    print(scorecard_table.to_string(index=False))

    # Save to CSV
    scorecard_table.to_csv('../reports/vendor_scorecard.csv', index=False) 