import pandas as pd
import glob
import os
from collections import Counter
import re

def generate_data_statistics():
    """Generate comprehensive statistics from all processed data."""
    
    # Load all processed CSV files
    csv_files = glob.glob('data/processed/*.csv')
    if not csv_files:
        print("No processed CSV files found.")
        return
    
    all_data = []
    for file in csv_files:
        df = pd.read_csv(file, encoding='utf-8')
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Basic statistics
    total_messages = len(combined_df)
    unique_channels = combined_df['channel'].nunique()
    messages_per_channel = combined_df['channel'].value_counts()
    
    # Text analysis
    text_messages = combined_df['text'].notna().sum()
    avg_text_length = combined_df['text'].str.len().mean()
    
    # Media analysis
    image_messages = combined_df['image_path'].notna().sum()
    video_messages = combined_df['doc_path'].str.contains('.mp4', na=False).sum()
    
    # Price analysis
    price_pattern = r'(\d+[,\d]*)\s*(ETB|ብր|birr)'
    price_messages = combined_df['processed_text'].str.contains(price_pattern, case=False, na=False).sum()
    
    # Extract all prices for analysis
    all_prices = []
    for text in combined_df['processed_text'].dropna():
        prices = re.findall(r'(\d+[,\d]*)\s*ETB', text, re.IGNORECASE)
        for price in prices:
            try:
                price_num = int(price.replace(',', ''))
                all_prices.append(price_num)
            except:
                continue
    
    # Phone number analysis
    phone_pattern = r'\+251\d{9}'
    phone_messages = combined_df['text'].str.contains(phone_pattern, na=False).sum()
    
    # Generate report
    stats = {
        'Total Messages': total_messages,
        'Unique Channels': unique_channels,
        'Messages with Text': f"{text_messages} ({text_messages/total_messages*100:.1f}%)",
        'Messages with Images': f"{image_messages} ({image_messages/total_messages*100:.1f}%)",
        'Messages with Videos': f"{video_messages} ({video_messages/total_messages*100:.1f}%)",
        'Messages with Prices': f"{price_messages} ({price_messages/total_messages*100:.1f}%)",
        'Messages with Phone Numbers': f"{phone_messages} ({phone_messages/total_messages*100:.1f}%)",
        'Average Text Length': f"{avg_text_length:.1f} characters",
        'Price Range': f"{min(all_prices) if all_prices else 0} - {max(all_prices) if all_prices else 0} ETB",
        'Average Price': f"{sum(all_prices)/len(all_prices) if all_prices else 0:.0f} ETB"
    }
    
    print("=== DATA STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== MESSAGES PER CHANNEL ===")
    for channel, count in messages_per_channel.items():
        print(f"{channel}: {count}")
    
    return stats

if __name__ == "__main__":
    generate_data_statistics()