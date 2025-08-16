import pandas as pd
import re
import logging

def validate_processed_data(csv_path):
    """Validate processed CSV data quality."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        total_messages = len(df)
        messages_with_text = df['text'].notna().sum()
        messages_with_media = df[['image_path', 'doc_path']].notna().any(axis=1).sum()
        
        price_pattern = r'\d+[,\d]*\s*(ETB|ብር|birr)'
        messages_with_prices = df['processed_text'].str.contains(price_pattern, case=False, na=False).sum()
        
        phone_pattern = r'\+251\d{9}'
        messages_with_phones = df['text'].str.contains(phone_pattern, na=False).sum()
        
        report = {
            'total_messages': total_messages,
            'messages_with_text': messages_with_text,
            'messages_with_media': messages_with_media,
            'messages_with_prices': messages_with_prices,
            'messages_with_phones': messages_with_phones,
            'text_coverage': f"{(messages_with_text/total_messages)*100:.1f}%",
            'media_coverage': f"{(messages_with_media/total_messages)*100:.1f}%"
        }
        
        print(f"Data validation report: {report}")
        return report
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return None

def extract_entities(text):
    """Extract basic entities from text."""
    if not isinstance(text, str):
        return {}
    
    prices = re.findall(r'(\d+[,\d]*)\s*(ETB|ብር|birr)', text, re.IGNORECASE)
    phones = re.findall(r'\+251\d{9}', text)
    locations = re.findall(r'(አዲስ አበባ|ቦሌ|መክሲኮ|መድሐኔዓለም)', text)
    
    return {
        'prices': [f"{p[0]} {p[1]}" for p in prices],
        'phones': phones,
        'locations': locations
    }