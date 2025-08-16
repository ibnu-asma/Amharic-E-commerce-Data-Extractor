import re
import unicodedata

def preprocess_amharic_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'(\d+)\s*(birr|ብር)', r' ETB', text, flags=re.IGNORECASE)
    return text