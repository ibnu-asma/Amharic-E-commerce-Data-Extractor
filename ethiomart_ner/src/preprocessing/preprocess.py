import re
import unicodedata
from transformers import AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_amharic_text(text):
    """Preprocess Amharic text for NER using XLM-RoBERTa tokenizer."""
    if not text:
        logging.info("Empty text input received")
        return ""
    
    try:
        # Normalize Unicode (NFC for Amharic Fidel script)
        text = unicodedata.normalize('NFC', text)
        logging.debug(f"Normalized text: {text}")
        
        # Remove Amharic-specific punctuation (።, ፣, ፤, etc.)
        amharic_punctuation = r'[።፣፤፥፦፧!]'
        text = re.sub(amharic_punctuation, '', text)
        logging.debug(f"After punctuation removal: {text}")
        
        # Convert Amharic numerals to Arabic numerals (e.g., ፩፻ → 100)
        amharic_to_arabic = {
            '፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5',
            '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፻': '100', '፲': '10'
        }
        text = re.sub(r'፩፻', '100', text)  # Special case for ፩፻
        for amh_num, arabic_num in amharic_to_arabic.items():
            text = text.replace(amh_num, arabic_num)
        logging.debug(f"After numeral conversion: {text}")
        
        # Standardize currency (e.g., "1000 birr" or "1000 ብር" to "1000 ETB")
        text = re.sub(r'(\d+)\s*(birr|ብር)', r'\1 ETB', text, flags=re.IGNORECASE)
        logging.debug(f"After currency standardization: {text}")
        
        # Remove common English noise words before tokenization
        text = re.sub(r'\b(extra|spaces?)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        logging.debug(f"After noise word removal: {text}")
        
        # Use XLM-RoBERTa tokenizer
        tokenizer = AutoTokenizer.from_pretrained('Davlan/xlm-roberta-base-finetuned-amharic')
        tokens = tokenizer.tokenize(text, clean_up_tokenization_spaces=True)
        logging.debug(f"Tokens: {tokens}")
        
        # Convert tokens back to text
        final_tokens = [t.replace('##', '').replace('▁', ' ') for t in tokens]
        cleaned_text = ''.join(final_tokens).strip()
        # Final whitespace normalization
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        logging.info(f"Processed text: {cleaned_text}")
        
        return cleaned_text
    except Exception as e:
        logging.error(f"Error preprocessing text: {text}, Error: {e}")
        # Fallback: basic preprocessing without tokenizer
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[።፣፤፥፦፧!]', '', text)
        text = re.sub(r'፩፻', '100', text)
        amharic_to_arabic = {'፩': '1', '፪': '2', '፫': '3', '፬': '4', '፭': '5', '፮': '6', '፯': '7', '፰': '8', '፱': '9', '፻': '100', '፲': '10'}
        for amh_num, arabic_num in amharic_to_arabic.items():
            text = text.replace(amh_num, arabic_num)
        text = re.sub(r'(\d+)\s*(birr|ብር)', r'\1 ETB', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(extra|spaces?)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text