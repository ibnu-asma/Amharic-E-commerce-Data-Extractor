import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalNERPredictor:
    def __init__(self, model_path="models/improved_ner_model"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Load label mappings
        try:
            with open(f"{model_path}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.label_to_id = mappings['label_to_id']
                self.id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
                # Always set label_list in the correct order
                self.label_list = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        except FileNotFoundError:
            self.label_list = ['O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT']
            self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
            self.id_to_label = {i: label for i, label in enumerate(self.label_list)}
    
    def predict_with_confidence(self, text, confidence_threshold=0.5):
        """Predict entities with confidence scoring."""
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities instead of just argmax
        probabilities = torch.softmax(outputs.logits, dim=2)
        predictions = torch.argmax(probabilities, dim=2)
        max_probs = torch.max(probabilities, dim=2)[0]
        
        word_ids = inputs.word_ids()
        
        # Extract entities with confidence filtering
        entities = []
        current_entity = []
        current_label = None
        current_start = None
        current_confidences = []
        
        for i, (word_id, pred_id, confidence) in enumerate(zip(word_ids, predictions[0], max_probs[0])):
            if word_id is not None and word_id < len(tokens):
                label = self.id_to_label[pred_id.item()]
                conf_score = confidence.item()
                
                # Only consider predictions above confidence threshold
                if conf_score < confidence_threshold and label != 'O':
                    label = 'O'  # Convert low-confidence predictions to O
                
                if label.startswith('B-'):
                    # Save previous entity if confidence is good
                    if current_entity and current_label and np.mean(current_confidences) >= confidence_threshold:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': current_start,
                            'end': current_start + len(current_entity) - 1,
                            'confidence': np.mean(current_confidences)
                        })
                    
                    # Start new entity
                    current_entity = [tokens[word_id]]
                    current_label = label[2:]
                    current_start = word_id
                    current_confidences = [conf_score]
                    
                elif label.startswith('I-') and current_entity and label[2:] == current_label:
                    # Continue current entity
                    if word_id == current_start + len(current_entity):
                        current_entity.append(tokens[word_id])
                        current_confidences.append(conf_score)
                    
                elif label == 'O':
                    # End current entity if confidence is good
                    if current_entity and current_label and np.mean(current_confidences) >= confidence_threshold:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': current_start,
                            'end': current_start + len(current_entity) - 1,
                            'confidence': np.mean(current_confidences)
                        })
                    current_entity = []
                    current_label = None
                    current_start = None
                    current_confidences = []
        
        # Handle entity at end
        if current_entity and current_label and np.mean(current_confidences) >= confidence_threshold:
            entities.append({
                'text': ' '.join(current_entity),
                'label': current_label,
                'start': current_start,
                'end': current_start + len(current_entity) - 1,
                'confidence': np.mean(current_confidences)
            })
        
        return entities
    
    def predict_simple(self, text):
        """Simple prediction with lower threshold for better recall."""
        return self.predict_with_confidence(text, confidence_threshold=0.3)
    
    def batch_predict(self, texts, confidence_threshold=0.4):
        """Predict entities for multiple texts."""
        results = []
        for text in texts:
            entities = self.predict_with_confidence(text, confidence_threshold)
            results.append({
                'text': text,
                'entities': entities
            })
        return results

    def predict_token_probs(self, text):
        """
        Returns per-token class probabilities for a given text.
        Output: probs [num_tokens, num_classes], tokens [list of tokens]
        """
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=2)[0]  # [seq_len, num_classes]
        word_ids = inputs.word_ids()
        # Only keep probabilities for real tokens (not special tokens)
        token_probs = []
        real_tokens = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(tokens):
                token_probs.append(probabilities[i].cpu().numpy())
                real_tokens.append(tokens[word_id])
        return np.array(token_probs), real_tokens

def test_final_model():
    """Test the final model with various confidence thresholds."""
    predictor = FinalNERPredictor("models/improved_ner_model")
    
    test_texts = [
        "ህጻን ጠርሙስ ዋጋ 2000 ETB በቦሌ",
        "አዲስ ጫማ 1500 ብር በመርካቶ", 
        "Samsung ስልክ 25000 ETB በአዲስ አበባ",
        "የቤት እቃዎች 5000 ብር በፒያሳ",
        "ሴቶች ቦርሳ 800 ETB በካዛንቺስ"
    ]
    
    print("🧪 FINAL MODEL TESTING WITH DIFFERENT CONFIDENCE LEVELS")
    print("=" * 60)
    
    for threshold in [0.3, 0.5, 0.7]:
        print(f"\n📊 Confidence Threshold: {threshold}")
        print("-" * 40)
        
        for text in test_texts:
            entities = predictor.predict_with_confidence(text, threshold)
            print(f"\nText: {text}")
            if entities:
                print("Entities:")
                for entity in entities:
                    print(f"  • '{entity['text']}' [{entity['label']}] (conf: {entity['confidence']:.3f})")
            else:
                print("  • No entities found")
    
    print("\n🎯 RECOMMENDED: Use confidence threshold 0.3-0.4 for best results")

if __name__ == "__main__":
    test_final_model()