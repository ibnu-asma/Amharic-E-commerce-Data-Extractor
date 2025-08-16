import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NERModelEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Load label mappings
        try:
            with open(f"{model_path}/label_mappings.json", 'r') as f:
                mappings = json.load(f)
                self.label_to_id = mappings['label_to_id']
                self.id_to_label = mappings['id_to_label']
                # Convert string keys back to int for id_to_label
                self.id_to_label = {int(k): v for k, v in self.id_to_label.items()}
        except FileNotFoundError:
            # Default mappings
            self.label_list = ['O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT']
            self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
            self.id_to_label = {i: label for i, label in enumerate(self.label_list)}
    
    def predict_entities(self, text):
        """Predict entities in text."""
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        word_ids = inputs.word_ids()
        
        entities = []
        current_entity = []
        current_label = None
        
        for i, (word_id, pred_id) in enumerate(zip(word_ids, predictions[0])):
            if word_id is not None:
                label = self.id_to_label[pred_id.item()]
                
                if label.startswith('B-'):
                    if current_entity:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': word_id - len(current_entity) + 1,
                            'end': word_id
                        })
                    current_entity = [tokens[word_id]]
                    current_label = label[2:]
                elif label.startswith('I-') and current_entity:
                    current_entity.append(tokens[word_id])
                else:
                    if current_entity:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': word_id - len(current_entity),
                            'end': word_id - 1
                        })
                    current_entity = []
                    current_label = None
        
        if current_entity:
            entities.append({
                'text': ' '.join(current_entity),
                'label': current_label,
                'start': len(tokens) - len(current_entity),
                'end': len(tokens) - 1
            })
        
        return entities
    
    def evaluate_on_test_set(self, test_texts, true_entities_list):
        """Evaluate model on a test set."""
        all_predictions = []
        all_true_labels = []
        
        for text, true_entities in zip(test_texts, true_entities_list):
            predicted_entities = self.predict_entities(text)
            
            # Convert to label sequence for evaluation
            tokens = text.split()
            pred_labels = ['O'] * len(tokens)
            true_labels = ['O'] * len(tokens)
            
            # Fill predicted labels
            for entity in predicted_entities:
                start, end = entity['start'], entity['end']
                label_type = entity['label']
                if start < len(tokens):
                    pred_labels[start] = f'B-{label_type}'
                    for i in range(start + 1, min(end + 1, len(tokens))):
                        pred_labels[i] = f'I-{label_type}'
            
            # Fill true labels
            for entity in true_entities:
                start, end = entity['start'], entity['end']
                label_type = entity['label']
                if start < len(tokens):
                    true_labels[start] = f'B-{label_type}'
                    for i in range(start + 1, min(end + 1, len(tokens))):
                        true_labels[i] = f'I-{label_type}'
            
            all_predictions.extend(pred_labels)
            all_true_labels.extend(true_labels)
        
        return all_true_labels, all_predictions
    
    def generate_evaluation_report(self, true_labels, predicted_labels):
        """Generate comprehensive evaluation report."""
        # Classification report
        report = classification_report(
            true_labels, predicted_labels, 
            output_dict=True, zero_division=0
        )
        
        # Entity-level metrics
        entity_metrics = {}
        for label in ['PRICE', 'LOC', 'PRODUCT']:
            b_label = f'B-{label}'
            i_label = f'I-{label}'
            
            if b_label in report and i_label in report:
                # Combine B- and I- metrics
                precision = (report[b_label]['precision'] + report[i_label]['precision']) / 2
                recall = (report[b_label]['recall'] + report[i_label]['recall']) / 2
                f1 = (report[b_label]['f1-score'] + report[i_label]['f1-score']) / 2
                
                entity_metrics[label] = {
                    'precision': precision,
                    'recall': recall,
                    'f1-score': f1
                }
        
        return report, entity_metrics
    
    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """Plot confusion matrix."""
        labels = list(set(true_labels + predicted_labels))
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix - NER Model')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def test_sample_texts(self):
        """Test model on sample Amharic e-commerce texts."""
        sample_texts = [
            "ህጻን ጠርሙስ ዋጋ 2000 ETB በቦሌ አካባቢ",
            "አዲስ ጫማ 1500 ብር በመርካቶ ገበያ",
            "Samsung ስልክ 25000 ETB በአዲስ አበባ",
            "የቤት እቃዎች 5000 ብር በፒያሳ",
            "ሴቶች ቦርሳ 800 ETB በካዛንቺስ",
            "ልጆች አሻንጉሊት 300 ብር በሜክሲኮ"
        ]
        
        print("🧪 Testing NER Model on Sample Texts")
        print("=" * 50)
        
        for i, text in enumerate(sample_texts, 1):
            entities = self.predict_entities(text)
            print(f"\n{i}. Text: {text}")
            print("   Entities:")
            
            if entities:
                for entity in entities:
                    print(f"      • {entity['text']} [{entity['label']}]")
            else:
                print("      • No entities found")
        
        return sample_texts

def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained NER model')
    parser.add_argument('--model', default='models/ner_model',
                       help='Path to trained model')
    parser.add_argument('--test', action='store_true',
                       help='Run tests on sample texts')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        logger.error("Please train the model first using Task 3")
        return
    
    # Initialize evaluator
    evaluator = NERModelEvaluator(args.model)
    
    if args.test:
        evaluator.test_sample_texts()
    
    print(f"\n✅ Model evaluation completed!")
    print(f"Model path: {args.model}")

if __name__ == "__main__":
    main()