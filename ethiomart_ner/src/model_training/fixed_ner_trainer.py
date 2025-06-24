import os
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedNERTrainer:
    def __init__(self, model_name="xlm-roberta-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_list = ['O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT']
        self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
        self.id_to_label = {i: label for i, label in enumerate(self.label_list)}
        
    def load_conll_data(self, file_path):
        """Load CoNLL format data."""
        sentences = []
        labels = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_tokens:
                        sentences.append(current_tokens)
                        labels.append(current_labels)
                        current_tokens = []
                        current_labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        current_tokens.append(token)
                        current_labels.append(label)
        
        if current_tokens:
            sentences.append(current_tokens)
            labels.append(current_labels)
            
        logger.info(f"Loaded {len(sentences)} sentences")
        return sentences, labels
    
    def calculate_class_weights(self, labels):
        """Calculate class weights to handle imbalanced data."""
        # Flatten all labels
        all_labels = []
        for sentence_labels in labels:
            all_labels.extend(sentence_labels)
        
        # Get unique labels and their counts
        unique_labels = list(set(all_labels))
        label_counts = {label: all_labels.count(label) for label in unique_labels}
        
        logger.info(f"Label distribution: {label_counts}")
        
        # Calculate weights (inverse frequency)
        total_samples = len(all_labels)
        weights = {}
        for label in self.label_list:
            if label in label_counts:
                # Higher weight for less frequent classes
                weights[self.label_to_id[label]] = total_samples / (len(unique_labels) * label_counts[label])
            else:
                weights[self.label_to_id[label]] = 1.0
        
        # Normalize weights
        max_weight = max(weights.values())
        for key in weights:
            weights[key] = weights[key] / max_weight
        
        logger.info(f"Class weights: {weights}")
        return weights
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with subword tokens."""
        tokenized_inputs = self.tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            padding=True,
            max_length=128
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            for pred_id, label_id in zip(prediction, label):
                if label_id != -100:
                    true_predictions.append(self.id_to_label[pred_id])
                    true_labels.append(self.id_to_label[label_id])
        
        # Calculate metrics with zero_division handling
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, true_predictions, average='weighted', zero_division=0
        )
        accuracy = accuracy_score(true_labels, true_predictions)
        
        # Calculate per-entity metrics
        entity_f1 = {}
        for entity_type in ['PRICE', 'LOC', 'PRODUCT']:
            entity_true = [1 if label.endswith(entity_type) else 0 for label in true_labels]
            entity_pred = [1 if label.endswith(entity_type) else 0 for label in true_predictions]
            if sum(entity_true) > 0:
                _, _, entity_f1[entity_type], _ = precision_recall_fscore_support(
                    entity_true, entity_pred, average='binary', zero_division=0
                )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'entity_f1': entity_f1
        }
    
    def create_weighted_trainer(self, model, train_dataset, val_dataset, class_weights, output_dir, epochs, batch_size, learning_rate):
        """Create trainer with class weights."""
        
        class WeightedTrainer(Trainer):
            def __init__(self, class_weights, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.class_weights = torch.tensor(list(class_weights.values()), dtype=torch.float)
                if torch.cuda.is_available():
                    self.class_weights = self.class_weights.cuda()
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                # Flatten for loss calculation
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=200,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            report_to=None,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def train(self, conll_file, output_dir="./fixed_ner_model", epochs=10, batch_size=8, learning_rate=3e-5):
        """Train NER model with class balancing."""
        logger.info("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.label_list)
        )
        
        logger.info("Loading and preparing data...")
        sentences, labels = self.load_conll_data(conll_file)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(labels)
        
        # Split data
        split_idx = int(len(sentences) * 0.8)
        train_sentences = sentences[:split_idx]
        train_labels = labels[:split_idx]
        val_sentences = sentences[split_idx:]
        val_labels = labels[split_idx:]
        
        logger.info(f"Training set: {len(train_sentences)} sentences")
        logger.info(f"Validation set: {len(val_sentences)} sentences")
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            "tokens": train_sentences,
            "ner_tags": train_labels
        })
        
        val_dataset = Dataset.from_dict({
            "tokens": val_sentences,
            "ner_tags": val_labels
        })
        
        # Apply tokenization
        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        val_dataset = val_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        # Create weighted trainer
        trainer = self.create_weighted_trainer(
            self.model, train_dataset, val_dataset, class_weights,
            output_dir, epochs, batch_size, learning_rate
        )
        
        logger.info("Starting training with class balancing...")
        trainer.train()
        
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label,
                'class_weights': class_weights
            }, f, indent=2)
        
        logger.info("Training completed!")
        return trainer
    
    def predict(self, text, model_path="./fixed_ner_model"):
        """Predict entities with the trained model."""
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not self.model:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        tokens = text.split()
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        predictions = torch.argmax(outputs.logits, dim=2)
        word_ids = inputs.word_ids()
        
        entities = []
        current_entity = []
        current_label = None
        current_start = None
        
        for i, (word_id, pred_id) in enumerate(zip(word_ids, predictions[0])):
            if word_id is not None and word_id < len(tokens):
                label = self.id_to_label[pred_id.item()]
                
                if label.startswith('B-'):
                    if current_entity and current_label:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': current_start,
                            'end': current_start + len(current_entity) - 1
                        })
                    current_entity = [tokens[word_id]]
                    current_label = label[2:]
                    current_start = word_id
                    
                elif label.startswith('I-') and current_entity and label[2:] == current_label:
                    if word_id == current_start + len(current_entity):
                        current_entity.append(tokens[word_id])
                        
                else:
                    if current_entity and current_label:
                        entities.append({
                            'text': ' '.join(current_entity),
                            'label': current_label,
                            'start': current_start,
                            'end': current_start + len(current_entity) - 1
                        })
                    current_entity = []
                    current_label = None
                    current_start = None
        
        if current_entity and current_label:
            entities.append({
                'text': ' '.join(current_entity),
                'label': current_label,
                'start': current_start,
                'end': current_start + len(current_entity) - 1
            })
        
        return entities

def main():
    """Main training function with class balancing."""
    trainer = FixedNERTrainer("xlm-roberta-base")
    
    # Train with class balancing
    trainer.train(
        conll_file="data/labeled/conll_labeled.txt",
        output_dir="models/fixed_ner_model",
        epochs=10,
        batch_size=8,
        learning_rate=3e-5
    )
    
    # Test prediction
    test_texts = [
        "ህጻን ጠርሙስ ዋጋ 2000 ETB በቦሌ",
        "አዲስ ጫማ 1500 ብር በመርካቶ"
    ]
    
    print("\n🧪 Testing fixed model:")
    for text in test_texts:
        entities = trainer.predict(text, "models/fixed_ner_model")
        print(f"\nText: {text}")
        print("Entities:")
        for entity in entities:
            print(f"  - {entity['text']} [{entity['label']}]")

if __name__ == "__main__":
    main()