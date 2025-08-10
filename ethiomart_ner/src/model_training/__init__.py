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
                # Always set label_list
                # Sort by id to ensure correct order
                self.label_list = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        except FileNotFoundError:
            self.label_list = ['O', 'B-PRICE', 'I-PRICE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT']
            self.label_to_id = {label: i for i, label in enumerate(self.label_list)}
            self.id_to_label = {i: label for i, label in enumerate(self.label_list)}