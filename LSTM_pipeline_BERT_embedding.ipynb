# BERT LSTM Classifier Testing Script for Google Colab
# Loads trained models from Google Drive and performs comprehensive testing
# Compatible with T4 GPU and saves results to Google Drive

# Install required packages (if not already installed)
!pip install transformers torch scikit-learn tqdm matplotlib seaborn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import warnings
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import pandas as pd
warnings.filterwarnings('ignore')

# Google Drive paths - UPDATE THESE PATHS TO MATCH YOUR STRUCTURE
DRIVE_BASE_PATH = "/content/drive/MyDrive/ELMo_Project"
MODEL_PATH = "/content/drive/MyDrive/ELMo_Project/model_outputs/best_model.pt"  # or final_model.pt
CONFIG_PATH = "/content/drive/MyDrive/ELMo_Project/model_outputs/config.json"
TEST_DATA_PATH = "/content/sample_data/technical_descriptions_dataset.json"  # Same dataset or new test data
RESULTS_PATH = "/content/drive/MyDrive/ELMo_Project/test_results"

class TextDataset(Dataset):
    """Custom Dataset for loading JSON text classification data"""
    
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32),
            'text': text,
            'original_label': item.get('label_str', 'Unknown')
        }

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, lstm_outputs):
        # lstm_outputs shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_outputs)  # (batch_size, seq_len, 1)
        attention_weights = self.softmax(attention_weights.squeeze(-1))  # (batch_size, seq_len)
        
        # Apply attention weights
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs)  # (batch_size, 1, hidden_dim)
        return weighted_output.squeeze(1), attention_weights

class BertLSTMClassifier(nn.Module):
    """BERT + LSTM + Attention + Classification model"""
    
    def __init__(self, model_name='bert-base-uncased', lstm_hidden_dim=128, 
                 lstm_layers=2, dropout=0.3, num_classes=1, freeze_bert=False):
        super(BertLSTMClassifier, self).__init__()
        
        # BERT embeddings
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert_dim = self.bert.config.hidden_size  # 768 for BERT-base
        
        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(lstm_hidden_dim * 2)  # *2 for bidirectional
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_outputs, _ = self.lstm(embeddings)
        
        # Apply attention
        attended_output, attention_weights = self.attention(lstm_outputs)
        
        # Apply dropout and classify
        output = self.dropout(attended_output)
        logits = self.classifier(output)
        
        return logits, attention_weights

def setup_test_directories():
    """Create necessary directories for test results"""
    directories = [
        RESULTS_PATH,
        f"{RESULTS_PATH}/plots",
        f"{RESULTS_PATH}/predictions",
        f"{RESULTS_PATH}/analysis"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")

def load_trained_model(model_path, config_path):
    """Load the trained model from Google Drive"""
    print(f"Loading model from: {model_path}")
    print(f"Loading config from: {config_path}")
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
    
    # Recreate model architecture
    model = BertLSTMClassifier(
        model_name=config["bert_model"],
        lstm_hidden_dim=config["lstm_hidden_dim"],
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        num_classes=config["num_classes"],
        freeze_bert=config["freeze_bert"]
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model trained on {config.get('total_samples', 'Unknown')} samples")
    print(f"🏆 Best validation accuracy: {config.get('best_val_accuracy', 'Unknown'):.4f}")
    
    return model, tokenizer, config, device

def load_test_data(json_file_path, use_full_dataset=False, test_split=0.2, random_seed=42):
    """Load test data from JSON file"""
    print(f"Loading test data from {json_file_path}...")
    
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Test data file not found: {json_file_path}")
    
    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Load JSON file
    with open(json_file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    desc_count = 0
    meta_count = 0
    
    # Process data
    for item in raw_data:
        text = item['line']
        label_str = item['label']
        
        if label_str == 'DESC':
            label = 1
            desc_count += 1
        else:  # META
            label = 0
            meta_count += 1
        
        data.append({'text': text, 'label': label, 'label_str': label_str})
    
    if use_full_dataset:
        test_data = data
        print(f"Using full dataset for testing: {len(test_data)} samples")
    else:
        # Split to get unseen test data (if using same dataset as training)
        _, test_data = train_test_split(
            data, 
            test_size=test_split, 
            random_state=random_seed, 
            stratify=[item['label'] for item in data]
        )
        print(f"Using {test_split*100}% of dataset for testing: {len(test_data)} samples")
    
    print(f"Test set - DESC: {sum(1 for item in test_data if item['label'] == 1)}")
    print(f"Test set - META: {sum(1 for item in test_data if item['label'] == 0)}")
    
    return test_data

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    texts = [item['text'] for item in batch]
    original_labels = [item['original_label'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': texts,
        'original_labels': original_labels
    }

def test_model(model, test_loader, device):
    """Comprehensive model testing"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_texts = []
    all_original_labels = []
    all_attention_weights = []
    
    print("Running model inference...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits, attention_weights = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(logits.squeeze())
            predictions = probabilities > 0.5
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(batch['texts'])
            all_original_labels.extend(batch['original_labels'])
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    return (np.array(all_predictions), np.array(all_probabilities), 
            np.array(all_labels), all_texts, all_original_labels, all_attention_weights)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix elements
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_positives'] = int(tp)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    
    # Precision, Recall, F1
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

def create_confusion_matrix_plot(y_true, y_pred, class_names=['META', 'DESC'], save_path=None):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()

def create_roc_curve_plot(y_true, y_prob, save_path=None):
    """Create and save ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()

def create_probability_distribution_plot(y_true, y_prob, save_path=None):
    """Create probability distribution plot"""
    desc_probs = y_prob[y_true == 1]
    meta_probs = y_prob[y_true == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(meta_probs, bins=30, alpha=0.7, label='META (True Label)', color='blue', density=True)
    plt.hist(desc_probs, bins=30, alpha=0.7, label='DESC (True Label)', color='red', density=True)
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Probability Distribution by True Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution plot saved to: {save_path}")
    
    plt.show()

def analyze_misclassifications(texts, y_true, y_pred, y_prob, original_labels, save_path=None):
    """Analyze misclassified examples"""
    # Find misclassified indices
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    misclassifications = []
    for idx in misclassified_idx:
        misclassifications.append({
            'text': texts[idx],
            'true_label': 'DESC' if y_true[idx] == 1 else 'META',
            'predicted_label': 'DESC' if y_pred[idx] == 1 else 'META',
            'probability': float(y_prob[idx]),
            'confidence': float(max(y_prob[idx], 1 - y_prob[idx])),
            'original_label': original_labels[idx]
        })
    
    # Sort by confidence (most confident mistakes first)
    misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total misclassifications: {len(misclassifications)}")
    print(f"Misclassification rate: {len(misclassifications)/len(y_true):.3f}")
    
    # Show top 10 most confident mistakes
    print(f"\nTop 10 Most Confident Misclassifications:")
    print("-" * 50)
    for i, misc in enumerate(misclassifications[:10]):
        print(f"\n{i+1}. Confidence: {misc['confidence']:.3f}")
        print(f"   True: {misc['true_label']} | Predicted: {misc['predicted_label']} | Prob: {misc['probability']:.3f}")
        print(f"   Text: {misc['text'][:100]}{'...' if len(misc['text']) > 100 else ''}")
    
    # Save detailed misclassifications to file
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(misclassifications, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed misclassifications saved to: {save_path}")
    
    return misclassifications

def predict_single_text(model, tokenizer, text, config, device):
    """Make prediction on a single text with detailed output"""
    # Tokenize
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config["max_length"],
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(input_ids, attention_mask)
        probability = torch.sigmoid(logits.squeeze()).item()
        prediction = "DESC" if probability > 0.5 else "META"
        confidence = max(probability, 1 - probability)
    
    return {
        'text': text,
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'logit': logits.squeeze().item(),
        'attention_weights': attention_weights.cpu().numpy()
    }

def interactive_testing(model, tokenizer, config, device):
    """Interactive testing function for custom inputs"""
    print(f"\n{'='*60}")
    print("INTERACTIVE TESTING")
    print(f"{'='*60}")
    print("Enter text to classify (type 'quit' to exit):")
    
    test_examples = []
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if not text:
            continue
        
        result = predict_single_text(model, tokenizer, text, config, device)
        
        print(f"\n📝 Text: {text}")
        print(f"🔮 Prediction: {result['prediction']}")
        print(f"📊 Probability: {result['probability']:.4f}")
        print(f"🎯 Confidence: {result['confidence']:.4f}")
        
        test_examples.append(result)
    
    return test_examples

def save_test_results(metrics, predictions_data, config, save_path):
    """Save comprehensive test results"""
    results = {
        'test_date': str(datetime.now()),
        'model_info': {
            'model_path': MODEL_PATH,
            'config_path': CONFIG_PATH,
            'best_val_accuracy': config.get('best_val_accuracy', 'Unknown'),
            'total_parameters': config.get('total_parameters', 'Unknown'),
            'bert_model': config.get('bert_model', 'Unknown')
        },
        'test_metrics': metrics,
        'test_data_info': {
            'total_samples': len(predictions_data['texts']),
            'desc_samples': int(np.sum(predictions_data['y_true'])),
            'meta_samples': int(len(predictions_data['y_true']) - np.sum(predictions_data['y_true']))
        },
        'detailed_predictions': []
    }
    
    # Add detailed predictions
    for i in range(len(predictions_data['texts'])):
        results['detailed_predictions'].append({
            'text': predictions_data['texts'][i],
            'true_label': 'DESC' if predictions_data['y_true'][i] == 1 else 'META',
            'predicted_label': 'DESC' if predictions_data['y_pred'][i] == 1 else 'META',
            'probability': float(predictions_data['y_prob'][i]),
            'correct': bool(predictions_data['y_true'][i] == predictions_data['y_pred'][i])
        })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Complete test results saved to: {save_path}")

def create_comprehensive_report(metrics, config, save_path=None):
    """Create a comprehensive testing report"""
    report = f"""
BERT-LSTM MODEL TESTING REPORT
{'='*60}

Model Information:
- Model Architecture: BERT + Bidirectional LSTM + Attention
- BERT Model: {config.get('bert_model', 'Unknown')}
- LSTM Hidden Dim: {config.get('lstm_hidden_dim', 'Unknown')}
- LSTM Layers: {config.get('lstm_layers', 'Unknown')}
- Total Parameters: {config.get('total_parameters', 'Unknown'):,}
- Training Best Val Accuracy: {config.get('best_val_accuracy', 'Unknown'):.4f}

Test Results:
{'='*30}
- Test Accuracy: {metrics['accuracy']:.4f}
- AUC-ROC Score: {metrics['auc_roc']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- Specificity: {metrics['specificity']:.4f}

Confusion Matrix:
- True Positives (DESC correctly predicted): {metrics['true_positives']}
- True Negatives (META correctly predicted): {metrics['true_negatives']}
- False Positives (META predicted as DESC): {metrics['false_positives']}
- False Negatives (DESC predicted as META): {metrics['false_negatives']}

Performance Summary:
{'='*30}
The model shows {'excellent' if metrics['accuracy'] > 0.9 else 'good' if metrics['accuracy'] > 0.8 else 'moderate'} performance with {metrics['accuracy']:.1%} accuracy.
- Precision of {metrics['precision']:.1%} means {metrics['precision']:.1%} of DESC predictions are correct
- Recall of {metrics['recall']:.1%} means the model finds {metrics['recall']:.1%} of all true DESC samples
- AUC-ROC of {metrics['auc_roc']:.3f} indicates {'excellent' if metrics['auc_roc'] > 0.9 else 'good' if metrics['auc_roc'] > 0.8 else 'moderate'} discriminative ability

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    print(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {save_path}")
    
    return report

def main():
    """Main testing function"""
    print("🚀 BERT-LSTM MODEL TESTING SCRIPT")
    print("="*60)
    
    # Setup directories
    setup_test_directories()
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Load trained model
        print("\n📁 Loading trained model...")
        model, tokenizer, config, device = load_trained_model(MODEL_PATH, CONFIG_PATH)
        
        # Load test data
        print("\n📊 Loading test data...")
        test_data = load_test_data(
            TEST_DATA_PATH, 
            use_full_dataset=False,  # Set to True if you have separate test data
            test_split=0.2,
            random_seed=config.get('random_seed', 42)
        )
        
        # Create test dataset and dataloader
        test_dataset = TextDataset(test_data, tokenizer, max_length=config["max_length"])
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2
        )
        
        # Run testing
        print("\n🧪 Running model testing...")
        y_pred, y_prob, y_true, texts, original_labels, attention_weights = test_model(model, test_loader, device)
        
        # Calculate metrics
        print("\n📈 Calculating metrics...")
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        # Confusion Matrix
        cm_path = f"{RESULTS_PATH}/plots/confusion_matrix.png"
        create_confusion_matrix_plot(y_true, y_pred, save_path=cm_path)
        
        # ROC Curve
        roc_path = f"{RESULTS_PATH}/plots/roc_curve.png"
        create_roc_curve_plot(y_true, y_prob, save_path=roc_path)
        
        # Probability Distribution
        prob_dist_path = f"{RESULTS_PATH}/plots/probability_distribution.png"
        create_probability_distribution_plot(y_true, y_prob, save_path=prob_dist_path)
        
        # Analyze misclassifications
        print("\n🔍 Analyzing misclassifications...")
        misc_path = f"{RESULTS_PATH}/analysis/misclassifications.json"
        misclassifications = analyze_misclassifications(
            texts, y_true, y_pred, y_prob, original_labels, save_path=misc_path
        )
        
        # Generate comprehensive report
        print("\n📋 Generating comprehensive report...")
        report_path = f"{RESULTS_PATH}/test_report.txt"
        report = create_comprehensive_report(metrics, config, save_path=report_path)
        
        # Save all predictions and results
        predictions_data = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_prob': y_prob.tolist(),
            'texts': texts,
            'original_labels': original_labels
        }
        
        results_path = f"{RESULTS_PATH}/complete_test_results.json"
        save_test_results(metrics, predictions_data, config, results_path)
        
        # Print classification report
        print(f"\n📊 DETAILED CLASSIFICATION REPORT:")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=['META', 'DESC']))
        
        # Summary
        print(f"\n🎉 TESTING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 RESULTS SAVED TO GOOGLE DRIVE:")
        print("="*60)
        print(f"📁 {RESULTS_PATH}/")
        print("  ├── 📁 plots/")
        print("  │   ├── 📊 confusion_matrix.png")
        print("  │   ├── 📈 roc_curve.png")
        print("  │   └── 📊 probability_distribution.png")
        print("  ├── 📁 predictions/")
        print("  ├── 📁 analysis/")
        print("  │   └── 🔍 misclassifications.json")
        print("  ├── 📄 test_report.txt")
        print("  └── 📊 complete_test_results.json")
        print("="*60)
        print(f"✅ Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"🎯 AUC-ROC Score: {metrics['auc_roc']:.4f}")
        print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
        
        # Option for interactive testing
        print(f"\n🔧 INTERACTIVE TESTING AVAILABLE")
        print("Run interactive_testing(model, tokenizer, config, device) to test custom texts")
        
        return model, tokenizer, config, device, metrics, predictions_data
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("\nPlease check:")
        print("1. Model file exists at the specified path")
        print("2. Config file exists at the specified path")
        print("3. Test data file exists")
        print("4. Google Drive is properly mounted")
        return None

def test_specific_examples():
    """Test with specific example texts"""
    # Load model first
    try:
        model, tokenizer, config, device = load_trained_model(MODEL_PATH, CONFIG_PATH)
        
        # Example texts for testing
        test_examples = [
            "The system processes user authentication requests through a secure token-based mechanism.",
            "John Smith is the lead developer responsible for the authentication module implementation.",
            "The database stores encrypted user credentials using AES-256 encryption algorithm.",
            "Sarah Johnson worked on the frontend design for the login interface last month.",
            "The API endpoint /auth/login accepts POST requests with username and password parameters.",
            "The development team met yesterday to discuss the new security requirements.",
            "Error handling is implemented using try-catch blocks with custom exception classes.",
            "Mike from the QA team found several bugs in the authentication flow during testing."
        ]
        
        print(f"\n🧪 TESTING SPECIFIC EXAMPLES")
        print("="*60)
        
        results = []
        for i, text in enumerate(test_examples, 1):
            result = predict_single_text(model, tokenizer, text, config, device)
            results.append(result)
            
            print(f"\n{i}. Text: {text}")
            print(f"   🔮 Prediction: {result['prediction']}")
            print(f"   📊 Probability: {result['probability']:.4f}")
            print(f"   🎯 Confidence: {result['confidence']:.4f}")
        
        # Save example results
        example_results_path = f"{RESULTS_PATH}/example_predictions.json"
        with open(example_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nExample predictions saved to: {example_results_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in specific examples testing: {str(e)}")
        return None

def batch_predict_from_file(input_file_path, output_file_path=None):
    """Predict labels for texts from a file"""
    try:
        # Load model
        model, tokenizer, config, device = load_trained_model(MODEL_PATH, CONFIG_PATH)
        
        print(f"📁 Loading texts from: {input_file_path}")
        
        # Load input file (assuming JSON format)
        with open(input_file_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        results = []
        
        print(f"🔮 Making predictions for {len(input_data)} texts...")
        
        for item in tqdm(input_data):
            text = item.get('line', item.get('text', ''))
            if text:
                result = predict_single_text(model, tokenizer, text, config, device)
                result['original_data'] = item
                results.append(result)
        
        # Save results
        if output_file_path is None:
            output_file_path = f"{RESULTS_PATH}/batch_predictions.json"
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Batch predictions saved to: {output_file_path}")
        
        # Print summary
        desc_predictions = sum(1 for r in results if r['prediction'] == 'DESC')
        meta_predictions = len(results) - desc_predictions
        
        print(f"\n📊 BATCH PREDICTION SUMMARY:")
        print(f"Total texts processed: {len(results)}")
        print(f"Predicted as DESC: {desc_predictions}")
        print(f"Predicted as META: {meta_predictions}")
        print(f"Average confidence: {np.mean([r['confidence'] for r in results]):.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        return None

def check_file_paths():
    """Check if all required files exist"""
    print("🔍 CHECKING FILE PATHS")
    print("="*40)
    
    files_to_check = [
        (MODEL_PATH, "Model file"),
        (CONFIG_PATH, "Config file"),
        (TEST_DATA_PATH, "Test data file")
    ]
    
    all_exist = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some files are missing. Please check your paths.")
        print("\nAvailable model files in your drive:")
        model_dir = f"{DRIVE_BASE_PATH}/model_outputs"
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                print(f"   📄 {file}")
        else:
            print(f"   📁 Directory not found: {model_dir}")
    
    return all_exist

# Additional utility functions
def compare_models():
    """Compare different model checkpoints if available"""
    model_dir = f"{DRIVE_BASE_PATH}/model_outputs"
    
    if not os.path.exists(model_dir):
        print("❌ Model directory not found")
        return
    
    # Find all model files
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    
    if len(model_files) < 2:
        print("❌ Need at least 2 model files to compare")
        return
    
    print("🔄 COMPARING MODEL CHECKPOINTS")
    print("="*50)
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', 'Unknown')
            val_acc = checkpoint.get('val_accuracy', 'Unknown')
            print(f"📊 {model_file}: Epoch {epoch}, Val Accuracy: {val_acc:.4f}")
        except Exception as e:
            print(f"❌ Error loading {model_file}: {str(e)}")

def visualize_attention_weights(model, tokenizer, text, config, device, save_path=None):
    """Visualize attention weights for a given text"""
    result = predict_single_text(model, tokenizer, text, config, device)
    attention_weights = result['attention_weights'].squeeze()
    
    # Get tokens
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens[:config["max_length"]-2] + ['[SEP]']
    
    # Pad or truncate attention weights to match tokens
    if len(attention_weights) > len(tokens):
        attention_weights = attention_weights[:len(tokens)]
    elif len(attention_weights) < len(tokens):
        padding = np.zeros(len(tokens) - len(attention_weights))
        attention_weights = np.concatenate([attention_weights, padding])
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tokens)), attention_weights, alpha=0.7)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.ylabel('Attention Weight')
    plt.title(f'Attention Weights\nPrediction: {result["prediction"]} (Prob: {result["probability"]:.3f})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to: {save_path}")
    
    plt.show()
    return attention_weights, tokens

# Usage Instructions
print("\n" + "🚀 TESTING SCRIPT READY!")
print("="*60)
print("USAGE OPTIONS:")
print("="*60)
print("1. 🧪 Full Testing Pipeline:")
print("   Run: main()")
print("\n2. 🔍 Check File Paths:")
print("   Run: check_file_paths()")
print("\n3. 🧩 Test Specific Examples:")
print("   Run: test_specific_examples()")
print("\n4. 🔄 Compare Model Checkpoints:")
print("   Run: compare_models()")
print("\n5. 🎯 Interactive Testing:")
print("   First run main(), then use the returned model:")
print("   model, tokenizer, config, device, metrics, data = main()")
print("   interactive_testing(model, tokenizer, config, device)")
print("\n6. 📊 Batch Prediction:")
print("   Run: batch_predict_from_file('path/to/input.json')")
print("\n7. 👁️ Visualize Attention:")
print("   Run: visualize_attention_weights(model, tokenizer, 'your text', config, device)")
print("="*60)

# Quick start function
def quick_test():
    """Quick test function to run everything"""
    print("🚀 RUNNING QUICK TEST...")
    
    # Check files first
    if not check_file_paths():
        return None
    
    # Run main testing
    results = main()
    
    if results:
        model, tokenizer, config, device, metrics, data = results
        
        # Test specific examples
        print("\n🧩 Testing specific examples...")
        test_specific_examples()
        
        # Compare models if available
        print("\n🔄 Comparing available models...")
        compare_models()
        
        print(f"\n✅ QUICK TEST COMPLETED!")
        print(f"🎯 Final Test Accuracy: {metrics['accuracy']:.4f}")
        
        return results
    
    return None

# Run initial check
print("\n🔍 INITIAL ENVIRONMENT CHECK:")
check_file_paths()

print(f"\n💡 TIP: Run quick_test() for a complete testing pipeline!")
print(f"💡 Or run main() for the standard testing procedure.")

if __name__ == "__main__":
    # Uncomment the line below to run automatically
     main()
  
