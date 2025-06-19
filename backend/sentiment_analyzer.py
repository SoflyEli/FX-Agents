import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data immediately
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)  # Additional data for wordnet
except Exception as e:
    print(f"Warning: Could not download some NLTK data: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing for both traditional ML and transformer models"""
    
    def __init__(self):
        # Download required NLTK data with better error handling
        nltk_downloads = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet')
        ]
        
        for name, path in nltk_downloads:
            try:
                nltk.data.find(path)
                logger.info(f"NLTK {name} already available")
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK {name}...")
                    nltk.download(name, quiet=True)
                    logger.info(f"Successfully downloaded NLTK {name}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK {name}: {e}")
        
        # Initialize with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not load stopwords, using fallback: {e}")
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Could not initialize lemmatizer: {e}")
            self.lemmatizer = None
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_for_traditional_ml(self, text: str) -> str:
        """Preprocessing for TF-IDF + Naive Bayes"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize - with fallback if NLTK punkt is not available
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenizer failed, using fallback: {e}")
            # Simple fallback tokenization
            tokens = text.split()
        
        # Remove stopwords and lemmatize - with fallbacks
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except Exception:
                        pass  # Use original token if lemmatization fails
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def preprocess_for_transformer(self, text: str) -> str:
        """Light preprocessing for transformer models"""
        # Keep more structure for transformers
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text

class ForexSentimentAnalyzer:
    """Main class for training and serving sentiment analysis models"""
    
    def __init__(self, model_save_path: str = "/app/backend/models"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(exist_ok=True)
        
        self.preprocessor = TextPreprocessor()
        self.label_mapping = {'bullish': 0, 'neutral': 1, 'bearish': 2}
        self.reverse_label_mapping = {0: 'bullish', 1: 'neutral', 2: 'bearish'}
        
        # Traditional ML components
        self.tfidf_pipeline = None
        
        # Transformer components
        self.finbert_tokenizer = None
        self.finbert_model = None
        self.finbert_pipeline = None
        
        # Evaluation results
        self.evaluation_results = {}
    
    def load_and_preprocess_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess the forex news dataset"""
        logger.info(f"Loading data from {csv_path}")
        
        # Create sample data if file doesn't exist (for demo purposes)
        if not Path(csv_path).exists():
            logger.warning(f"Dataset not found at {csv_path}. Creating sample data for demonstration.")
            sample_data = {
                'headline': [
                    "EUR/USD surges to new monthly highs as ECB signals rate hikes",
                    "GBP/JPY remains flat amid mixed economic signals",
                    "USD/CHF plunges on weak US employment data",
                    "Gold prices rally as inflation concerns mount",
                    "Oil futures stable despite geopolitical tensions",
                    "Bitcoin crashes below key support level",
                    "Fed maintains dovish stance, dollar weakens",
                    "European markets mixed in quiet trading session",
                    "Yen strengthens on safe-haven demand",
                    "Crude oil jumps on supply concerns"
                ] * 10,  # Repeat for more data
                'sentiment_label': [
                    'bullish', 'neutral', 'bearish', 'bullish', 'neutral',
                    'bearish', 'bearish', 'neutral', 'bullish', 'bullish'
                ] * 10,
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
                'currency_pair': ['EUR/USD', 'GBP/JPY', 'USD/CHF', 'XAU/USD', 'WTI/USD'] * 20
            }
            df = pd.DataFrame(sample_data)
        else:
            df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Sentiment distribution:\n{df['sentiment_label'].value_counts()}")
        
        # Preprocess text for both approaches
        df['processed_traditional'] = df['headline'].apply(
            self.preprocessor.preprocess_for_traditional_ml
        )
        df['processed_transformer'] = df['headline'].apply(
            self.preprocessor.preprocess_for_transformer
        )
        
        # Encode labels
        df['label_encoded'] = df['sentiment_label'].map(self.label_mapping)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                           stratify=df['sentiment_label'])
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df
    
    def train_traditional_ml(self, train_df: pd.DataFrame) -> Pipeline:
        """Train TF-IDF + Multinomial Naive Bayes model"""
        logger.info("Training traditional ML model (TF-IDF + Multinomial NB)")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('nb', MultinomialNB())
        ])
        
        # Hyperparameter tuning
        param_grid = {
            'tfidf__max_features': [3000, 5000, 7000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'nb__alpha': [0.1, 0.5, 1.0, 2.0]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', 
                                 n_jobs=-1, verbose=1)
        grid_search.fit(train_df['processed_traditional'], train_df['label_encoded'])
        
        self.tfidf_pipeline = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save model
        joblib.dump(self.tfidf_pipeline, self.model_save_path / 'tfidf_nb_model.pkl')
        
        return self.tfidf_pipeline
    
    def train_finbert(self, train_df: pd.DataFrame) -> None:
        """Fine-tune FinBERT model"""
        logger.info("Training FinBERT model")
        
        try:
            # Load tokenizer and model
            model_name = "yiyanghkust/finbert-tone"
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )
            
            # Prepare dataset
            train_texts = train_df['processed_transformer'].tolist()
            train_labels = train_df['label_encoded'].tolist()
            
            train_encodings = self.finbert_tokenizer(
                train_texts, truncation=True, padding=True, max_length=128
            )
            
            train_dataset = Dataset.from_dict({
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask'],
                'labels': train_labels
            })
            
            # Training arguments - more conservative for demo
            training_args = TrainingArguments(
                output_dir=str(self.model_save_path / 'finbert_checkpoints'),
                num_train_epochs=1,  # Reduced for faster training
                per_device_train_batch_size=8,  # Reduced batch size
                per_device_eval_batch_size=8,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=str(self.model_save_path / 'logs'),
                logging_steps=10,
                evaluation_strategy="no",
                save_strategy="epoch",
                load_best_model_at_end=False,
                learning_rate=5e-5,  # Slightly higher learning rate
                no_cuda=torch.cuda.is_available() == False,  # Use CPU if no GPU
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.finbert_model,
                args=training_args,
                train_dataset=train_dataset,
                tokenizer=self.finbert_tokenizer,
            )
            
            # Train model
            trainer.train()
            
            # Save model
            trainer.save_model(str(self.model_save_path / 'finbert_finetuned'))
            self.finbert_tokenizer.save_pretrained(str(self.model_save_path / 'finbert_finetuned'))
            
            # Create pipeline for inference
            self.finbert_pipeline = pipeline(
                "text-classification",
                model=str(self.model_save_path / 'finbert_finetuned'),
                tokenizer=str(self.model_save_path / 'finbert_finetuned'),
                return_all_scores=True
            )
            
            logger.info("FinBERT training completed")
            
        except Exception as e:
            logger.error(f"FinBERT training failed: {e}")
            logger.info("Attempting to use pre-trained FinBERT without fine-tuning...")
            
            try:
                # Fallback to using the original model without fine-tuning
                self.finbert_pipeline = pipeline(
                    "text-classification",
                    model="yiyanghkust/finbert-tone",
                    return_all_scores=True
                )
                logger.info("Using pre-trained FinBERT without fine-tuning")
            except Exception as e2:
                logger.error(f"Failed to load pre-trained FinBERT: {e2}")
                self.finbert_pipeline = None
    
    def evaluate_models(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate both models on test set"""
        logger.info("Evaluating models")
        
        results = {}
        
        # Evaluate traditional ML model
        if self.tfidf_pipeline:
            logger.info("Evaluating TF-IDF + NB model")
            tfidf_pred = self.tfidf_pipeline.predict(test_df['processed_traditional'])
            tfidf_proba = self.tfidf_pipeline.predict_proba(test_df['processed_traditional'])
            
            results['traditional_ml'] = {
                'accuracy': accuracy_score(test_df['label_encoded'], tfidf_pred),
                'classification_report': classification_report(
                    test_df['label_encoded'], tfidf_pred, 
                    target_names=['bullish', 'neutral', 'bearish']
                ),
                'confusion_matrix': confusion_matrix(test_df['label_encoded'], tfidf_pred),
                'predictions': tfidf_pred,
                'probabilities': tfidf_proba
            }
        
        # Evaluate FinBERT model
        if self.finbert_pipeline:
            logger.info("Evaluating FinBERT model")
            finbert_predictions = []
            finbert_probabilities = []
            
            for text in test_df['processed_transformer']:
                result = self.finbert_pipeline(text)
                # Map FinBERT labels to our format
                label_map = {'LABEL_0': 0, 'LABEL_1': 1, 'LABEL_2': 2}
                pred_label = label_map[result[0]['label']]
                finbert_predictions.append(pred_label)
                
                # Get probabilities for all classes
                probs = [0.0, 0.0, 0.0]
                for item in result:
                    probs[label_map[item['label']]] = item['score']
                finbert_probabilities.append(probs)
            
            results['finbert'] = {
                'accuracy': accuracy_score(test_df['label_encoded'], finbert_predictions),
                'classification_report': classification_report(
                    test_df['label_encoded'], finbert_predictions,
                    target_names=['bullish', 'neutral', 'bearish']
                ),
                'confusion_matrix': confusion_matrix(test_df['label_encoded'], finbert_predictions),
                'predictions': finbert_predictions,
                'probabilities': finbert_probabilities
            }
        
        self.evaluation_results = results
        
        # Print results
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()} RESULTS:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Classification Report:\n{metrics['classification_report']}")
        
        return results
    
    def plot_confusion_matrices(self) -> None:
        """Plot confusion matrices for both models"""
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return
        
        fig, axes = plt.subplots(1, len(self.evaluation_results), figsize=(12, 5))
        if len(self.evaluation_results) == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Bullish', 'Neutral', 'Bearish'],
                       yticklabels=['Bullish', 'Neutral', 'Bearish'],
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name.title()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.model_save_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {self.model_save_path / 'confusion_matrices.png'}")
    
    def load_models(self) -> None:
        """Load pre-trained models for inference"""
        logger.info("Loading pre-trained models")
        
        # Load traditional ML model
        tfidf_path = self.model_save_path / 'tfidf_nb_model.pkl'
        if tfidf_path.exists():
            try:
                self.tfidf_pipeline = joblib.load(tfidf_path)
                logger.info("Loaded TF-IDF + NB model")
            except Exception as e:
                logger.error(f"Failed to load TF-IDF model: {e}")
                self.tfidf_pipeline = None
        else:
            logger.info("TF-IDF model not found, will train new model")
        
        # Load FinBERT model
        finbert_path = self.model_save_path / 'finbert_finetuned'
        if finbert_path.exists():
            try:
                self.finbert_pipeline = pipeline(
                    "text-classification",
                    model=str(finbert_path),
                    tokenizer=str(finbert_path),
                    return_all_scores=True
                )
                logger.info("Loaded fine-tuned FinBERT model")
            except Exception as e:
                logger.error(f"Failed to load fine-tuned FinBERT: {e}")
                logger.info("Attempting to load pre-trained FinBERT...")
                try:
                    self.finbert_pipeline = pipeline(
                        "text-classification",
                        model="yiyanghkust/finbert-tone",
                        return_all_scores=True
                    )
                    logger.info("Loaded pre-trained FinBERT model")
                except Exception as e2:
                    logger.error(f"Failed to load pre-trained FinBERT: {e2}")
                    self.finbert_pipeline = None
        else:
            logger.info("Fine-tuned FinBERT not found, trying pre-trained...")
            try:
                self.finbert_pipeline = pipeline(
                    "text-classification",
                    model="yiyanghkust/finbert-tone",
                    return_all_scores=True
                )
                logger.info("Loaded pre-trained FinBERT model")
            except Exception as e:
                logger.error(f"Failed to load pre-trained FinBERT: {e}")
                self.finbert_pipeline = None
    
    def predict(self, headline: str) -> Dict:
        """Predict sentiment for a given headline"""
        results = {
            'headline': headline,
            'finbert': None,
            'nb': None,
            'primary': None
        }
        
        # Traditional ML prediction
        if self.tfidf_pipeline:
            processed_text = self.preprocessor.preprocess_for_traditional_ml(headline)
            nb_pred = self.tfidf_pipeline.predict([processed_text])[0]
            nb_proba = self.tfidf_pipeline.predict_proba([processed_text])[0]
            nb_confidence = float(np.max(nb_proba))
            
            results['nb'] = {
                'label': self.reverse_label_mapping[nb_pred],
                'score': nb_confidence
            }
        
        # FinBERT prediction
        if self.finbert_pipeline:
            processed_text = self.preprocessor.preprocess_for_transformer(headline)
            finbert_result = self.finbert_pipeline(processed_text)
            
            # FinBERT returns nested list: [[{'label': 'Positive', 'score': 0.99}, ...]]
            if finbert_result and len(finbert_result) > 0 and len(finbert_result[0]) > 0:
                scores = finbert_result[0]  # Get the inner list
                
                # Get the highest scoring prediction
                best_pred = max(scores, key=lambda x: x['score'])
                
                # Map FinBERT labels to our format (Positive=bullish, Negative=bearish, Neutral=neutral)
                label_map = {
                    'Positive': 'bullish', 
                    'Negative': 'bearish', 
                    'Neutral': 'neutral'
                }
                
                results['finbert'] = {
                    'label': label_map[best_pred['label']],
                    'score': float(best_pred['score'])
                }
                
                # Determine primary model based on confidence
                if results['finbert']['score'] >= 0.80:
                    results['primary'] = 'finbert'
                elif results['nb']:
                    results['primary'] = 'nb'
                else:
                    results['primary'] = 'finbert'
            else:
                logger.warning("FinBERT returned unexpected result format")
                results['primary'] = 'nb' if results['nb'] else None
        else:
            results['primary'] = 'nb' if results['nb'] else None
        
        return results

# Training script
def train_models(dataset_path: str = "/path/to/forex_news_annotated_dataset.csv"):
    """Main training function"""
    analyzer = ForexSentimentAnalyzer()
    
    # Load and preprocess data
    train_df, test_df = analyzer.load_and_preprocess_data(dataset_path)
    
    # Train traditional ML model
    analyzer.train_traditional_ml(train_df)
    
    # Train FinBERT model
    analyzer.train_finbert(train_df)
    
    # Evaluate models
    analyzer.evaluate_models(test_df)
    
    # Plot confusion matrices
    analyzer.plot_confusion_matrices()
    
    logger.info("Training completed successfully!")
    return analyzer

if __name__ == "__main__":
    # Run training
    analyzer = train_models()