# Standard library imports
import re
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Constants
DATA_DELIMITER = ':::'
TEXT_CLEAN_PATTERN = r'[^\w\s]'
def load_data(filepath):
    """Load movie data from delimited text file into DataFrame.
    
    Args:
        filepath (str): Path to data file
        
    Returns:
        pd.DataFrame: DataFrame containing movie data
    """
    try:
        df = pd.read_csv(
            filepath,
            sep=DATA_DELIMITER,
            header=None,
            engine='python',
            encoding='utf-8'
        )
        
        if len(df.columns) >= 4:  # Training data
            df.columns = ['id', 'title', 'genre', 'description']
        else:  # Test data
            df.columns = ['id', 'title', 'description']
            
        return df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return pd.DataFrame()

def clean_text(text):
    """Clean and preprocess text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing extra whitespace
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ''
        
    text = text.lower()
    text = re.sub(TEXT_CLEAN_PATTERN, '', text)
    return ' '.join(text.split())

# Data paths (update these to your actual paths)
DATA_DIR = Path(r'F:\DOWNLOADS\Genre Classification Dataset')
train_path = DATA_DIR / 'train_data.txt'
test_path = DATA_DIR / 'test_data.txt'
desc_path = DATA_DIR / 'description.txt'
solution_path = DATA_DIR / 'test_data_solution.txt'

# Load and preprocess data
train_df = load_data(train_path)
test_df = load_data(test_path)

if not train_df.empty:
    train_df['clean_description'] = train_df['description'].apply(clean_text)
    
if not test_df.empty:
    test_df['clean_description'] = test_df['description'].apply(clean_text)
    try:
        solution_df = load_data(solution_path)
        if not solution_df.empty and 'genre' in solution_df.columns:
            test_df['genre'] = solution_df['genre']
    except Exception as e:
        print(f"Warning: Could not load solution file - {str(e)}")

# Load dataset description
try:
    with open(desc_path, 'r', encoding='utf-8') as f:
        dataset_description = f.read()
except Exception as e:
    print(f"Warning: Could not load description file - {str(e)}")
    dataset_description = ""

# Prepare training and validation data
if not train_df.empty:
    X_train = train_df['clean_description']
    y_train = train_df['genre']
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train
    )
else:
    raise ValueError("No training data loaded")

# Prepare test data
X_test = test_df['clean_description'] if not test_df.empty else None
y_test = test_df.get('genre', None)

# Model configurations
models = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('clf', MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        ))
    ]),
    'SVM': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ('clf', LinearSVC(
            class_weight='balanced',
            max_iter=1000
        ))
    ])
}

def evaluate_model(model, X, y_true, name):
    """Evaluate model performance and generate visualizations.
    
    Args:
        model: Trained model
        X: Features
        y_true: True labels
        name: Model name
    """
    y_pred = model.predict(X)
    
    print(f'\n=== {name} ===')
    print(classification_report(
        y_true, 
        y_pred, 
        zero_division=0
    ))
    
    # Confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt='d'
    )
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Train and evaluate models
for name, model in models.items():
    print(f'\nTraining {name}...')
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, name)
