import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import sys

nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(ratings_filepath):
    """
    Reads in data of rated comments and outputs dataframes for ML model

    INPUT:
        ratings_filepath: filepath to rated comments

    OUTPUTS:
        X: features dataframe
        y: target dataframe
        classes: list of class names
    """
    # Load data from csv
    df = pd.read_csv(ratings_filepath, encoding="utf-8", delimiter=";")

    # Create features (X) and target (y) dataframes
    X = df['translation']

    # Use LabelEncoder to convert class names into integers
    le = LabelEncoder()
    le.fit(df['sentiment'])

    # Display class mapping
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Mapping of classes: {le_name_mapping}")

    # Use mapped classes to transform targets
    y = le.transform(df['sentiment'])

    # Save classes for printing results later
    classes = le.classes_

    return X, y, classes


def tokenize(text):
    """
    Cleans text data in order to use it for machine learning later.

    INPUT:
        text: Preprocessed filtered comment

    OUTPUT:
        cleaned_tokens: Cleaned text (w/o urls, normalized, tokenized, w/o stopwords, lemmatized) as list
    """
    # Use regex expression to detect urls in text
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)

    # Replace all urls with placeholder
    for detected_url in detected_urls:
        text = text.replace(detected_url, "url")

    # Normalization, but keeping emoticons for sentiment analysis
    text = re.sub(r"[.,;:?!()&#’]", "", text.lower())

    # Tokenization
    words = TweetTokenizer().tokenize(text)

    # Removal of stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization (nouns)
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in words]

    return cleaned_tokens


def build_model():
    """
    Builds a ML Pipeline using a RandomForest classifier and GridSearchCV to tune it to its optimal hyperparameters.

    INPUT:
        None

    OUTPUT:
        model: GridSearch output from cross-validation
    """
    # Weights to better balance dataset
    weights = {0: 6.5, 1: 3.0, 2: 1.0}

    '''
    # Set up of ML Pipeline for DecisionTreeClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', DecisionTreeClassifier(random_state=42, class_weight=weights))
    ])

    params = {
        'clf__max_depth': [3, 5],
        'clf__min_samples_split': [2, 4],  # min number of data points in node before the node is split
        'clf__max_features': ['log2', 'auto']  # max number of features considered for splitting a node
    }
    '''

    # Set up of ML Pipeline for RandomForestClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(random_state=42, class_weight=weights))
    ])

    params = {
        'clf__n_estimators': [5, 10, 25],  # number of trees in the forest
        'clf__max_depth': [2, 3],  # max depth of the tree
        'clf__min_samples_split': [2, 4],  # min number of data points in node before the node is split
        'clf__max_features': ['log2', 'auto']  # max number of features considered for splitting a node
    }

    # Create model with GridSearchCV
    model = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', verbose=0, n_jobs=1, cv=3)

    return model


def evaluate_model(model, X_test, y_test, classes):
    """
    Evaluate trained ML model performance against test data.

    INPUTS:
        model: Trained ML model
        X_test: Test data (features)
        y_test: Test data (targets)
        classes: Names of classes

    OUTPUT:
        Printed results from GridSearch; classification report and accuracy for all classes
    """
    # Print results from GridSearch
    model_results_table = pd.concat([pd.DataFrame(model.cv_results_["params"]),
                                    pd.DataFrame(model.cv_results_["mean_test_score"],
                                                 columns=["Accuracy"])], axis=1)

    print("GridSearch Results Table")
    print(model_results_table)

    print(f"Best-performing parameters from GridSearch: {model.best_params_}")
    print(f"Best score using above parameters: {model.best_score_}")

    # Make predictions on test data
    y_pred = model.predict(X_test)

    # Print counts of predicted classes
    predicted_classes = dict(zip(*np.unique(y_pred, return_counts=True)))
    print(f"Predicted classes: {predicted_classes}")

    # Print classification reports
    print("Classification report incl. overall micro, macro, weighted and sample averages")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))


def save_model(model, model_filepath):
    """
    Save trained model to pickle file

    INPUTS:
        model: trained model
        model_filepath: filepath to save model as pickle file (byte stream)

    OUTPUT:
        pickle file // none
    """
    # Save trained model in binary
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        ratings_filepath, model_filepath = sys.argv[1:]
        print(f"Loading data from ratings: {ratings_filepath}")
        X, y, classes = load_data(ratings_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, classes)

        print(f"Saving model as {model_filepath}")
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the comments_rated.csv ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'models/train_classifier.py models/comments_rated.csv models/model.pkl')


if __name__ == '__main__':
    main()
