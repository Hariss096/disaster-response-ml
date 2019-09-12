# import libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import pickle

def load_data(database_filepath):
    """
    Input: Database file path as string
    Reads the table within db that contains output of ETL pipeline
    Returns: Messages, Categories and their names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_query("SELECT * FROM transformed_data", engine)
    X = df.message.values
    y = df.drop(["id", "message", "original", "genre"], axis=1)
    return X, y, y.columns

def tokenize(text):
    """
    Input: Text message as string
    Returns: Normalized, Tokenized, Lemmatized text
    """
    # Normalizing
    text = text.lower()
    # tokenizing
    tokenized = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # lemmatizing
    clean_tokens = []
    for tok in tokenized:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Returns: ML pipeline
    """
    pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Inputs: pre-built model, test set features and labels, category names
    Prints classification report for each category
    """
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(classification_report(Y_test.iloc[i].values, y_pred[i]))


def save_model(model, model_filepath):
    """
    Inputs: pre-built model, file path as string where model should be saved
    Saves model to the desired filepath
    """
    output = open(model_filepath, 'wb')
    pickle.dump(model, output)


def main():
    """
    Runs ML pipeline: Loads data from database, builds model, trains model, evaluate and then
    Save it as pickle file
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        parameters = {
        'cvect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [1, 10],
        }
        cv = GridSearchCV(model, param_grid=parameters)
        cv.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(cv, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(cv, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()