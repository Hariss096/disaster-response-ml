import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Input: file path of messages and categories csv files as strings
    Returns: Single dataframe (messages and categories merged on "id" column)
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """
    Input: Raw dataframe with categories in 1 column
    Splits categories column to 36 different categories and 
    retain only numeric values using string formatting
    Returns: Single df with each category binary encoded
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2]).values
    categories.columns = category_colnames

    # strip extra text from category values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Remove original categories column from dataframe
    df.drop("categories", axis=1, inplace=True)
    # Concatenate splitted categories with dataframe
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Input: dataframe to save into the database, database name as string
    Creates db with desired name and saves dataframe as database table with provided name
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('transformed_data', engine, index=False, if_exists='replace')


def main():
    """
    Runs ETL pipeline, Extracts data from csv files, Transforms data and then
    Loads transformed data to database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()