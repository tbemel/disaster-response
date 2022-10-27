"""
USER STORY: load messages & taggings from the csv file to a dataframe
, merge both, clean the data and save to a sql db
AUTHOR:  Thierry Bemelmans
CREATED: 2022-SEP-27
VERSION: 1.0

ARGUMENTS: 
(1) Filepath of the messages dataset file
(2) Filepath of the categories datasets file
(3) Filepath of the database to save the cleaned data 

COMMAND EXAMPLE: 
    python process_data.py '\
        'disaster_messages.csv disaster_categories.csv '\
        'DisasterResponse.db'
              
"""

import sys
import pandas as pd
import logging
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    USER STORY: reads message & categories file and merge them
    INPUT: message & categories file path
    OUTPUT: merged dataframe on id
    """

    # load messages & categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id', copy=True, left_index=False, right_index=False)
    logging.debug(f'message & categories files reads with shape: {df.shape}')

    return df


def clean_data(df):
    """
    USER STORY: transform the 'categories' columns, spliting the categoriess by ';' 
        into multiple columns with a binary value
    INPUT: unprocessed dataframe
    OUTPUT: cleaned dataframe
    """
    
    # Split the values in the `categories` column on the `;` character so 
    # that each value becomes a separate column. 
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0:0]
    logging.debug(f'Top row of categories with column names:\n {row}')
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [field[:-2] for field in row.values[0]]
    logging.debug(f'Categories columns: {category_colnames}')

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
      

    # drop the original categories column from `df`
    try:
        df.drop('categories', inplace=True, axis=1)
    except: 
        pass    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    df.head()
    
    # check number of duplicates
    logging.warning(f"Duplicated rows found & removed: {df['id'].duplicated().sum()}")
    
    # drop duplicates
    df['id'].drop_duplicates(inplace=True)
    return df
    
def save_data(df, database_filename):
    """
    USER STORY:save the dataframe to a sqlite table
    INPUT: dataframe
    OUTPUT: sql db name and table
    """
    db_name = f'sqlite:///{database_filename}'
    engine = create_engine(db_name)
                    
    tableName = 'messages'
    result = df.to_sql(tableName, engine, if_exists='replace', index=False)    
    logging.info(f'DB Sqlite saved as {database_filename} on table {tableName} \n with result: {result}')
    return result


def main():
    """ 
    Main function, accepts 3 arguments
    ARGUMENTS: 
    (1) Filepath of the messages dataset file
    (2) Filepath of the categories datasets file
    (3) Filepath of the database to save the cleaned data  
    """
                    
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('App started with logging')

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