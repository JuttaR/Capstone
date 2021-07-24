import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(database_filepath):
    """
    Reads in translated and tokenized comments from SQLite database and produces csv for manual rating

    INPUT:
        database_filepath: filepath to SQLite database

    OUTPUTS:
        None
    """
    # Create SQLAlchemy engine and read in data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df_comments_new', engine)

    # Drop same comment id
    df.drop_duplicates(subset=['comment_id'], keep='last', inplace=True)

    # Drop same comment text across posts, i.e. score only unique comments
    df.drop_duplicates(subset=['comment_text'], keep='first', inplace=True)

    # Save data to csv for manual rating of sentiment
    df.to_csv('models/comments_for_rating_new.csv', encoding="utf-8")


# DO MANUAL RATING IN EXCEL #


def main():
    if len(sys.argv) == 2:

        database_filepath = sys.argv[1]

        print(f"Loading comments from {database_filepath} ...")
        load_data(database_filepath)

        print('Saved comments_for_rating_new.csv for manual sentiment rating. ' \
              'Please run train_classifier.py after finishing & saving ratings.')

    else:
        print('Please provide the filepath of the database as the first argument.' \
              '\n\nExample: python models/process_comments.py data/database.db ')


if __name__ == '__main__':
    main()
