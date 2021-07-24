import base64
from datetime import datetime
from deep_translator import GoogleTranslator
import joblib
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.casual import TweetTokenizer
import pandas as pd
import re
from sqlalchemy import create_engine
import sys

nltk.download(['punkt', 'stopwords', 'wordnet'])


def load_data(har_filepath):
    """
    Reads in .har file and outputs relevant entries for further analysis

    INPUT:
        har_filepath: filepath to .har file containing Instagram browsing history

    OUTPUT:
        entries: list of entries in har file
    """
    # Read in HAR file as JSON
    with open(har_filepath, 'r', encoding='utf8') as f:
        raw = json.loads(f.read())

    # Filter raw data for entries
    entries = raw['log']['entries']

    return entries


def filter_data(entries):
    """
    Filters entries for different regex patterns and outputs relevant entries for further analysis

    INPUT:
        entries: list of entries in har file

    OUTPUTS:
        url_indices: list of indices of entries that match specified url pattern
        url_decoded_shortcode: list of shortcodes of viewed Instagram posts (None if only timesline was browsed)
        url_followers: list of indices of entries that match specified pattern to extract follower data
    """
    # Match pattern for relevant URLs, followers & shortcodes, i.e. the unique identifier of a post in the URL
    # e.g. CIeQ4YfpGLA in https://www.instagram.com/p/CIeQ4YfpGLA/
    url_pattern = re.compile("https:\/\/www\.instagram\.com\/graphql\/query\/\?query_hash=.*")
    shortcode_pattern = re.compile("(?<=shortcode%22%3A%22)(.{11})")
    follower_pattern = re.compile(".*?__a=1")

    # Filter for relevant request urls, shortcodes and follower data
    url_indices = []
    url_decoded_shortcode = []
    url_followers = []
    for i in range(len(entries)):
        if url_pattern.match(entries[i]['request']['url']):
            url_indices.append(i)
            if shortcode_pattern.search(entries[i]['request']['url']):
                url_decoded_shortcode.append(shortcode_pattern.search(entries[i]['request']['url']).group(1))
            else:
                url_decoded_shortcode.append(None)
        elif follower_pattern.match(entries[i]['request']['url']):
            url_followers.append(i)
        else:
            continue

    return url_indices, url_decoded_shortcode, url_followers


def decoding_entries(entries, url_indices, url_followers):
    """
    Decodes relevant entries with base64 and utf-8 decoding for further analysis

    INPUTS:
        entries: list of entries in .har file
        url_indices: list of indices of entries that match specified pattern
        url_followers: list of indices of entries that match specified pattern to extract follower data

    OUTPUTS:
        valid_posts: List of decoded content of relevant entries
        browsed_at: List of timestamps to allow updates of database
        valid_followers: List of decoded content of relevant follower info
        browsed_at_followers: List of timestamps to allow updates of follower database
    """
    # Extracting relevant text from utf8 and base64 encoded posts
    valid_posts = []
    browsed_at = []
    for index in url_indices:
        try:
            # utf8: Load JSON
            post = json.loads(entries[index]['response']['content']['text'])
            valid_posts.append(post)
            browsed = datetime.strptime(entries[index]['startedDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
            browsed_at.append(browsed)

        except ValueError:  # Raised if encoded in base64
            try:
                # Decode base64: Load JSON
                post = json.loads(base64.b64decode(entries[index]['response']['content']['text']))
                valid_posts.append(post)
                browsed = datetime.strptime(entries[index]['startedDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
                browsed_at.append(browsed)

            except (ValueError, KeyError) as error:
                continue

        except KeyError:
            continue

    # Extracting relevant text from utf8 and base64 encoded follower info
    valid_followers = []
    browsed_at_followers = []
    for index in url_followers:
        try:
            # utf8: Load JSON
            follower = json.loads(entries[index]['response']['content']['text'])
            valid_followers.append(follower)
            browsed = datetime.strptime(entries[index]['startedDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
            browsed_at_followers.append(browsed)

        except ValueError:  # Raised if encoded in base64
            try:
                # Decode base64: Load JSON
                follower = json.loads(base64.b64decode(entries[index]['response']['content']['text']))
                valid_followers.append(follower)
                browsed = datetime.strptime(entries[index]['startedDateTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
                browsed_at_followers.append(browsed)

            except (ValueError, KeyError) as error:
                continue

        except KeyError:
            continue

    return valid_posts, browsed_at, valid_followers, browsed_at_followers


def create_dfs():
    """
    Creates pandas dataframes to hold relevant post, comment and follower data

    INPUT:
        None

    OUTPUTS:
        df_posts: pandas dataframe containing Instagram posts data
        df_comments: pandas dataframe containing Instagram comments for respective posts
        df_followers: pandas dataframe containing Instagram follower data for respective profiles
    """
    # Create dataframe to hold relevant post data
    df_posts = pd.DataFrame({'medium_id': pd.Series([], dtype='int64'),
                             'updated': pd.Series([], dtype='datetime64[ns]'),
                             'shortcode': pd.Series([], dtype='str'),
                             'owner_id': pd.Series([], dtype='int64'),
                             'username': pd.Series([], dtype='str'),
                             'taken_at': pd.Series([], dtype='datetime64[ns]'),
                             'typename': pd.Series([], dtype='int64'),
                             'is_video': pd.Series([], dtype='bool'),
                             'caption': pd.Series([], dtype='str'),
                             'likes': pd.Series([], dtype='int64'),
                             'comments': pd.Series([], dtype='int64'),
                             'video_views': pd.Series([], dtype='int64'),
                             'image': pd.Series([], dtype='str')})

    df_posts.set_index(['medium_id'])

    # Create dataframe to hold relevant comment data
    df_comments = pd.DataFrame({'comment_id': pd.Series([], dtype='int64'),
                                'updated': pd.Series([], dtype='datetime64[ns]'),
                                'shortcode': pd.Series([], dtype='str'),
                                'created_at': pd.Series([], dtype='datetime64[ns]'),
                                'comment_text': pd.Series([], dtype='str'),
                                'comment_likes': pd.Series([], dtype='int64')})

    df_comments.set_index(['comment_id'])

    # Create dataframe to hold relevant follower data
    df_followers = pd.DataFrame({'username': pd.Series([], dtype='str'),
                                 'followers': pd.Series([], dtype='int64'),
                                 'updated': pd.Series([], dtype='datetime64[ns]')})

    df_followers.set_index(['username'])

    return df_posts, df_comments, df_followers


def extract_data(df_posts, df_comments, df_followers, valid_posts, browsed_at, valid_followers, browsed_at_followers,
                 url_decoded_shortcode):
    """
    Extracts relevant data for saving in SQLite database

    INPUTS:
        df_posts: empty pandas dataframe with only column names and data types
        df_comments: empty pandas dataframe with only column names and data types
        df_followers: empty pandas dataframe with only column names and data types
        valid_posts: List of decoded content of relevant entries
        browsed_at: List of timestamps to allow updates of database
        valid_followers: List of decoded content of relevant follower info
        browsed_at_followers: List of timestamps to allow updates of follower database
        url_decoded_shortcode: list of shortcodes of viewed Instagram posts (None if only timeline was browsed)

    OUTPUTS:
        df_posts: filled pandas dataframe with posts data
        df_comments: filled pandas dataframe with comments data
        df_followers: filled pandas dataframe with followers data
    """
    # Pattern to match images
    image_pattern = re.compile(".{47}\.jpg")

    # Extract data from posts
    for i in range(len(valid_posts)):
        try:
            # Works with single posts
            medium_id = valid_posts[i]['data']['shortcode_media']['id']
            updated = browsed_at[i]
            shortcode = valid_posts[i]['data']['shortcode_media']['shortcode']
            owner_id = valid_posts[i]['data']['shortcode_media']['owner']['id']
            username = valid_posts[i]['data']['shortcode_media']['owner']['username']
            taken_at = datetime.fromtimestamp(valid_posts[i]['data']['shortcode_media']['taken_at_timestamp'])
            typename = valid_posts[i]['data']['shortcode_media']['__typename']
            is_video = valid_posts[i]['data']['shortcode_media']['is_video']
            caption = valid_posts[i]['data']['shortcode_media']['edge_media_to_caption']['edges'][0]['node']['text']
            likes = valid_posts[i]['data']['shortcode_media']['edge_media_preview_like']['count']
            comments = valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['count']
            if is_video:
                video_views = valid_posts[i]['data']['shortcode_media']['video_view_count']
            else:
                video_views = None
            image = image_pattern.search(valid_posts[i]['data']['shortcode_media']['display_url']).group()

            df_posts = df_posts.append({'medium_id': medium_id,
                                        'updated': updated,
                                        'shortcode': shortcode,
                                        'owner_id': owner_id,
                                        'username': username,
                                        'taken_at': taken_at,
                                        'typename': typename,
                                        'is_video': is_video,
                                        'caption': caption,
                                        'likes': likes,
                                        'comments': comments,
                                        'video_views': video_views,
                                        'image': image},
                                       ignore_index=True)

        except KeyError:  # Raised with timeline
            try:
                # Works with timeline
                for j in range(len(valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'])):
                    try:
                        medium_id = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'id']
                        updated = browsed_at[i]
                        shortcode = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'shortcode']
                        owner_id = \
                            valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node']['owner'][
                                'id']
                        username = \
                            valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node']['owner'][
                                'username']
                        taken_at = datetime.fromtimestamp(
                            valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                                'taken_at_timestamp'])
                        typename = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            '__typename']
                        is_video = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'is_video']
                        caption = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'edge_media_to_caption']['edges'][0]['node']['text']
                        likes = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'edge_media_preview_like']['count']
                        comments = valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                            'edge_media_to_comment']['count']
                        if is_video:
                            video_views = \
                                valid_posts[i]['data']['user']['edge_owner_to_timeline_media']['edges'][j]['node'][
                                    'video_view_count']
                        else:
                            video_views = None
                        # print(f"parsed timeline post {j} of entry {i}: {shortcode}")
                        df_posts = df_posts.append({'medium_id': medium_id,
                                                    'updated': updated,
                                                    'shortcode': shortcode,
                                                    'owner_id': owner_id,
                                                    'username': username,
                                                    'taken_at': taken_at,
                                                    'typename': typename,
                                                    'is_video': is_video,
                                                    'caption': caption,
                                                    'likes': likes,
                                                    'comments': comments,
                                                    'video_views': video_views},
                                                   ignore_index=True)
                    except (KeyError, IndexError) as errors:
                        continue
            except KeyError:
                try:
                    # Works with comments
                    for k in range(
                            len(valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['edges'])):
                        try:
                            comment_id = \
                                valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['edges'][k][
                                    'node']['id']
                            updated = browsed_at[i]
                            shortcode = url_decoded_shortcode[i]
                            created_at = datetime.fromtimestamp(
                                valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['edges'][k][
                                    'node']['created_at'])
                            comment_text = \
                                valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['edges'][k][
                                    'node']['text']
                            comment_likes = \
                                valid_posts[i]['data']['shortcode_media']['edge_media_to_parent_comment']['edges'][k][
                                    'node']['edge_liked_by']['count']

                            df_comments = df_comments.append({'comment_id': comment_id,
                                                              'updated': updated,
                                                              'shortcode': shortcode,
                                                              'created_at': created_at,
                                                              'comment_text': comment_text,
                                                              'comment_likes': comment_likes},
                                                             ignore_index=True)
                        except (KeyError, IndexError) as errors:
                            continue
                except (KeyError, IndexError) as errors:
                    continue

    # Extract data from followers
    for i in range(len(valid_followers)):
        try:
            username = valid_followers[i]['graphql']['user']['username']
            followers = valid_followers[i]['graphql']['user']['edge_followed_by']['count']
            updated = browsed_at_followers[i]
            df_followers = df_followers.append({'username': username,
                                                'followers': followers,
                                                'updated': updated},
                                               ignore_index=True)
        except (KeyError, IndexError, ValueError) as errors:
            continue

    return df_posts, df_comments, df_followers


def clean_data(df_posts, df_comments, df_followers):
    """
    Drops outdated or irrelevant data from dataframes and sorts results

    INPUTS:
        df_posts: pandas dataframe containing Instagram posts data
        df_comments: pandas dataframe containing Instagram comments for respective posts
        df_followers: pandas dataframe containing Instagram follower data for respective profiles

    OUTPUTS:
        df_posts: pandas dataframe containing cleaned posts data
        df_followers_clean: pandas dataframe containing cleaned Instagram follower data for respective profiles
        df_comments_filtered: pandas dataframe containing cleaned Instagram comments on posts of respective profiles
    """
    # List of relevant profiles
    competitive_set = ['mini', 'audiofficial', 'fiat']

    # Drops data from irrelevant profiles
    df_posts_filtered = df_posts[df_posts.username.isin(competitive_set)]
    df_followers_filtered = df_followers[df_followers.username.isin(competitive_set)]

    # Drop duplicates keeping only the last occurrence
    df_posts_clean = df_posts_filtered.drop_duplicates(subset=['shortcode'], keep='last')
    df_followers_clean = df_followers_filtered.drop_duplicates(subset=['username'], keep='last')

    # Sort taken_at
    df_posts_sorted = df_posts_clean.sort_values(by='taken_at', ascending=True)

    # Evaluate only comments to relevant posts
    df_comments_filtered = df_comments[df_comments.shortcode.isin(df_posts_sorted.shortcode)]

    return df_posts, df_followers_clean, df_comments_filtered


def translate_comments(df_comments_filtered):
    """
    Translates with Google Translate using auto-detection of language to convert all comments into English.
    PLEASE NOTE: This process takes some time (up to 2 seconds per comment)

    INPUT:
        df_comments_filtered: pandas dataframe containing cleaned Instagram comments on posts of respective profiles

    OUTPUT:
        df_comments_translated: pandas dataframe containing only English-language comments
    """
    # Translate comments into English
    en_comments = []
    for text in df_comments_filtered['comment_text']:
        try:
            en_comment = GoogleTranslator(source='auto', target='en').translate(text=text)
            if en_comment:
                en_comments.append(en_comment)
            else:
                en_comments.append(text)
        except Exception:  # Tried to except just NotValidPayload Exception according to documentation,
            # however this was not working and producing NameError instead
            en_comments.append("google_translator_error")

    df_comments_filtered['translation'] = en_comments

    # Drop rows with translation errors
    df_comments_translated = df_comments_filtered[df_comments_filtered['translation'] != 'google_translator_error']

    return df_comments_translated


def tokenize(text):
    """
    Cleans text data in order to use it for sentiment scoring.

    INPUT:
        Preprocessed filtered comment

    OUTPUT:
        Cleaned comment (w/o urls, normalized, tokenized, w/o stopwords, lemmatized)
    """
    # Use regex expression to detect urls in text
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)

    # Replace all urls with placeholder
    for detected_url in detected_urls:
        text = text.replace(detected_url, "url")

    # Normalization (keeping emoticons for sentiment analysis)
    text = re.sub(r"[.,;:?!()&#’]", "", text.lower())

    # Tokenization
    words = TweetTokenizer().tokenize(text)

    # Removal of stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # Lemmatization (nouns)
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w).strip() for w in words]

    return cleaned_tokens


def classify_comments(df_comments_translated, model_filepath):
    """
    Loads trained model and classifies comments accordingly

    INPUT:
        df_comments_translated: Preprocessed filtered comments

    OUTPUT:
        df_comments_scored: pandas dataframe with comments scored by sentiment
    """
    # Load model
    model = joblib.load(model_filepath)

    # Predict sentiment
    sentiment_class = model.predict(df_comments_translated['translation'])

    # Add predictions to dataframe and replace class names
    df_comments_translated['pred_sentiment'] = sentiment_class.tolist()
    df_comments_scored = df_comments_translated.replace({'pred_sentiment': {0: 'negative', 1: 'neutral', 2: 'positive'}})

    return df_comments_scored


def save_data(df_posts, df_followers_clean, df_comments_scored, database_filepath):
    """
    Saves dataframes to SQLite database

    INPUTS:
        df_posts: pandas dataframe containing cleaned posts data merged with sentiment score of comments
        df_followers_clean: pandas dataframe containing cleaned Instagram follower data for respective profiles
        df_comments_scored: pandas dataframe scored cleaned Instagram comments on posts of respective profiles
        database_filepath: database filepath

    OUTPUT:
        None
    """
    # Create SQLAlchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # Save dataframe to SQLite database; replace if already exists
    df_posts.to_sql('df_posts', engine, index=False, if_exists='append')
    df_followers_clean.to_sql('df_followers', engine, index=False, if_exists='append')
    df_comments_scored.to_sql('df_comments', engine, index=False, if_exists='append')


def main():
    if len(sys.argv) == 4:

        har_filepath, database_filepath, model_filepath = sys.argv[1:]

        print(f"Loading har file from {har_filepath} ...")
        entries = load_data(har_filepath)

        print("Filtering data...")
        url_indices, url_decoded_shortcode, url_followers = filter_data(entries)

        print("Decoding entries with utf-8 and/or base64 encoding...")
        valid_posts, browsed_at, valid_followers, browsed_at_followers = decoding_entries(entries, url_indices,
                                                                                          url_followers)
        print(f"Number of decoded posts: {len(valid_posts)}")
        print(f"Number of decoded follower data: {len(valid_followers)}")

        print("Creating dataframes...")
        df_posts, df_comments, df_followers = create_dfs()

        print("Extracting data...")
        df_posts, df_comments, df_followers = extract_data(df_posts, df_comments, df_followers, valid_posts, browsed_at,
                                                           valid_followers, browsed_at_followers, url_decoded_shortcode)

        print("Cleaning data...")
        df_posts, df_followers_clean, df_comments_filtered = clean_data(df_posts, df_comments, df_followers)

        print("Translating comments... please note this may take some time as it is using Google Translate...")
        df_comments_translated = translate_comments(df_comments_filtered)

        print("Loading model, tokenizing and classifying comments...")
        df_comments_scored = classify_comments(df_comments_translated, model_filepath)

        print(f"Saving data to {database_filepath}...")
        save_data(df_posts, df_followers_clean, df_comments_scored, database_filepath)

        print("Cleaned data saved to database!")

        print('Run process_comments.py next if you would like to rate the sentiment of comments manually ' \
              'otherwise run app.py to show dashboard')

    else:
        print('Please provide the filepath of the Instagram browsing history ' \
              '(.har file) as the first argument, the filepath of the database ' \
              'to save the cleaned data to as the second argument, and the filepath ' \
              'of the model to classify the sentiment of a comment as third argument. ' \
              '\n\nExample: python data/parse_data.py ' \
              'data/audi_example.har ' \
              'data/database.db ' \
              'models/model.pkl ')


if __name__ == '__main__':
    main()
