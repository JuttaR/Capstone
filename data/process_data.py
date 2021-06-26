import base64
from datetime import datetime
import json
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import sys
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

        INPUT:
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
                                'comment_likes': pd.Series([], dtype='int64'),
                                'vader_sentiment': pd.Series([], dtype='float64')})

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

            # print(f"parsed single post {i}: {shortcode}")
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
                    # works with comments
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
                            vader_sentiment = SentimentIntensityAnalyzer().polarity_scores(comment_text)['compound']

                            df_comments = df_comments.append({'comment_id': comment_id,
                                                              'updated': updated,
                                                              'shortcode': shortcode,
                                                              'created_at': created_at,
                                                              'comment_text': comment_text,
                                                              'comment_likes': comment_likes,
                                                              'vader_sentiment': vader_sentiment},
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
    Drops outdated or irrelevant data from dataframes; scores and then merges average sentiment data from comments using
    VADER with corresponding post

    INPUTS:
        df_posts: pandas dataframe containing Instagram posts data
        df_comments: pandas dataframe containing Instagram comments for respective posts
        df_followers: pandas dataframe containing Instagram follower data for respective profiles

    OUTPUT:
        df_posts_comments: pandas dataframe containing cleaned posts data merged with sentiment score of comments
        df_followers_clean: pandas dataframe containing cleaned Instagram follower data for respective profiles
    """
    # List of relevant profiles
    competitive_set = ['mini', 'audiofficial', 'fiat']

    # Drop posts from irrelevant profiles
    df_posts_filtered = df_posts[df_posts.username.isin(competitive_set)]
    df_followers_filtered = df_followers[df_followers.username.isin(competitive_set)]

    # Drop duplicates keeping only the last occurrence
    df_posts_clean = df_posts_filtered.drop_duplicates(subset=['shortcode'], keep='last')
    df_followers_clean = df_followers_filtered.drop_duplicates(subset=['username'], keep='last')

    # Sort taken_at in descending order
    df_posts_sorted = df_posts_clean.sort_values(by='taken_at', ascending=True)

    # Take average of comment sentiment
    avg_sentiment = pd.pivot_table(df_comments,
                                   values='vader_sentiment',
                                   index=['shortcode'],
                                   aggfunc=np.mean).reset_index()
    # Merge sentiment data with posts data
    df_posts_comments = df_posts_sorted.merge(avg_sentiment, on='shortcode', how='left')

    return df_posts_comments, df_followers_clean


def save_data(df_posts_comments, df_followers_clean, database_filepath):
    """
    Saves dataframes to SQLite database

    INPUTS:
        df_posts_comments: pandas dataframe containing cleaned posts data merged with sentiment score of comments
        df_followers_clean: pandas dataframe containing cleaned Instagram follower data for respective profiles
        database_filepath: database filepath

    OUTPUT:
        None
    """
    # create SQLAlchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # save dataframe to SQLite database; replace if already exists
    df_posts_comments.to_sql('df', engine, index=False, if_exists='append')
    df_followers_clean.to_sql('df_followers', engine, index=False, if_exists='append')


def main():
    if len(sys.argv) == 3:

        har_filepath, database_filepath = sys.argv[1:]

        print(f"Loading har file from {har_filepath} ...")
        entries = load_data(har_filepath)

        print(f"Filtering data...")
        url_indices, url_decoded_shortcode, url_followers = filter_data(entries)

        print(f"Decoding entries with utf-8 and/or base64 encoding...")
        valid_posts, browsed_at, valid_followers, browsed_at_followers = decoding_entries(entries, url_indices,
                                                                                          url_followers)
        print(f"Number of decoded posts: {len(valid_posts)}")
        print(f"Number of decoded follower data: {len(valid_followers)}")

        print(f"Creating dataframes...")
        df_posts, df_comments, df_followers = create_dfs()

        print(f"Extracting data...")
        df_posts, df_comments, df_followers = extract_data(df_posts, df_comments, df_followers, valid_posts, browsed_at,
                                                           valid_followers, browsed_at_followers, url_decoded_shortcode)

        print('Cleaning data...')
        df_posts_comments, df_followers_clean = clean_data(df_posts, df_comments, df_followers)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df_posts_comments, df_followers_clean, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepath of the Instagram browsing history ' \
              '(.har file) as the first argument, as well as the filepath of the ' \
              'database to save the cleaned data to as the second argument.' \
              '\n\nExample: python data/process_data.py ' \
              'data/audi_example.har ' \
              'data/database.db ')


if __name__ == '__main__':
    main()