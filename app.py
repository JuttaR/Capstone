from datetime import date
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine

# Load data from database
print("Connecting to SQL database...")
engine = create_engine('sqlite:///data/database.db')

print("Reading in SQL tables...")
df_posts = pd.read_sql_table('df_posts', engine)
df_followers = pd.read_sql_table('df_followers', engine)
df_comments = pd.read_sql_table('df_comments', engine)

print(f"Rows of tables: Posts: {len(df_posts)},"
      f"Followers: {len(df_followers)},"
      f"Comments: {len(df_comments)}, ")

# Drop all non-necessary data for current visualizations
df_posts.drop_duplicates(subset=['shortcode'], keep='last', inplace=True)
df_posts.drop(columns=['medium_id', 'owner_id', 'video_views', 'image', 'typename', 'is_video', 'updated'],
              inplace=True)
df_followers.drop_duplicates(subset=['username'], keep='last', inplace=True)
df_comments.drop_duplicates(subset=['comment_id'], keep='last', inplace=True)
print(f"Rows of tables after: Posts: {len(df_posts)},"
      f"Followers: {len(df_followers)},"
      f"Comments: {len(df_comments)}, ")

# Create day column for aggregating data in visualizations
df_posts['day'] = pd.to_datetime(df_posts['taken_at']).dt.date

# Create sentiment distribution for each post
df_sentiment_grouped = df_comments.groupby(['shortcode', 'pred_sentiment']).size().unstack(fill_value=0)

# Merge sentiments with posts and fill blanks for comments that are not available publicly
df_posts_scored = pd.merge(df_posts, df_sentiment_grouped, how="outer", on='shortcode')
df_posts_scored.update(df_posts_scored[['negative', 'neutral', 'positive']].fillna(0))

# Create dropdown options
profile_options = [{'label': str(profile), 'value': profile} for profile in (df_posts['username'].unique())]

# Import icon
navicon = 'logo.png'

# Initialize app
print("Initializing app...")
app = dash.Dash(__name__, title='MINI Instagram Dashboard', external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# Set chart styling template
mini_template = dict(layout=
                     go.Layout(title_font=dict(family="Lato", size=24, color='#212529'),
                               font=dict(family="Lato", size=12, color='#212529'),
                               plot_bgcolor='#fff', paper_bgcolor='#fff',
                               xaxis=dict(gridcolor='#212529', zerolinecolor='#212529', zerolinewidth=2,),
                               yaxis=dict(gridcolor='#212529', zerolinecolor='#212529', zerolinewidth=2,),
                               hoverlabel=dict(bordercolor='#212529', font_size=14, font_family="Lato"),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                               ))

# Create navigation bar with icon, dropdown and links
navbar = dbc.Navbar(dbc.Container([html.A(
                                        dbc.Row([dbc.Col(html.Img(src=app.get_asset_url(navicon), height="50px")),
                                                 dbc.Col(dbc.NavbarBrand("Interactive MINI Instagram Dashboard",
                                                                         className="navbar-brand")), ],
                                                align="left", no_gutters=True,),
                                        href="#",),
                                  dbc.NavbarToggler(id="navbar-toggler"),
                                  dbc.Collapse(dbc.Nav([dbc.NavItem(
                                      dbc.NavLink("Alpha Version", href="#")),
                                      dbc.DropdownMenu(children=[
                                          dbc.DropdownMenuItem("MINI profile",
                                                               href="https://www.instagram.com/mini",
                                                               target="_blank"),
                                          dbc.DropdownMenuItem("FIAT profile",
                                                               href="https://www.instagram.com/fiat",
                                                               target="_blank"),
                                          dbc.DropdownMenuItem("AUDI profile",
                                                               href="https://www.instagram.com/audiofficial",
                                                               target="_blank"),
                                          dbc.DropdownMenuItem("Github repository",
                                                               href="https://github.com/JuttaR/Capstone",
                                                               target="_blank")],
                                                       nav=True, in_navbar=True, label="More info",)],
                                      className="ml-auto", navbar=True),
                                      id="navbar-collapse", navbar=True,), ]),
                    color="primary", dark=True, className="navbar navbar-expand-lg navbar-dark bg-primary")

# Create intro card
card = dbc.Card([dbc.CardHeader("Udacity Nanodegree in Data Science - Capstone Project"),
                dbc.CardBody([
                                html.H4("About MINI Instagram Dashboard", className="card-title"),
                                html.P("This project is a web tool for marketers at MINI. It takes in Instagram data of"
                                       " the global channels of MINI, FIAT and AUDI, analyses, and visualizes the data"
                                       " in various charts. These help the global Brand Management team at MINI to"
                                       " monitor developments on Instagram. Please check out my Github repo for"
                                       " details.",
                                       className="card-text"),
                                dbc.Button("Explore Github repository", color="primary",
                                           href='https://github.com/JuttaR/Capstone', target='_blank'),
                            ], className='card-body')], className='card border-primary mt-3 mb-3')

# APP LAYOUT
app.layout = html.Div([
                navbar,
                html.Div([
                    dbc.Container([
                        # INTRO CARD
                        dbc.Row(dbc.Col(html.Div(card))),
                        # DATE PICKER
                        dbc.Row(dbc.Col(html.Div([
                            dbc.Card([dbc.CardHeader("Select date range"),
                                dbc.CardBody([
                                    dcc.DatePickerRange(
                                        id='date-picker',
                                        min_date_allowed=date(2021, 1, 1),
                                        max_date_allowed=date(2021, 6, 30),
                                        start_date=date(2021, 1, 1),
                                        end_date=date(2021, 6, 15),
                                        clearable=True,
                                        reopen_calendar_on_clear=True,
                                        first_day_of_week=1,
                                        display_format='YYYY-MM-DD',
                                        persistence=True,
                                        persisted_props=['start_date', 'end_date'],
                                        persistence_type='session',
                                        updatemode='singledate'
                                    ), dcc.Markdown(id='date-output')
                                ], className='card-body')], className='card border-primary mb-3')]))),
                        # LIKES GRAPH
                        dbc.Row([dbc.Col(html.Div(
                            dbc.Card([dbc.CardHeader("Likes attributed to posting day"),
                                dbc.CardBody([
                                    dcc.Dropdown(id="profile-picker-likes",
                                                 options=profile_options,
                                                 value=['mini', 'fiat', 'audiofficial'],
                                                 multi=True,
                                                 className='custom-dropdown'),
                                    dcc.Graph(id='graph-line-likes'),
                                ], className='card-body')], className='card border-primary mb-3')
                            )),
                            # COMMENTS & SENTIMENT GRAPH
                            dbc.Col(html.Div(
                                    dbc.Card([dbc.CardHeader("Comments & sentiment attributed to posting day"),
                                              dbc.CardBody([
                                               dcc.Dropdown(id="profile-picker-sentiment",
                                                            options=profile_options,
                                                            value=['mini', 'fiat'],
                                                            multi=True,
                                                            className='custom-dropdown'),
                                               dcc.Graph(id='graph-line-sentiment'),
                                                ], className='card-body')], className='card border-primary mb-3')
                                    )),
                        ]),
                        # POSTING INSIGHTS
                        dbc.Row([dbc.Col(html.Div(
                            dbc.Card([dbc.CardHeader("Posting insights"),
                                      dbc.CardBody([
                                        dcc.Dropdown(id="profile-picker-insights",
                                                     options=profile_options,
                                                     value=['mini', 'fiat', 'audiofficial'],
                                                     multi=True,
                                                     className='custom-dropdown'),
                                        # 1st ROW OF POSTING INSIGHTS CHARTS
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Current followers"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="current-followers")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Total posts"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="total-posts")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Total likes"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="total-likes")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Total comments"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="total-comments"),
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                                ]),
                                        # 2nd ROW OF POSTING INSIGHTS CHARTS
                                        dbc.Row([
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Sentiment distribution"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="avg-sentiment")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Avg. posts / week"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="avg-posts")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Avg. likes per post"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="avg-likes")
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                            dbc.Col(
                                                dbc.Card([dbc.CardHeader("Avg. comments per post"),
                                                          dbc.CardBody([
                                                            dcc.Graph(id="avg-comments"),
                                                            ], className="card-body px-0 mx-0",
                                                              style={'padding': '0 0'})], className="mt-3"), width=3),
                                                ])
                                      ], className='card-body')], className='card border-primary mb-3')
                            ))]),
                        # TOP POST
                        dbc.Row([dbc.Col(html.Div(
                            dbc.Card([dbc.CardHeader("Best-performing posts by likes and by comments"),
                                      dbc.CardBody([
                                        dcc.Dropdown(id="profile-picker-top",
                                                     options=profile_options,
                                                     value='mini',
                                                     multi=False,
                                                     className='custom-dropdown'),
                                        dcc.Markdown(id='top-posts-likes'),
                                        dcc.Markdown(id='top-posts-comments'),
                                        ], className='card-body')], className='card border-primary mb-3')
                            )),
                            # LOW POST
                            dbc.Col(html.Div(
                                dbc.Card([dbc.CardHeader("Least-performing posts by likes and by comments"),
                                          dbc.CardBody([
                                            dcc.Dropdown(id="profile-picker-bottom",
                                                         options=profile_options,
                                                         value='mini',
                                                         multi=False,
                                                         className='custom-dropdown'),
                                            dcc.Markdown(id='low-posts-likes'),
                                            dcc.Markdown(id='low-posts-comments'),
                                            ], className='card-body')], className='card border-primary mb-3')
                                )),
                                ]),

                        ])
                ])
            ],)

# FUNCTIONS


# DATE PICKER
@app.callback(
    Output('date-output', 'children'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_output(start_date, end_date):
    date_text = f"Displaying data from {start_date} to {end_date}."
    return date_text


# LIKES GRAPH
@app.callback(
    Output('graph-line-likes', 'figure'),
    [Input('profile-picker-likes', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    traces = []
    for profile in value:
        df_posts_profile = df_posts[df_posts['username'] == profile]
        df_posts_profile_day = df_posts_profile.groupby('day')
        df_likes = df_posts_profile_day.likes.sum().to_frame().reset_index().rename(columns={0: 'likes'})
        traces.append(dict(type='scatter', mode='lines+markers', x=df_likes['day'], y=df_likes['likes'],
                           name=f'{profile}', opacity=0.8,))

    return {'data': traces,
            'layout': go.Layout(yaxis=dict(title='Likes'),
                                xaxis=dict(range=[start_date, end_date]),
                                template=mini_template)}


# COMMENTS & SENTIMENT GRAPH
@app.callback(Output('graph-line-sentiment', 'figure'),
              [Input('profile-picker-sentiment', 'value'),
               Input('date-picker', 'start_date'),
               Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    traces = []
    for profile in value:
        df_posts_scored_profile = df_posts_scored[df_posts_scored['username'] == profile]
        df_posts_scored_profile_day = df_posts_scored_profile.groupby('day')
        comments_sum = df_posts_scored_profile_day.comments.sum().to_frame().reset_index()
        total_score = (df_posts_scored_profile_day.positive.sum() + df_posts_scored_profile_day.neutral.sum() \
                       + df_posts_scored_profile_day.negative.sum()).to_frame(name='total_score').reset_index()
        pos_sentiment = df_posts_scored_profile_day.positive.sum().to_frame().reset_index()

        traces.append(dict(type='bar', opacity=0.8, x=comments_sum['day'], y=comments_sum['comments'],
                           name=f'{profile}: Comments',))
        traces.append(dict(type='scatter', mode='markers', x=pos_sentiment['day'],
                           y=pos_sentiment['positive']/total_score['total_score'],
                           yaxis='y2', name=f'{profile}: Sentiment',))

    return {'data': traces,
            'layout': go.Layout(xaxis=dict(range=[start_date, end_date]),
                                yaxis=dict(title='Comments'),
                                yaxis2=dict(range=[0, 1], title='% Positive Sentiment', overlaying='y', side='right'),
                                template=mini_template)}


# POSTING INSIGHTS
# 1st ROW
# CURRENT FOLLOWERS
@app.callback(
    Output('current-followers', 'figure'),
    [Input('profile-picker-insights', 'value')])
def update_figure(value):
    x = []
    y = []
    for profile in value:
        df_followers_profile = df_followers[df_followers['username'] == profile]
        x.append(df_followers_profile['followers'].values[0])
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# TOTAL POSTS
@app.callback(
    Output('total-posts', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append(len(df_posts_profile))
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# TOTAL LIKES
@app.callback(
    Output('total-likes', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append(df_posts_profile.likes.sum())
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# TOTAL COMMENTS
@app.callback(
    Output('total-comments', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append(df_posts_profile.comments.sum())
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# 2nd ROW
# SENTIMENT DISTRIBUTION
@app.callback(
    Output('avg-sentiment', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    pos = []
    neu = []
    neg = []
    for profile in value:
        df_posts_scored_time = df_posts_scored[
            (df_posts_scored['taken_at'] > pd.to_datetime(start_date)) & (df_posts_scored['taken_at'] <
                                                                          pd.to_datetime(end_date))]
        df_posts_scored_profile = df_posts_scored_time[df_posts_scored_time['username'] == profile]

        total_score = df_posts_scored_profile.positive.sum() + df_posts_scored_profile.neutral.sum() +\
                      df_posts_scored_profile.negative.sum()

        pos_sent = df_posts_scored_profile.positive.sum()
        neu_sent = df_posts_scored_profile.neutral.sum()
        neg_sent = df_posts_scored_profile.negative.sum()

        pos.append(round((pos_sent/total_score), 3))
        neu.append(round((neu_sent/total_score), 3))
        neg.append(round((neg_sent/total_score), 3))

    traces = dict(type='bar', opacity=0.8, x=pos, y=value, name='positive', orientation='h',
                  marker=dict(color='rgba(1, 152, 117, 1)')), \
             dict(type='bar', opacity=0.8, x=neu, y=value, name='neutral', orientation='h',
                  marker=dict(color='rgba(189, 195, 199, 1)')), \
             dict(type='bar', opacity=0.8, x=neg, y=value, name='negative', orientation='h',
                  marker=dict(color='rgba(246, 71, 71, 1)'))

    return {'data': traces,
            'layout': go.Layout(height=300, template=mini_template, barmode='stack')}


# AVG POSTS
@app.callback(
    Output('avg-posts', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    analysed_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append((len(df_posts_profile))/analysed_days * 7)
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# AVG. LIKES
@app.callback(
    Output('avg-likes', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append((df_posts_profile.likes.sum())/(df_posts_profile.shortcode.nunique()))
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# AVG. COMMENTS
@app.callback(
    Output('avg-comments', 'figure'),
    [Input('profile-picker-insights', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_figure(value, start_date, end_date):
    x = []
    y = []
    for profile in value:
        df_posts_time = df_posts[
            (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
        df_posts_profile = df_posts_time[df_posts_time['username'] == profile]
        x.append((df_posts_profile.comments.sum())/(df_posts_profile.shortcode.nunique()))
        y.append(profile)

    fig = go.Bar(x=x, y=y, orientation='h')

    return go.Figure(data=[fig],
                     layout=go.Layout(height=300, template=mini_template))


# TOP POSTS
@app.callback(
    Output('top-posts-likes', 'children'),
    [Input('profile-picker-top', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_post(value, start_date, end_date):
    df_posts_time = df_posts[(df_posts['taken_at']
                              > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
    df_posts_profile = df_posts_time[df_posts_time['username'] == value]
    top_likes_row = df_posts_profile[df_posts_profile['likes'] == df_posts_profile['likes'].max()]
    top_likes = f"**Number of likes: {top_likes_row.iloc[0]['likes']}**  "
    top_caption = f"**Caption:** {top_likes_row.iloc[0]['caption']}  "
    top_date = f"Posted at: {top_likes_row.iloc[0]['taken_at']}  "
    top_link = f"[View post on Instagram](https://www.instagram.com/p/{top_likes_row.iloc[0]['shortcode'].strip()})  "
    sep = f"  ___  "
    return top_likes, top_caption, top_date, top_link, sep


@app.callback(
    Output('top-posts-comments', 'children'),
    [Input('profile-picker-top', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_post(value, start_date, end_date):
    df_posts_time = df_posts[
        (df_posts['taken_at'] > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
    df_posts_profile = df_posts_time[df_posts_time['username'] == value]
    top_comments_row = df_posts_profile[df_posts_profile['comments'] == df_posts_profile['comments'].max()]
    top_comments = f"**Number of comments: {top_comments_row.iloc[0]['comments']}**  "
    top_caption = f"**Caption:** {top_comments_row.iloc[0]['caption']}  "
    top_date = f"Posted at: {top_comments_row.iloc[0]['taken_at']}  "
    top_link = f"[View post on Instagram](https://www.instagram.com/p/" \
               f"{top_comments_row.iloc[0]['shortcode'].strip()})  "
    return top_comments, top_caption, top_date, top_link


# LOW POSTS
@app.callback(
    Output('low-posts-likes', 'children'),
    [Input('profile-picker-bottom', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_post(value, start_date, end_date):
    df_posts_time = df_posts[(df_posts['taken_at']
                              > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
    df_posts_profile = df_posts_time[df_posts_time['username'] == value]
    low_likes_row = df_posts_profile[df_posts_profile['likes'] == df_posts_profile['likes'].min()]
    low_likes = f"**Number of likes: {low_likes_row.iloc[0]['likes']}**  "
    low_caption = f"**Caption:** {low_likes_row.iloc[0]['caption']}  "
    low_date = f"Posted at: {low_likes_row.iloc[0]['taken_at']}  "
    low_link = f"[View post on Instagram](https://www.instagram.com/p/{low_likes_row.iloc[0]['shortcode'].strip()})  "
    sep = f"  ___  "
    return low_likes, low_caption, low_date, low_link, sep


@app.callback(
    Output('low-posts-comments', 'children'),
    [Input('profile-picker-bottom', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')])
def update_post(value, start_date, end_date):
    df_posts_time = df_posts[
        (df_posts['taken_at']
         > pd.to_datetime(start_date)) & (df_posts['taken_at'] < pd.to_datetime(end_date))]
    df_posts_profile = df_posts_time[df_posts_time['username'] == value]
    low_comments_row = df_posts_profile[df_posts_profile['comments'] == df_posts_profile['comments'].min()]
    low_comments = f"**Number of comments: {low_comments_row.iloc[0]['comments']}**  "
    low_caption = f"**Caption:** {low_comments_row.iloc[0]['caption']}  "
    low_date = f"Posted at: {low_comments_row.iloc[0]['taken_at']}  "
    low_link = f"[View post on Instagram](https://www.instagram.com/p/" \
               f"{low_comments_row.iloc[0]['shortcode'].strip()})  "
    return low_comments, low_caption, low_date, low_link


# RESPONSIVE NAVBAR
@app.callback(Output(f"navbar-collapse", "is_open"),
              [Input(f"navbar-toggler", "n_clicks")],
              [State(f"navbar-collapse", "is_open")])
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# RUN APP
if __name__ == '__main__':
    app.run_server()
