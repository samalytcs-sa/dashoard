import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
import io
import re
import os
from collections import Counter
import tempfile
from datetime import datetime
import urllib.request

# Google API Configuration
GEMINI_API_KEY = "AIzaSyBJIfFfVWjeJqNcwc4Z1_gt01IrgBOmZvE"
os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY

# Initialize Google Gemini AI for enhanced sentiment analysis
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    AI_AVAILABLE = True
    print("Google Gemini AI initialized successfully")
except ImportError:
    AI_AVAILABLE = False
    gemini_model = None
    print("Google Gemini AI not available - install google-generativeai package")
except Exception as e:
    AI_AVAILABLE = False
    gemini_model = None
    print(f"Error initializing Gemini AI: {e}")

# Skip local sentiment model initialization
print("Using Gemini AI for all text analysis")
sentiment_model = None

# Arabic spam keywords for filtering
spam_keywords = [
    'اربح', 'مال', 'مجاني', 'اتصل الآن', 'عرض خاص',
    'خصم', 'مليون', 'دولار', 'ريال', 'فوز', 'جائزة', 'صدقة',
    'سريع', 'فوري', 'حصري', 'محدود', 'الآن فقط',
    'جنس', 'حب', 'ممارسة', 'علاقة', 'مداعبة', 'حميمية',
    'رغبة', 'إثارة', 'مثير', 'قضيب', 'مهبل', 'صورة إباحية',
    'فيلم إباحي', 'مباشر', 'عري', 'لقب', 'لعب', 'مثير جنسيًا',
    'فضيحة', 'خيانة', 'ممارسة جنسية', 'شهوة', 'مغتصب', 'عشيق',
    'عشيقة', 'مثير', 'نزوة', 'شرج', 'جماع', 'نكاح',
    'عريانة', 'عشيقة', 'عضو تناسلي', 'قذف', 'مثير', 'استمناء',
    'قسيمة', 'الفحولة', 'جيجل', 'المسيار', 'مسيار',
    'الرابط', 'اتصل', 'فاتورتها', 'اجر', 'سداد', 'بروفايلك'
]

def show_number_on_card(value, label):
    """إنشاء بطاقة Bootstrap لعرض الإحصائيات"""
    colors = ['primary', 'success', 'info', 'warning', 'danger']
    color_index = hash(str(label)) % len(colors)
    selected_color = colors[color_index]
    
    icon = '📊'
    if 'tweet' in label.lower() or 'minute' in label.lower():
        icon = '⚡'
    elif 'like' in label.lower():
        icon = '❤️'
    elif 'retweet' in label.lower():
        icon = '🔄'
    elif 'user' in label.lower():
        icon = '👥'
    elif 'reply' in label.lower():
        icon = '💬'
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, style={'fontSize': '2.5em', 'marginBottom': '10px'}),
                html.H2(str(value), style={
                     'margin': '0', 
                     'fontSize': '2.2em', 
                     'fontWeight': 'bold',
                     'color': 'white'
                 }),
                 html.P(label, style={
                     'margin': '8px 0 0 0', 
                     'fontSize': '1em', 
                     'opacity': '0.95', 
                     'fontWeight': '500',
                     'color': 'white'
                 })
            ], style={'textAlign': 'center'})
        ])
    ], color=selected_color, style={
        'minWidth': '180px',
        'margin': '10px',
        'boxShadow': '0 8px 25px rgba(0, 0, 0, 0.2)',
        'transition': 'all 0.3s ease'
    })

def is_spam(text):
    """Check if a text contains spam keywords"""
    if not text or pd.isna(text):
        return False
    text_lower = str(text).lower()
    return any(keyword.lower() in text_lower for keyword in spam_keywords)

def filter_spam_tweets(df):
    """Remove tweets/posts that contain spam keywords"""
    if df.empty:
        return df
    
    text_col = None
    for col in ['Message', 'text', 'content', 'tweet', 'post']:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        return df
    
    original_count = len(df)
    df_filtered = df[~df[text_col].apply(is_spam)].copy()
    filtered_count = len(df_filtered)
    
    print(f"Spam filtering: Removed {original_count - filtered_count} spam tweets out of {original_count} total tweets")
    return df_filtered

def parse_contents(contents, filename):
    """Parses the uploaded file content and returns a pandas DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif re.search(r'\.xls(x)?$', filename):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "File type not supported. Please upload a CSV or Excel file."
        
        df = filter_spam_tweets(df)
        
    except Exception as e:
        return None, f"There was an error processing this file: {e}"

    return df, None

def analyze_sentiment_with_gemini(texts, batch_size=3):
    """Analyze sentiment using Google Gemini AI"""
    if not AI_AVAILABLE or not gemini_model:
        return ['neutral'] * len(texts)
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        try:
            prompt = """أنت محلل مشاعر متخصص في النصوص العربية ووسائل التواصل الاجتماعي. حلل المشاعر في النصوص التالية بدقة عالية:

القواعد:
- إيجابي (positive): الفرح، الحب، الإعجاب، التشجيع، التهنئة
- سلبي (negative): الغضب، الحزن، الانتقاد، الشكوى، المطالبة بالتغيير، الاستغاثة
- محايد (neutral): المعلومات العادية، الأسئلة، الأخبار، النصوص الدينية

النصوص للتحليل:
"""
            
            for idx, text in enumerate(batch):
                if isinstance(text, str) and len(text.strip()) >= 3:
                    clean_text = text.strip()[:200]
                    prompt += f"النص {idx+1}: \"{clean_text}\"\n"
                else:
                    prompt += f"النص {idx+1}: \"[نص فارغ]\"\n"
            
            prompt += "\nأرجع النتيجة بصيغة JSON فقط: [{\"sentiment\": \"positive\"}, {\"sentiment\": \"negative\"}, {\"sentiment\": \"neutral\"}]"
            
            response = gemini_model.generate_content(prompt)
            
            import json
            try:
                response_text = response.text.strip()
                if '```json' in response_text:
                    response_text = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    response_text = response_text.split('```')[1].strip()
                
                ai_results = json.loads(response_text)
                for result in ai_results:
                    sentiment = result.get('sentiment', 'neutral')
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'
                    results.append(sentiment)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"JSON parsing error: {e}")
                results.extend(['neutral'] * len(batch))
                
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            results.extend(['neutral'] * len(batch))
    
    return results

def generate_ai_summary(df):
    """Generate an intelligent summary of the social media data"""
    if not AI_AVAILABLE or not gemini_model or df.empty:
        return "لا توجد بيانات متاحة للملخص أو خدمة الذكاء الاصطناعي غير متاحة."
    
    try:
        total_posts = len(df)
        
        message_col = None
        for col in ['Message']:
            if col in df.columns:
                message_col = col
                break
        
        if not message_col:
            return "لم يتم العثور على محتوى نصي في البيانات لإنشاء الملخص."
        
        sample_messages = df[message_col].dropna().head(10).tolist()
        sample_messages = [str(msg)[:200] for msg in sample_messages if str(msg).strip()]
        
        if not sample_messages:
            return "لم يتم العثور على محتوى نصي صالح للتحليل."
        
        prompt = f"""قم بتحليل مجموعة بيانات وسائل التواصل الاجتماعي وقدم ملخصاً شاملاً:

نظرة عامة:
- إجمالي المنشورات: {total_posts}
- عينة من الرسائل:
"""
        
        for i, msg in enumerate(sample_messages[:5], 1):
            prompt += f"\n{i}. {msg}..."
        
        prompt += """\n\nيرجى تقديم ملخص يتضمن:
1. المشاعر العامة
2. المواضيع الرئيسية
3. رؤى مهمة
4. الأنماط الملحوظة
5. توصيات

اجعل الملخص مختصراً (200-300 كلمة) باللغة العربية."""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"Error generating AI summary: {e}")
        return f"فشل في إنشاء الملخص. تحتوي مجموعة البيانات على {len(df)} منشور."

def find_text_column(df):
    """Find the main text column in the dataframe"""
    text_columns = ['Message', 'text', 'content', 'tweet', 'post', 'description']
    for col in text_columns:
        if col in df.columns:
            return col
    return None

def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    hashtags = re.findall(r'#\w+', str(text))
    return hashtags

def create_gauge_chart(level):
    """Create a gauge chart showing tweet activity level"""
    plot_bgcolor = 'rgba(192,192,192,.0)'
    quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043"]
    quadrant_text = ["", "<b>Level Three</b>", "<b>Level Two</b>", "<b>Level One</b>", "<b>Very Low</b>"]
    n_quadrants = len(quadrant_colors) - 1

    current_value = level
    min_value = 0
    max_value = 50
    hand_length = np.sqrt(2) / 4
    hand_angle = np.pi * (1 - (max(min_value, min(max_value, current_value)) - min_value) / (max_value - min_value))

    fig = go.Figure(
        data=[
            go.Pie(
                values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
                rotation=90,
                hole=0.5,
                marker_colors=quadrant_colors,
                text=quadrant_text,
                textinfo="text",
                hoverinfo="skip",
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=0, t=10, l=10, r=10),
            width=450,
            height=450,
            paper_bgcolor=plot_bgcolor,
            plot_bgcolor=plot_bgcolor,
            annotations=[
                go.layout.Annotation(
                    text=f"<b>Tweet Activity Level Score</b><br><span style='font-size:24px'>{current_value}</span>",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.25, yanchor="bottom", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="white")
                )
            ],
            shapes=[
                go.layout.Shape(
                    type="circle",
                    x0=0.48, x1=0.52,
                    y0=0.48, y1=0.52,
                    fillcolor="#333",
                    line_color="#333",
                ),
                go.layout.Shape(
                    type="line",
                    x0=0.5, x1=0.5 + hand_length * np.cos(hand_angle),
                    y0=0.5, y1=0.5 + hand_length * np.sin(hand_angle),
                    line=dict(color="#333", width=4)
                )
            ]
        )
    )
    
    return fig

def get_activity_level_and_chart(max_tweets_per_minute):
    """Determine activity level and create gauge chart based on max tweets per minute"""
    if pd.isna(max_tweets_per_minute) or max_tweets_per_minute == 'N/A':
        level_text = "No Data"
        chart_fig = create_gauge_chart(5)
    elif 5 < max_tweets_per_minute < 10:
        level_text = "Level One"
        chart_fig = create_gauge_chart(20)
    elif 10 < max_tweets_per_minute < 20:
        level_text = "Level Two" 
        chart_fig = create_gauge_chart(30)
    elif 20 < max_tweets_per_minute < 400:
        level_text = "Level Three"
        chart_fig = create_gauge_chart(45)
    else:
        level_text = "Very Low Activity"
        chart_fig = create_gauge_chart(5)
    
    return level_text, chart_fig

def create_tweet_generator_pie_chart(df):
    """Create pie chart for Tweet Generator distribution"""
    if df is None or df.empty or 'Tweet Generator' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد بيانات متاحة لمولد التغريدات",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    
    # Count values in Tweet Generator column
    generator_counts = df['Tweet Generator'].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=generator_counts.values,
        names=generator_counts.index,
        title="Tweet Generator Distribution | توزيع مولدات التغريدات"
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(
            font=dict(size=16, color='white'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_media_type_pie_chart(df):
    """Create pie chart for MediaTypeList distribution"""
    if df is None or df.empty or 'MediaTypeList' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد بيانات متاحة لأنواع الوسائط",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    
    # Handle different data formats in MediaTypeList
    media_types = []
    for media_list in df['MediaTypeList'].dropna():
        if isinstance(media_list, str):
            # If it's a string, split by common delimiters
            if ',' in media_list:
                types = [t.strip() for t in media_list.split(',')]
            elif ';' in media_list:
                types = [t.strip() for t in media_list.split(';')]
            elif '|' in media_list:
                types = [t.strip() for t in media_list.split('|')]
            else:
                types = [media_list.strip()]
            media_types.extend(types)
        else:
            # If it's not a string, convert to string
            media_types.append(str(media_list))
    
    if not media_types:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد أنواع وسائط صالحة",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='white')
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig
    
    # Count media types
    from collections import Counter
    media_counts = Counter(media_types)
    
    # Create pie chart
    fig = px.pie(
        values=list(media_counts.values()),
        names=list(media_counts.keys()),
        title="Media Type Distribution | توزيع أنواع الوسائط"
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        title=dict(
            font=dict(size=16, color='white'),
            x=0.5
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )
    
    return fig

def create_hashtag_bar_chart(df):
    """Create a modern bar chart for hashtag distribution"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد بيانات متاحة لتحليل الهاشتاغ",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='black')
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        return fig
    
    all_hashtags = []
    text_col = find_text_column(df)
    
    if text_col and text_col in df.columns:
        for text in df[text_col].dropna():
            hashtags = extract_hashtags(str(text))
            all_hashtags.extend(hashtags)
    
    if not all_hashtags:
        fig = go.Figure()
        fig.add_annotation(
            text="لم يتم العثور على هاشتاغات في البيانات",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color='black')
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        return fig
    
    hashtag_counts = Counter(all_hashtags)
    top_hashtags = dict(hashtag_counts.most_common(15))
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_hashtags.values()),
            y=list(top_hashtags.keys()),
            orientation='h',
            marker=dict(
                color='rgba(29, 78, 216, 0.8)',
                line=dict(color='rgba(29, 78, 216, 1.0)', width=1)
            ),
            text=[f"{val}" for val in top_hashtags.values()],
            textposition='outside',
            textfont=dict(size=12, color='black')
        )
    ])
    
    fig.update_layout(
        title=dict(
            text="أهم الهاشتاغات | Top Hashtags",
            font=dict(size=18, color='black', family='Arial'),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title=dict(text="عدد التكرارات | Count", font=dict(size=14, color='black')),
            tickfont=dict(size=12, color='black'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text="الهاشتاغات | Hashtags", font=dict(size=14, color='black')),
            tickfont=dict(size=12, color='black'),
            gridcolor='rgba(200,200,200,0.3)',
            showgrid=True,
            automargin=True  # This will automatically adjust margins for long labels
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', family='Arial'),
        margin=dict(l=120, r=50, t=60, b=50),  # Increased left margin for hashtag labels
        height=500,
        showlegend=False
    )
    
    return fig

def generate_top_influencers_display(df, top_n=10):
    """Generate display for top influencers"""
    if df.empty or 'influence_score' not in df.columns:
        return html.Div("No influencer data available | لا توجد بيانات المؤثرين", 
                       className="text-center text-muted")
    
    top_influencers = df.nlargest(top_n, 'influence_score')
    
    influencer_cards = []
    for idx, (_, influencer) in enumerate(top_influencers.iterrows(), 1):
        username = influencer.get('SenderScreenName', 'Unknown User')
        followers = influencer.get('Sender Followers Count', 0)
        likes = influencer.get('Favorites', 0)
        influence_score = influencer.get('influence_score', 0)
        message = str(influencer.get('Message', ''))
        if len(message) > 100:
            message = message[:100] + "..."
        
        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"#{idx} {username}", className="card-title text-primary"),
                html.P([
                    html.Strong(f"📊 Influence Score: {influence_score:.3f}"),
                    html.Br(),
                    f"👥 Followers: {followers:,} | 👍 Likes: {likes:,}",
                    html.Br(),
                    html.Small(message, className="text-muted")
                ])
            ])
        ], className="mb-2", style={'fontSize': '12px'})
        
        influencer_cards.append(card)
    
    return influencer_cards

def get_dashboard_figures(df):
    """Generates all Plotly figures from a given DataFrame"""
    global sample_df
    if df.empty:
        return {}, {}, {}, {}, "Uploaded data is empty or invalid."

    try:
        df['CreatedTime'] = pd.to_datetime(df['CreatedTime'])
        
        print(f"Processing {len(df)} total messages")
        
        # SPAM FILTERING
        original_count = len(df)
        df_clean = filter_spam_tweets(df)
        spam_removed = original_count - len(df_clean)
        print(f"Removed {spam_removed} spam messages")
        
        # INFLUENCE CALCULATION
        followers_col = 'Sender Followers Count'
        likes_col = 'Favorites'
        
        if followers_col in df_clean.columns:
            df_clean[followers_col] = pd.to_numeric(df_clean[followers_col], errors='coerce').fillna(0)
        else:
            df_clean[followers_col] = 0
            
        if likes_col in df_clean.columns:
            df_clean[likes_col] = pd.to_numeric(df_clean[likes_col], errors='coerce').fillna(0)
        else:
            df_clean[likes_col] = 0
        
        # Calculate influence scores
        max_followers = df_clean[followers_col].max() if df_clean[followers_col].max() > 0 else 1
        max_likes = df_clean[likes_col].max() if df_clean[likes_col].max() > 0 else 1
        
        normalized_followers = df_clean[followers_col] / max_followers
        normalized_likes = df_clean[likes_col] / max_likes
        
        df_clean['influence_score'] = (normalized_followers * 0.6) + (normalized_likes * 0.4)
        
        # TOP 100 SELECTION
        if len(df_clean) > 100:
            top_influential = df_clean.nlargest(100, 'influence_score')
            print(f"Selected top 100 most influential messages")
        else:
            top_influential = df_clean.copy()
            print(f"Using all {len(top_influential)} messages")
        
        df = df_clean.copy()
        
        # AI SENTIMENT ANALYSIS
        print("Running AI sentiment analysis...")
        messages_for_ai = top_influential['Message'].dropna().astype(str).tolist()
        
        if AI_AVAILABLE and messages_for_ai:
            try:
                print(f"Analyzing {len(messages_for_ai)} messages with Gemini AI...")
                gemini_sentiments = analyze_sentiment_with_gemini(messages_for_ai)
                
                sample_df = top_influential.copy()
                sample_df['sentiment'] = [s.lower() for s in gemini_sentiments[:len(top_influential)]]
                
                print(f"Sentiment analysis completed for {len(gemini_sentiments)} messages")
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                sample_df = pd.DataFrame()
        else:
            sample_df = pd.DataFrame()

        # Create sentiment figure
        if not sample_df.empty and 'sentiment' in sample_df.columns:
            sentiment_counts = sample_df['sentiment'].value_counts()
            sentiment_fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Sentiment Distribution (Sample Analysis)'
            )
        else:
            sentiment_fig = px.pie(
                values=[1],
                names=['No sentiment data available'],
                title='Sentiment Distribution (No Analysis)'
            )

        # Tweets per Minute Timeline
        df_timeline = df.set_index('CreatedTime').resample('min').size().reset_index(name='count')
        timeline_fig = px.line(
            df_timeline,
            x='CreatedTime',
            y='count',
            title='Tweets per Minute'
        )

        # Top Countries Bar Chart
        if 'Country' in df.columns:
            country_counts = df['Country'].value_counts().head(10)
            country_fig = px.bar(
                x=country_counts.index,
                y=country_counts.values,
                title='Top 10 Countries by Tweet Count'
            )
        else:
            country_fig = {}

        # Users by Followers Bar Chart
        if 'SenderListedName' in df.columns and 'Sender Followers Count' in df.columns:
            df_users = df.groupby('SenderListedName').agg({
                'Sender Followers Count': 'max',
                'Message': 'count'
            }).reset_index()
            df_users = df_users.sort_values(by='Sender Followers Count', ascending=False).head(10)
            followers_fig = px.bar(
                df_users,
                x='SenderListedName',
                y='Sender Followers Count',
                title='Top 10 Users by Follower Count'
            )
        else:
            followers_fig = {}

        summary_text = "Dashboard loaded successfully. Upload a file to see detailed analysis."

        return sentiment_fig, timeline_fig, country_fig, followers_fig, summary_text

    except Exception as e:
        print(f"Error in get_dashboard_figures: {e}")
        return {}, {}, {}, {}, f"Error generating charts: {e}"

# Global variables for caching
cached_ai_results = {
    'sentiments': [],
    'emotions': [],
    'texts': [],
    'negative_tweets': []
}

# Create the Dash app instance
custom_css = {
    'external_stylesheets': [dbc.themes.BOOTSTRAP],
    'suppress_callback_exceptions': True
}

app = dash.Dash(__name__, **custom_css)

# Global variables
current_dataframe = None
sample_df = pd.DataFrame()

# Add custom CSS styles with original Arabic UI
app.index_string = '''
<!DOCTYPE html>
<html dir="rtl" lang="ar">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;700&family=Cairo:wght@400;700&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Cairo', 'Noto Sans Arabic', 'Arial', sans-serif !important;
            }
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                direction: rtl;
                text-align: right;
            }
            .container-fluid {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                margin: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            }
            .card {
                border: none;
                border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            }
            .card-header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                border-radius: 15px 15px 0 0 !important;
                border: none;
                padding: 15px 20px;
                font-weight: bold;
            }
            .card-header h4 {
                margin: 0;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
            }
            .upload-area {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                border: 2px dashed #4facfe;
                border-radius: 10px;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                background: linear-gradient(135deg, #fed6e3 0%, #a8edea 100%);
                border-color: #00f2fe;
                transform: scale(1.02);
            }
            h1 {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: bold;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
            }
            .statistics-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin: 10px;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
                transition: all 0.3s ease;
            }
            .statistics-card:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
            }
            .hr-custom {
                border: none;
                height: 3px;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                border-radius: 2px;
                margin: 30px 0;
            }
            .sentiment-positive {
                background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
            }
            .sentiment-negative {
                background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
            }
            .sentiment-neutral {
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
            }
            .ai-summary {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border: 2px solid #dee2e6;
                border-radius: 12px;
                padding: 25px;
                direction: rtl;
                text-align: right;
                font-family: 'Cairo', 'Noto Sans Arabic', 'Arial', sans-serif;
                color: #2c3e50;
                font-size: 18px;
                line-height: 1.8;
                box-shadow: inset 0 2px 8px rgba(0,0,0,0.1);
                min-height: 150px;
                transition: all 0.3s ease;
            }
            .arabic-text {
                direction: rtl;
                text-align: right;
                font-family: 'Cairo', 'Noto Sans Arabic', 'Arial', sans-serif;
                unicode-bidi: bidi-override;
            }
            .card-body {
                direction: rtl;
                text-align: right;
            }
            .container-fluid {
                direction: rtl;
            }
            .plotly-graph-div {
                direction: ltr;
            }
            .ai-summary:hover {
                background: linear-gradient(135deg, #e9ecef 0%, #f8f9fa 100%);
                box-shadow: inset 0 4px 12px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout with original Arabic UI
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("📱 لوحة تحليل وسائل التواصل الاجتماعي 📊", className="text-center my-4 arabic-text"), width=12)
    ),

    dbc.Row([
        dbc.Col([
            dbc.Alert([
                dbc.Spinner(size="sm", color="primary"),
                " جاري تحليل البيانات... يرجى الانتظار | Analysis in progress... Please wait"
            ], id="progress-alert", color="info", is_open=False, className="text-center")
        ], width=12)
    ]),

    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("📁 رفع ملف البيانات | Upload Your Data File", style={'color': 'white'}, className="arabic-text")),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            '🚀 اسحب وأفلت أو ',
                            html.A('اختر ملف | Select a File', style={'color': '#4facfe', 'fontWeight': 'bold'}, className="arabic-text")
                        ], className="arabic-text"),
                        className='upload-area',
                        style={
                            'width': '100%',
                            'height': '80px',
                            'lineHeight': '80px',
                            'textAlign': 'center',
                            'margin': '10px',
                            'fontSize': '18px',
                            'fontWeight': 'bold'
                        },
                        multiple=False
                    ),
                    html.Div(id='output-data-upload', className="mt-2"),
                ])
            ]),
            width=12
        )
    ),
    
    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(id='max-tweet-per-minute-value', className="card-title"),
                    html.P("Max Tweets/Minute", className="card-text")
                ]),
            ),
            md=2
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(id='likes-value', className="card-title"),
                    html.P("Total Likes", className="card-text")
                ]),
            ),
            md=2
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(id='retweets-value', className="card-title"),
                    html.P("Total Retweets", className="card-text")
                ]),
            ),
            md=2
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(id='replies-value', className="card-title"),
                    html.P("Total Replies", className="card-text")
                ]),
            ),
            md=2
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3(id='unique-users-value', className="card-title"),
                    html.P("Unique Users", className="card-text")
                ]),
            ),
            md=2
        ),
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col(html.H4("👑 Top Influencers | أهم المؤثرين", style={'color': 'white'}), width=8),
                        dbc.Col([
                            dbc.Button(
                                "📥 Download CSV | تحميل CSV",
                                id="download-influencers-btn",
                                color="success",
                                size="sm",
                                className="float-right",
                                style={'marginTop': '-5px'}
                            ),
                            dcc.Download(id="download-influencers")
                        ], width=4)
                    ])
                ]),
                dbc.CardBody(html.Div(id='top-influencers-list', style={
                    'maxHeight': '400px',
                    'overflowY': 'auto',
                    'fontSize': '14px',
                    'direction': 'rtl',
                    'textAlign': 'right'
                }))
            ]),
            md=12
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("😊 Sentiment Analysis", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='sentiment-pie-chart'))
            ]),
            md=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("⏰ Tweets per Minute Timeline", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='timeline-chart'))
            ]),
            md=6
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("🌍 Top Countries by Tweets", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='country-bar-chart'))
            ]),
            md=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("👥 Users by Follower Count", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='followers-bar-chart'))
            ]),
            md=6
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    # Add Tweet Generator and Media Type pie charts
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("📱 Tweet Generator Distribution | توزيع مولدات التغريدات", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='tweet-generator-pie-chart'))
            ]),
            md=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("🎬 Media Type Distribution | توزيع أنواع الوسائط", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='media-type-pie-chart'))
            ]),
            md=6
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    # Add Tweet Activity Level Gauge Chart
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("📊 Tweet Activity Level | مستوى نشاط التغريدات", style={'color': 'white'})),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Graph(id='activity-gauge-chart')
                        ], md=8),
                        dbc.Col([
                            html.Div(id='activity-level-text', style={
                                'fontSize': '24px',
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                                'marginTop': '50px',
                                'padding': '20px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '10px',
                                'border': '2px solid #dee2e6'
                            })
                        ], md=4)
                    ])
                ])
            ]),
            md=12
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("#️⃣ Top Hashtags Distribution", style={'color': 'white'})),
                dbc.CardBody(dcc.Graph(id='hashtag-bar-chart'))
            ]),
            md=12
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("🤖 AI-Generated News Feed Summary", style={'color': 'white'})),
                dbc.CardBody(html.Div(id='summary-text', style={
                    'whiteSpace': 'pre-line', 
                    'fontSize': '18px', 
                    'lineHeight': '1.8',
                    'direction': 'rtl',
                    'textAlign': 'right',
                    'fontFamily': 'Arial, sans-serif',
                    'color': '#2c3e50',
                    'background': 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                    'padding': '25px',
                    'borderRadius': '12px',
                    'border': '2px solid #dee2e6',
                    'boxShadow': 'inset 0 2px 8px rgba(0,0,0,0.1)',
                    'minHeight': '150px'
                }))
            ]),
            md=12
        )
    ], className="mb-4"),

    html.Hr(className='hr-custom'),

    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("😡 Top 10 Negative Tweets (by Followers & Likes)", style={'color': 'white'})),
                dbc.CardBody(html.Div(id='top-negative-tweets', style={
                    'maxHeight': '400px',
                    'overflowY': 'auto',
                    'fontSize': '14px',
                    'direction': 'rtl',
                    'textAlign': 'right'
                }))
            ]),
            md=6
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(html.H4("😊 Top 10 Positive Tweets (by Followers & Likes)", style={'color': 'white'})),
                dbc.CardBody(html.Div(id='top-positive-tweets', style={
                    'maxHeight': '400px',
                    'overflowY': 'auto',
                    'fontSize': '14px',
                    'direction': 'rtl',
                    'textAlign': 'right'
                }))
            ]),
            md=6
        )
    ])
])

# Main callback
@app.callback(
    [Output('progress-alert', 'is_open'),
     Output('output-data-upload', 'children'),
     Output('sentiment-pie-chart', 'figure'),
     Output('timeline-chart', 'figure'),
     Output('country-bar-chart', 'figure'),
     Output('followers-bar-chart', 'figure'),
     Output('hashtag-bar-chart', 'figure'),
     Output('tweet-generator-pie-chart', 'figure'),
     Output('media-type-pie-chart', 'figure'),
     Output('activity-gauge-chart', 'figure'),
     Output('activity-level-text', 'children'),
     Output('summary-text', 'children'),
     Output('max-tweet-per-minute-value', 'children'),
     Output('likes-value', 'children'),
     Output('retweets-value', 'children'),
     Output('replies-value', 'children'),
     Output('unique-users-value', 'children'),
     Output('top-negative-tweets', 'children'),
     Output('top-positive-tweets', 'children'),
     Output('top-influencers-list', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename')]
)
def update_dashboard_from_upload(contents, filename):
    global current_dataframe
    
    # Create empty figures for when no data is available
    empty_fig = {'data': [], 'layout': {'title': 'No data available'}}
    
    if contents is None:
        current_dataframe = None
        return (
            False,  # progress alert closed
            html.Div('Please upload a file to view the dashboard.'),
            empty_fig,  # sentiment-pie-chart
            empty_fig,  # timeline-chart
            empty_fig,  # country-bar-chart
            empty_fig,  # followers-bar-chart
            empty_fig,  # hashtag-bar-chart
            empty_fig,  # tweet-generator-pie-chart
            empty_fig,  # media-type-pie-chart
            create_gauge_chart(5),  # activity-gauge-chart
            'No Data',  # activity-level-text
            'AI-generated summary will appear here after a file is uploaded.',  # summary-text
            'N/A',  # max-tweet-per-minute-value
            'N/A',  # likes-value
            'N/A',  # retweets-value
            'N/A',  # replies-value
            'N/A',  # unique-users-value
            html.Div('Upload a file to see top negative tweets.'),  # top-negative-tweets
            html.Div('Upload a file to see top positive tweets.'),   # top-positive-tweets
            html.Div('Upload a file to see top influencers.')  # top-influencers-list
        )

    df, error_message = parse_contents(contents, filename)
    if error_message:
        current_dataframe = None
        return (
            False,  # progress alert closed
            html.Div(error_message, className="text-danger"),
            empty_fig,  # sentiment-pie-chart
            empty_fig,  # timeline-chart
            empty_fig,  # country-bar-chart
            empty_fig,  # followers-bar-chart
            empty_fig,  # hashtag-bar-chart
            empty_fig,  # tweet-generator-pie-chart
            empty_fig,  # media-type-pie-chart
            create_gauge_chart(5),  # activity-gauge-chart
            'Error',  # activity-level-text
            "Error: Could not process the uploaded file.",  # summary-text
            'N/A',  # max-tweet-per-minute-value
            'N/A',  # likes-value
            'N/A',  # retweets-value
            'N/A',  # replies-value
            'N/A',  # unique-users-value
            html.Div('Error processing file.'),  # top-negative-tweets
            html.Div('Error processing file.'),   # top-positive-tweets
            html.Div('Error processing file.')  # top-influencers-list
        )
    
    try:
        # Generate all figures from the parsed DataFrame
        sentiment_fig, timeline_fig, country_fig, followers_fig, _ = get_dashboard_figures(df)
        
        # Generate hashtag chart
        hashtag_fig = create_hashtag_bar_chart(df)
        
        # Generate Tweet Generator and Media Type pie charts
        tweet_generator_fig = create_tweet_generator_pie_chart(df)
        media_type_fig = create_media_type_pie_chart(df)
        
        # Calculate summary statistics
        if 'CreatedTime' in df.columns:
            df['CreatedTime'] = pd.to_datetime(df['CreatedTime'])
            max_tweet_per_minute = df.groupby(pd.Grouper(key='CreatedTime', freq='1Min')).size().max()
        elif 'Action Time' in df.columns:
            df['Action Time'] = pd.to_datetime(df['Action Time'])
            max_tweet_per_minute = df.groupby(pd.Grouper(key='Action Time', freq='1Min')).size().max()
        else:
            max_tweet_per_minute = 'N/A'

        # Create activity level gauge chart
        activity_level_text, activity_gauge_fig = get_activity_level_and_chart(max_tweet_per_minute)

        likes = df['Favorites'].sum() if 'Favorites' in df.columns else 'N/A'
        retweets = df['Retweets'].sum() if 'Retweets' in df.columns else 'N/A'
        
        if 'Message' in df.columns:
            replies = df['Message'].astype(str).str.contains(r'(^RE:|@\w+)', case=False, na=False).sum()
        else:
            replies = 'N/A'
        unique_users = df['SenderScreenName'].nunique() if 'SenderScreenName' in df.columns else 'N/A'
    
        # Generate AI-powered summary
        ai_summary = generate_ai_summary(df)
        
        # Generate top tweets by sentiment
        def get_top_tweets_by_sentiment(df, sentiment_type, top_n=10):
            """Get top tweets by sentiment based on followers and likes combined score"""
            sentiment_col = None
            if 'sentiment' in df.columns:
                sentiment_col = 'sentiment'
            elif 'Sentiment' in df.columns:
                sentiment_col = 'Sentiment'
            
            if sentiment_col is None:
                return html.Div(f'No {sentiment_type} tweets found.')
            
            sentiment_value = sentiment_type.lower()
            filtered_df = df[df[sentiment_col] == sentiment_value].copy()
            
            if filtered_df.empty:
                return html.Div(f'No {sentiment_type} tweets found.')
            
            # Apply spam filtering
            if 'Message' in filtered_df.columns:
                original_count = len(filtered_df)
                filtered_df = filtered_df[~filtered_df['Message'].apply(is_spam)].copy()
                filtered_count = len(filtered_df)
                print(f"Spam filtering for {sentiment_type}: Removed {original_count - filtered_count} spam tweets")
            
            if filtered_df.empty:
                return html.Div(f'No non-spam {sentiment_type} tweets found.')
            
            # Calculate influence scores
            followers_col = 'Sender Followers Count'
            likes_col = 'Favorites'
            
            if followers_col in filtered_df.columns:
                filtered_df[followers_col] = pd.to_numeric(filtered_df[followers_col], errors='coerce').fillna(0)
            else:
                filtered_df[followers_col] = 0
                
            if likes_col in filtered_df.columns:
                filtered_df[likes_col] = pd.to_numeric(filtered_df[likes_col], errors='coerce').fillna(0)
            else:
                filtered_df[likes_col] = 0
            
            # Create weighted scoring system
            max_followers = filtered_df[followers_col].max() if filtered_df[followers_col].max() > 0 else 1
            max_likes = filtered_df[likes_col].max() if filtered_df[likes_col].max() > 0 else 1
            
            normalized_followers = filtered_df[followers_col] / max_followers
            normalized_likes = filtered_df[likes_col] / max_likes
            
            filtered_df['influence_score'] = (normalized_followers * 0.6) + (normalized_likes * 0.4)
            
            # Sort by influence score and get top N
            top_tweets = filtered_df.nlargest(top_n, 'influence_score')
            
            # Create display elements
            tweet_elements = []
            for idx, (_, tweet) in enumerate(top_tweets.iterrows(), 1):
                message_text = tweet.get('Message', 'No message available')
                if pd.isna(message_text):
                    message_text = 'No message available'
                
                if len(str(message_text)) > 200:
                    message_text = str(message_text)[:200] + "..."
                
                followers = int(tweet.get(followers_col, 0))
                likes = int(tweet.get(likes_col, 0))
                retweets = int(tweet.get('Retweets', 0))
                influence_score = tweet.get('influence_score', 0)
                username = tweet.get('SenderScreenName', 'Unknown')
                permalink = tweet.get('Permalink', '')
                
                # Create username element - either clickable link or plain text
                if permalink and not pd.isna(permalink) and str(permalink).strip():
                    username_element = html.A(
                        f"{idx}. @{username}",
                        href=str(permalink).strip(),
                        target="_blank",
                        style={
                            'color': '#4facfe', 
                            'marginBottom': '5px',
                            'textDecoration': 'none',
                            'fontWeight': 'bold'
                        }
                    )
                else:
                    username_element = html.Span(
                        f"{idx}. @{username}",
                        style={
                            'color': '#4facfe', 
                            'marginBottom': '5px',
                            'fontWeight': 'bold'
                        }
                    )
                
                tweet_card = html.Div([
                    html.H6(username_element),
                    html.P(str(message_text),
                          style={'marginBottom': '8px', 'fontSize': '13px'}),
                    html.Small([
                        f"👥 {followers:,} followers | ",
                        f"❤️ {likes} likes | ",
                        f"🔄 {retweets} retweets | ",
                        f"📊 Score: {influence_score:.3f}"
                    ], style={'color': '#666', 'fontSize': '11px'})
                ], style={
                    'padding': '12px',
                    'marginBottom': '10px',
                    'border': '1px solid #e0e0e0',
                    'borderRadius': '8px',
                    'backgroundColor': '#f9f9f9'
                })
                tweet_elements.append(tweet_card)
            
            return html.Div(tweet_elements)
        
        # Use sample_df for tweet display if available
        global sample_df
        if not sample_df.empty and 'sentiment' in sample_df.columns:
            print(f"Using sample_df with {len(sample_df)} tweets for sentiment display")
            top_negative_tweets = get_top_tweets_by_sentiment(sample_df, 'negative')
            top_positive_tweets = get_top_tweets_by_sentiment(sample_df, 'positive')
        else:
            print("No sample_df available")
            top_negative_tweets = html.Div('No sentiment analysis available for negative tweets.')
            top_positive_tweets = html.Div('No sentiment analysis available for positive tweets.')
        
        # Store dataframe globally
        current_dataframe = df
        
        success_message = html.Div(f'File "{filename}" processed successfully!', className="text-success")
        
        # Return all outputs
        return (
            False,  # progress-alert
            success_message,
            sentiment_fig,
            timeline_fig,
            country_fig,
            followers_fig,
            hashtag_fig,
            tweet_generator_fig,  # tweet-generator-pie-chart
            media_type_fig,  # media-type-pie-chart
            activity_gauge_fig,  # activity-gauge-chart
            activity_level_text,  # activity-level-text
            ai_summary,
            str(max_tweet_per_minute),
            str(likes),
            str(retweets),
            str(replies),
            str(unique_users),
            top_negative_tweets,
            top_positive_tweets,
            generate_top_influencers_display(sample_df)
        )
        
    except Exception as e:
        current_dataframe = None
        return (
            False,  # progress alert closed
            html.Div(f"Error processing file: {str(e)}", className="text-danger"),
            empty_fig,  # sentiment-pie-chart
            empty_fig,  # timeline-chart
            empty_fig,  # country-bar-chart
            empty_fig,  # followers-bar-chart
            empty_fig,  # hashtag-bar-chart
            empty_fig,  # tweet-generator-pie-chart
            empty_fig,  # media-type-pie-chart
            create_gauge_chart(5),  # activity-gauge-chart
            'Error',  # activity-level-text
            "Error: Could not process the uploaded file.",  # summary-text
            'N/A',  # max-tweet-per-minute-value
            'N/A',  # likes-value
            'N/A',  # retweets-value
            'N/A',  # replies-value
            'N/A',  # unique-users-value
            html.Div('Error processing file.'),  # top-negative-tweets
            html.Div('Error processing file.'),   # top-positive-tweets
            html.Div('Error processing file.')  # top-influencers-list
        )

# CSV download callback
@app.callback(
    Output('download-influencers', 'data'),
    Input('download-influencers-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_influencers_csv(n_clicks):
    global sample_df
    if n_clicks and not sample_df.empty:
        influencers_df = sample_df.copy()
        
        columns_to_include = ['SenderScreenName', 'Message', 'Sender Followers Count', 'Favorites', 'Retweets', 'influence_score']
        available_columns = [col for col in columns_to_include if col in influencers_df.columns]
        
        if available_columns:
            export_df = influencers_df[available_columns].copy()
            if 'influence_score' in export_df.columns:
                export_df = export_df.sort_values('influence_score', ascending=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"top_influencers_{timestamp}.csv"
            
            return dcc.send_data_frame(export_df.to_csv, filename, index=False)
    
    return None

# Run the app
if __name__ == '__main__':
    # Get port from environment variable or default to 8053
    port = int(os.environ.get('PORT', 8053))
    # For deployment, bind to 0.0.0.0, for local development use 127.0.0.1
    host = '0.0.0.0' if 'PORT' in os.environ else '127.0.0.1'
    
    app.run_server(debug=False, host=host, port=port)
        
