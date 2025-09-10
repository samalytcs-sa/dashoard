import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import base64
import io
from collections import Counter
from textblob import TextBlob
import re
import warnings
import google.generativeai as genai
import json
import time
warnings.filterwarnings('ignore')

# Import spam detection function
def detect_spam(text: str) -> bool:
    """Enhanced spam detection function"""
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Arabic spam keywords
    spam_keywords = [
    'اربح', 'مال', 'مجاني', 'اتصل الآن', 'عرض خاص',
    'خصم', 'مليون', 'دولار', 'ريال', 'فوز', 'جائزة',
    'سريع', 'فوري', 'حصري', 'محدود', 'الآن فقط',
    'جنس', 'حب', 'ممارسة', 'علاقة', 'مداعبة', 'حميمية',
    'رغبة', 'إثارة', 'مثير', 'قضيب', 'مهبل', 'صورة إباحية',
    'فيلم إباحي', 'مباشر', 'عري', 'لقب', 'لعب', 'مثير جنسيًا',
    'فضيحة', 'خيانة', 'ممارسة جنسية', 'شهوة', 'مغتصب', 'عشيق',
    'عشيقة', 'مثير', 'نزوة', 'شرج', 'جماع', 'نكاح',
    'عريانة', 'عشيقة', 'عضو تناسلي', 'قذف', 'مثير', 'استمناء',
    'قسيمة', 'الفحولة', 'جيجل', 'المسيار', 'مسيار',
    'الرابط', 'اتصل',
    # Added words
    'اشترِ الآن', 'توفير', 'اكسب', 'تخفيضات', 'عرض لفترة محدودة', 'فرصة رائعة',
    'مجانًا', 'تحقق من', 'سعر خاص', 'حصريًا', 'لا تفوت', 'مكافأة',
    'تسجيل مجاني', 'اطلب الآن', 'لا تنتظر', 'خصم كبير', 'سارع بالشراء',
    'صفقة اليوم', 'محدود العدد', 'هدية مجانية', 'تخفيض فوري',
    'اربح جائزة', 'الفرصة الأخيرة', 'قسط بدون فوائد', 'عرض مذهل',
    'عروض حصرية', 'توصيل مجاني', 'مبلغ ضخم', 'فائز', 'ربح مضمون',
    'سحب الجوائز', 'هبة', 'فرصة استثمار', 'مبلغ نقدي', 'زيادة الدخل',
    'السرعة مهمة', 'لا تفوت العرض', 'توفير كبير', 'منتج جديد',
    'تجربة مجانية', 'شراء الآن', 'فوز سهل', 'جائزة نقدية', 'ربح سريع',
    'عرض خاص جدا', 'مفاجأة اليوم', 'خصم لا يقاوم', 'اشتر الآن',
    'عرض حصري', 'فرصة لا تعوض', 'تسوق الآن', 'اكسب النقاط', 'هديتك بانتظارك'
]

    # Check for spam keywords
    spam_score = sum(1 for keyword in spam_keywords if keyword in text_lower)
    
    # Check for excessive punctuation
    exclamation_ratio = text.count('!') / len(text) if text else 0
    
    # Check for repeated characters
    repeated_chars = len(re.findall(r'(.)\1{3,}', text))
    
    # Check for all caps (for Arabic, check for excessive diacritics)
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Spam detection logic
    is_spam = (
        spam_score >= 2 or
        exclamation_ratio > 0.1 or
        repeated_chars > 2 or
        caps_ratio > 0.5
    )
    
    return is_spam

# Initialize Dash app with Arabic RTL support
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP
])
app.title = "لوحة تحليل المشاعر والهاشتاغ العربية"

# Configure Google Gemini AI
import os
api_key = os.environ.get('GEMINI_API_KEY', 'AIzaSyBJIfFfVWjeJqNcwc4Z1_gt01IrgBOmZvE')
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Global variables to store data
df1 = None
df2 = None

# Global variables to cache AI analysis results
cached_ai_results = {
    'sentiments': [],
    'emotions': [],
    'texts': [],
    'summary': None,
    'positive_tweets': [],
    'negative_tweets': []
}

def initialize_cache():
    """Initialize the cache with empty values"""
    global cached_ai_results
    cached_ai_results = {
        'sentiments': [],
        'emotions': [],
        'texts': [],
        'summary': None,
        'positive_tweets': [],
        'negative_tweets': []
    }
    print("Cache initialized successfully")

def ensure_sentiment_analysis(df):
    """
    Guaranteed sentiment analysis that always returns results
    This function ensures we always have sentiment data for the pie chart
    """
    global cached_ai_results
    
    if df is None or df.empty:
        print("DataFrame is None or empty")
        return []
    
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    if len(df) > 0:
        print(f"Sample row: {df.iloc[0].to_dict()}")
    
    # Find text column
    text_col = find_text_column(df)
    if not text_col or text_col not in df.columns:
        print(f"No suitable text column found. Available columns: {df.columns.tolist()}")
        # Try to use any column that might contain text
        for col in df.columns:
            if df[col].dtype == 'object' and not df[col].isna().all():
                text_col = col
                print(f"Using column '{col}' as text column")
                break
        
        if not text_col:
            print("No text column found at all")
            return []
    
    print(f"Using text column: {text_col}")
    
    # Get texts and perform sentiment analysis
    texts = df[text_col].dropna().tolist()
    print(f"Found {len(texts)} non-null texts")
    
    if not texts:
        print("No valid texts found")
        return []
    
    # Test basic sentiment analysis
    test_text = "This is a positive message!"
    test_result = analyze_sentiment_basic(test_text)
    print(f"Basic sentiment test: {test_result}")
    
    sentiments = []
    print("Starting sentiment analysis...")
    
    for i, text in enumerate(texts[:100]):  # Limit to first 100 for performance
        if isinstance(text, str) and len(text.strip()) >= 3:
            try:
                sentiment_result = analyze_sentiment_basic(str(text))
                sentiment = sentiment_result[0] if isinstance(sentiment_result, tuple) else sentiment_result
                sentiments.append(sentiment)
                if i < 5:  # Debug first 5
                    print(f"Text {i+1}: '{text[:50]}...' -> {sentiment}")
            except Exception as e:
                print(f"Error analyzing text {i+1}: {e}")
                sentiments.append('neutral')
        else:
            sentiments.append('neutral')
    
    print(f"Generated {len(sentiments)} sentiment results")
    print(f"Sentiment distribution: {Counter(sentiments)}")
    
    # Update cache
    cached_ai_results['sentiments'] = sentiments
    cached_ai_results['texts'] = texts[:len(sentiments)]
    
    print(f"Cache updated with {len(cached_ai_results['sentiments'])} sentiments")
    
    return sentiments

# Initialize cache at startup
initialize_cache()

# Helper functions for robust data handling
def find_column_safely(df, possible_names):
    """Safely find a column from a list of possible names"""
    if df is None or df.empty:
        return None
    for name in possible_names:
        if name in df.columns:
            return name
    # Try partial matches
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

def find_text_column(df):
    """Find the main text column in the dataframe"""
    text_columns = ['text', 'content', 'tweet', 'message', 'post', 'description']
    return find_column_safely(df, text_columns)

def generate_empty_dashboard_response():
    """Generate safe empty response for dashboard when errors occur"""
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text="حدث خطأ في معالجة البيانات",
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False,
        font=dict(size=16, color='white')
    )
    empty_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return [
        html.Div("لا توجد بيانات متاحة", className="text-white"),  # KPI cards
        empty_fig,  # sentiment_chart
        empty_fig,  # emotions_chart
        empty_fig,  # hashtags_chart
        empty_fig,  # engagement_chart
        empty_fig,  # time_series_chart
        empty_fig,  # places_chart
        empty_fig,  # timeline_chart
        empty_fig,  # tweets_per_minute_gauge
        "لا توجد بيانات للتحليل"  # summary
    ]

def extract_hashtags(text):
    """Extract hashtags from text"""
    if pd.isna(text):
        return []
    hashtags = re.findall(r'#\w+', str(text))
    return hashtags

def analyze_sentiment_batch_with_ai(texts, batch_size=10):
    """
    Batch sentiment analysis using Google Gemini AI with chunking to avoid timeouts
    Returns list of tuples: [(sentiment, score, emotions_dict), ...]
    """
    results = []
    
    # Process texts in smaller chunks to avoid API timeout
    chunk_size = min(batch_size, 10)  # Maximum 10 texts per API call
    
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(texts) + chunk_size - 1)//chunk_size} ({len(chunk)} texts)...")
        
        try:
            # Create prompt for current chunk
            chunk_prompt = "Analyze the following texts for sentiment and emotions. Return a JSON array with one object per text:\n\n"
            
            for idx, text in enumerate(chunk):
                if isinstance(text, str) and len(text.strip()) >= 3:
                    chunk_prompt += f"Text {idx+1}: \"{text[:150]}...\"\n"
                else:
                    chunk_prompt += f"Text {idx+1}: \"[empty or invalid text]\"\n"
            
            chunk_prompt += """
            
Respond with only this JSON array format (no other text):
            [
                {
                    "sentiment": "positive",
                    "confidence": 0.8,
                    "emotions": {
                        "joy": 0.5,
                        "anger": 0.0,
                        "fear": 0.0,
                        "sadness": 0.0,
                        "surprise": 0.0,
                        "trust": 0.3
                    }
                },
                ...
            ]
            
            Sentiment must be: positive, negative, or neutral
            Confidence: 0.0 to 1.0
            Emotions: values 0.0 to 1.0
            """
            
            # Add timeout handling
            response = model.generate_content(
                chunk_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048
                )
            )
            response_text = response.text.strip()
            
            # Clean the response text
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].strip()
            
            # Try to extract JSON array from the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                chunk_data = json.loads(json_text)
            else:
                raise ValueError("No valid JSON array found in response")
            
            print(f"AI response parsed successfully: {len(chunk_data)} results")
            
            # Process each result in the chunk
            for idx, text in enumerate(chunk):
                if idx < len(chunk_data):
                    result = chunk_data[idx]
                    sentiment = result.get('sentiment', 'neutral')
                    if sentiment not in ['positive', 'negative', 'neutral']:
                        sentiment = 'neutral'
                        
                    confidence = float(result.get('confidence', 0.5))
                    emotions = result.get('emotions', {})
                    
                    print(f"AI result {idx+1}: {sentiment} (confidence: {confidence})")
                    
                    # Validate emotions
                    valid_emotions = {}
                    for emotion in ['joy', 'anger', 'fear', 'sadness', 'surprise', 'trust']:
                        valid_emotions[emotion] = max(0.0, min(1.0, float(emotions.get(emotion, 0.0))))
                    
                    results.append((sentiment, confidence, valid_emotions))
                else:
                    # Fallback for missing results
                    sentiment, score = analyze_sentiment_basic(text)
                    emotions = analyze_emotions(text)
                    results.append((sentiment, score, emotions))
                    print(f"Fallback result {idx+1}: {sentiment} (score: {score})")
            
            # Small delay between chunks to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Chunk AI analysis failed, falling back to basic analysis for chunk: {e}")
            print(f"Response text was: {response_text[:200] if 'response_text' in locals() else 'No response'}")
            # Fallback to basic analysis for current chunk
            for text in chunk:
                sentiment, score = analyze_sentiment_basic(text)
                emotions = analyze_emotions(text)
                results.append((sentiment, score, emotions))
                print(f"Basic analysis result: {sentiment} (score: {score})")
    
    return results

def analyze_sentiment_with_ai(text):
    """
    Single text sentiment analysis - now uses batch function for consistency
    Returns tuple: (sentiment, score, emotions_dict)
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'neutral', 0.0, {}
    
    # Use batch function with single text
    results = analyze_sentiment_batch_with_ai([text], batch_size=1)
    return results[0] if results else ('neutral', 0.0, {})

def analyze_sentiment_basic(text):
    """
    Enhanced sentiment analysis with Arabic and English word-based approach
    Returns tuple: (sentiment, score)
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 'neutral', 0.0
    
    text_lower = text.lower()
    
    # Expanded Arabic positive words
    arabic_positive = [
        'ممتاز', 'رائع', 'جميل', 'حلو', 'جيد', 'أحب', 'سعيد', 'فرح', 'مبسوط',
        'حبيبي', 'عظيم', 'مذهل', 'بديع', 'لذيذ', 'حماس', 'متحمس', 'فخور',
        'مشكور', 'شكرا', 'الله يعطيك العافية', 'بارك الله', 'ماشاء الله',
        'تسلم', 'يسلمو', 'حلال عليك', 'كفو', 'تبارك', 'نعم', 'اي', 'صح',
        'موافق', 'اتفق', 'صحيح', 'اكيد', 'طبعا', 'حبيت', 'عجبني', 'اعجبني'
    ]
    
    # Expanded Arabic negative words
    arabic_negative = [
        'سيء', 'قبيح', 'مقرف', 'بشع', 'أكره', 'حزين', 'زعلان', 'مضايق',
        'غاضب', 'عصبان', 'مش عاجبني', 'ما يعجبني', 'فاشل', 'خايب', 'وسخ',
        'قذر', 'مقزز', 'مش حلو', 'وحش', 'عيب', 'حرام', 'غلط', 'خطأ',
        'لا', 'مش موافق', 'ارفض', 'مستحيل', 'ابدا', 'كلا', 'معارض',
        'مش صح', 'غير صحيح', 'كذب', 'باطل', 'مرفوض', 'مش مقبول'
    ]
    
    # English positive words
    english_positive = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
        'love', 'like', 'happy', 'joy', 'pleased', 'satisfied', 'perfect', 'best',
        'brilliant', 'outstanding', 'superb', 'marvelous', 'incredible', 'yes',
        'agree', 'correct', 'right', 'true', 'absolutely', 'definitely', 'sure'
    ]
    
    # English negative words
    english_negative = [
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
        'sad', 'angry', 'upset', 'disappointed', 'frustrated', 'annoyed', 'worst',
        'fail', 'failure', 'wrong', 'false', 'no', 'disagree', 'refuse', 'never',
        'impossible', 'reject', 'deny', 'oppose', 'against', 'unacceptable'
    ]
    
    # Emoji sentiment mapping
    positive_emojis = ['😊', '😃', '😄', '😁', '🙂', '😍', '🥰', '😘', '👍', '❤️', '💕', '🎉', '✨', '🌟']
    negative_emojis = ['😢', '😭', '😞', '😔', '😠', '😡', '🤬', '👎', '💔', '😤', '😒', '🙄', '😩', '😫']
    
    # Count positive and negative indicators
    positive_count = 0
    negative_count = 0
    
    # Check Arabic words
    for word in arabic_positive:
        if word in text:
            positive_count += 1
    
    for word in arabic_negative:
        if word in text:
            negative_count += 1
    
    # Check English words
    words = text_lower.split()
    for word in words:
        if word in english_positive:
            positive_count += 1
        elif word in english_negative:
            negative_count += 1
    
    # Check emojis
    for emoji in positive_emojis:
        if emoji in text:
            positive_count += 1
    
    for emoji in negative_emojis:
        if emoji in text:
            negative_count += 1
    
    # Determine sentiment based on word counts
    if positive_count > negative_count:
        sentiment = 'positive'
        score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        sentiment = 'negative'
        score = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
    else:
        # Use TextBlob as fallback with higher thresholds
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.3:  # Higher threshold for positive
                sentiment = 'positive'
                score = polarity
            elif polarity < -0.3:  # Higher threshold for negative
                sentiment = 'negative'
                score = abs(polarity)
            else:
                sentiment = 'neutral'
                score = 0.0
        except:
            sentiment = 'neutral'
            score = 0.0
    
    return sentiment, score

def analyze_emotions(text):
    """
    Analyze emotions in text using keyword matching
    Returns dictionary with emotion scores
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
    
    text_lower = text.lower()
    emotions = {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
    
    # Arabic emotion keywords
    joy_words_ar = ['فرح', 'سعيد', 'مبسوط', 'حماس', 'متحمس', 'ضحك', 'بهجة', 'سرور']
    anger_words_ar = ['غضب', 'عصبان', 'غاضب', 'زعلان', 'مضايق', 'عنف', 'كره']
    fear_words_ar = ['خوف', 'خايف', 'قلق', 'توتر', 'رعب', 'فزع', 'هلع']
    sadness_words_ar = ['حزن', 'حزين', 'كآبة', 'أسى', 'ألم', 'وجع', 'معاناة']
    surprise_words_ar = ['مفاجأة', 'تعجب', 'استغراب', 'دهشة', 'ذهول']
    trust_words_ar = ['ثقة', 'أمان', 'اطمئنان', 'يقين', 'اعتماد']
    
    # English emotion keywords
    joy_words_en = ['joy', 'happy', 'excited', 'cheerful', 'delighted', 'thrilled', 'elated']
    anger_words_en = ['angry', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'frustrated']
    fear_words_en = ['fear', 'scared', 'afraid', 'worried', 'anxious', 'terrified', 'panic']
    sadness_words_en = ['sad', 'depressed', 'melancholy', 'grief', 'sorrow', 'miserable']
    surprise_words_en = ['surprised', 'amazed', 'shocked', 'astonished', 'stunned']
    trust_words_en = ['trust', 'confident', 'secure', 'reliable', 'faith', 'believe']
    
    # Count emotion indicators
    for word in joy_words_ar + joy_words_en:
        if word in text_lower:
            emotions['joy'] += 1
    
    for word in anger_words_ar + anger_words_en:
        if word in text_lower:
            emotions['anger'] += 1
    
    for word in fear_words_ar + fear_words_en:
        if word in text_lower:
            emotions['fear'] += 1
    
    for word in sadness_words_ar + sadness_words_en:
        if word in text_lower:
            emotions['sadness'] += 1
    
    for word in surprise_words_ar + surprise_words_en:
        if word in text_lower:
            emotions['surprise'] += 1
    
    for word in trust_words_ar + trust_words_en:
        if word in text_lower:
            emotions['trust'] += 1
    
    return emotions

def generate_ai_summary(texts, sentiments, emotions_list, hashtags):
    """
    Generate an AI-powered comprehensive news feed-style summary
    """
    if not texts:
        return "No data available for summary."
    
    try:
        # Prepare data for AI analysis
        sample_texts = texts[:20]  # Use first 20 texts for AI analysis (reduced for faster processing)
        top_hashtags = [tag for tag, count in Counter(hashtags).most_common(10)]
        
        sentiment_counts = Counter(sentiments)
        emotion_totals = {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
        for emotions in emotions_list:
            for emotion, count in emotions.items():
                emotion_totals[emotion] += count
        
        prompt = f"""
        قم بإنشاء تحليل شامل بأسلوب الأخبار بناءً على بيانات وسائل التواصل الاجتماعي التالية:
        
        **نظرة عامة على البيانات:**
        - إجمالي الرسائل المحللة: {len(texts)}
        - توزيع المشاعر: {dict(sentiment_counts)}
        - أهم المشاعر: {emotion_totals}
        - الهاشتاغات الشائعة: {top_hashtags}
        
        **عينة من الرسائل:**
        {chr(10).join([f"- {text[:100]}..." for text in sample_texts[:10]])}
        
        يرجى تقديم ملخص احترافي بأسلوب الأخبار يتضمن:
        1. 📊 الملخص التنفيذي مع الإحصائيات الرئيسية
        2. 🎯 المواضيع والموضوعات الرئيسية
        3. 💭 تحليل المشاعر العامة
        4. 🔥 الهاشتاغات الرائجة وسياقها
        5. 📈 الرؤى والأنماط الرئيسية
        6. 🚨 المخاوف البارزة أو النقاط الإيجابية
        7. 📋 التوصيات لأصحاب المصلحة
        
        قم بتنسيق الاستجابة باللغة العربية مع الرموز التعبيرية والأقسام الواضحة. اجعلها جذابة ومفيدة مثل تقرير تحليلات وسائل التواصل الاجتماعي المهني.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"AI summary generation failed, using basic summary: {e}")
        return generate_summary(texts, sentiments, emotions_list)

def generate_summary(texts, sentiments, emotions_list):
    """
    Generate a news feed-style summary of the analyzed texts
    """
    if not texts:
        return "No data available for summary."
    
    total_texts = len(texts)
    positive_count = sum(1 for s in sentiments if s == 'positive')
    negative_count = sum(1 for s in sentiments if s == 'negative')
    neutral_count = sum(1 for s in sentiments if s == 'neutral')
    
    # Calculate dominant emotions
    emotion_totals = {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
    for emotions in emotions_list:
        for emotion, count in emotions.items():
            emotion_totals[emotion] += count
    
    dominant_emotion = max(emotion_totals, key=emotion_totals.get) if max(emotion_totals.values()) > 0 else 'neutral'
    
    # Generate summary
    summary = f"📊 **Analysis Summary** ({total_texts} messages analyzed)\n\n"
    
    # Sentiment breakdown
    summary += f"**Sentiment Distribution:**\n"
    summary += f"• Positive: {positive_count} ({positive_count/total_texts*100:.1f}%)\n"
    summary += f"• Negative: {negative_count} ({negative_count/total_texts*100:.1f}%)\n"
    summary += f"• Neutral: {neutral_count} ({neutral_count/total_texts*100:.1f}%)\n\n"
    
    # Emotional tone
    summary += f"**Emotional Tone:** {dominant_emotion.title()}\n\n"
    
    # Key insights
    if positive_count > negative_count:
        summary += "🟢 **Overall Mood:** Positive - The conversation shows optimistic engagement\n"
    elif negative_count > positive_count:
        summary += "🔴 **Overall Mood:** Negative - The conversation indicates concerns or dissatisfaction\n"
    else:
        summary += "🟡 **Overall Mood:** Balanced - Mixed reactions in the conversation\n"
    
    # Top emotions
    top_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)[:3]
    if any(count > 0 for _, count in top_emotions):
        summary += f"\n**Top Emotions Detected:**\n"
        for emotion, count in top_emotions:
            if count > 0:
                summary += f"• {emotion.title()}: {count} mentions\n"
    
    return summary

def create_emergency_summary(df):
    """Create a basic summary when all else fails"""
    if df is None or df.empty:
        return "لا توجد بيانات متاحة للتحليل."
    
    row_count = len(df)
    col_count = len(df.columns)
    
    return f"""
# ملخص البيانات الأساسي

📊 **إحصائيات عامة:**
- إجمالي السجلات: {row_count:,}
- عدد الأعمدة: {col_count}

📈 **حالة التحليل:**
- تم تحميل البيانات بنجاح
- التحليل المتقدم قيد التطوير

🔄 **الخطوات التالية:**
- سيتم إجراء تحليل المشاعر تلقائياً
- رفع ملفات إضافية لتحليل أعمق
"""

def test_gemini_api():
    """Test if Gemini API is working"""
    try:
        test_response = model.generate_content("Test message")
        print(f"Gemini API test successful: {test_response.text[:50]}...")
        return True
    except Exception as e:
        print(f"Gemini API test failed: {e}")
        return False

# حساب التفاعل باستخدام عدد المفضلات وعدد إعادة التغريد
def calculate_engagement_metrics(df):
    """
    Calculate engagement metrics from the dataframe
    """
    if df is None or df.empty:
        return {'max_tweet_per_minute': 0, 'likes': 0, 'retweets': 0, 'reply': 0, 'unique_users': 0}
    
    # Handle different column names
    favorites_col = None
    retweets_col = None
    time_col = None
    user_col = None
    
    # Find appropriate columns using safe detection
    favorites_col = find_column_safely(df, ['favorites', 'favorite_count', 'likes', 'like_count', 'fav_count'])
    retweets_col = find_column_safely(df, ['retweets', 'retweet_count', 'rt_count', 'shares'])
    time_col = find_column_safely(df, ['time', 'timestamp', 'created_at', 'date', 'datetime', 'posted_at'])
    user_col = find_column_safely(df, ['SenderScreenName', 'user', 'username', 'screen_name', 'user_name', 'author', 'name', 'user_id', 'screen_name_s'])
    
    # Calculate metrics
    try:
        # Max tweets per minute
        max_tweet_per_minute = 0
        if time_col and time_col in df.columns:
            try:
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors='coerce')
                df_time = df_time.dropna(subset=[time_col])
                if not df_time.empty:
                    max_tweet_per_minute = df_time.groupby(pd.Grouper(key=time_col, freq='1Min')).size().max()
            except:
                max_tweet_per_minute = len(df) // 60 if len(df) > 60 else len(df)
        
        # Likes
        likes = 0
        if favorites_col and favorites_col in df.columns:
            likes = pd.to_numeric(df[favorites_col], errors='coerce').fillna(0).sum()
        
        # Retweets
        retweets = 0
        if retweets_col and retweets_col in df.columns:
            retweets = pd.to_numeric(df[retweets_col], errors='coerce').fillna(0).sum()
        
        # Reply (using a proxy metric)
        reply = len(df)  # Total number of posts as proxy
        
        # Unique users
        unique_users = 1
        if user_col and user_col in df.columns:
            unique_users = df[user_col].nunique()
            print(f"Debug: Found user column '{user_col}' with {unique_users} unique users")
        else:
            print(f"Debug: No user column found. Available columns: {list(df.columns)}")
        
        return {
            'max_tweet_per_minute': int(max_tweet_per_minute),
            'likes': int(likes),
            'retweets': int(retweets),
            'reply': int(reply),
            'unique_users': int(unique_users)
        }
    except Exception as e:
        print(f"Error calculating engagement metrics: {e}")
        return {'max_tweet_per_minute': 0, 'likes': 0, 'retweets': 0, 'reply': 0, 'unique_users': 0}

def analyze_tweets_per_minute_level(max_tweet_per_minute):
    """
    Analyze tweet frequency level based on tweets per minute
    """
    if 5 < max_tweet_per_minute < 10:
        return 20, 'Level one'
    elif 10 < max_tweet_per_minute < 20:
        return 30, 'Level two'
    elif 20 < max_tweet_per_minute < 400:
        return 45, 'Level three'
    else:
        return 5, 'Very low'

def create_tweets_per_minute_gauge(level, max_tweets_per_minute=0):
    """
    Create an enhanced tweets per minute gauge chart with Arabic support
    """
    import numpy as np
    
    # Enhanced color scheme for better visibility
    plot_bgcolor = 'rgba(0,0,0,0)'
    quadrant_colors = [plot_bgcolor, "#dc3545", "#fd7e14", "#ffc107", "#28a745"]
    quadrant_text = ["", "<b>مستوى عالي</b>", "<b>مستوى متوسط</b>", "<b>مستوى منخفض</b>", "<b>مستوى ضعيف</b>"]
    n_quadrants = len(quadrant_colors) - 1
    
    current_value = min(max(level, 0), 50)  # Clamp between 0 and 50
    min_value = 0
    max_value = 50
    
    # Determine activity level and color
    if current_value >= 30:
        activity_level = "نشاط عالي"
        level_color = "#dc3545"
    elif current_value >= 20:
        activity_level = "نشاط متوسط"
        level_color = "#fd7e14"
    elif current_value >= 10:
        activity_level = "نشاط منخفض"
        level_color = "#ffc107"
    else:
        activity_level = "نشاط ضعيف"
        level_color = "#28a745"
    
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
                textfont=dict(size=10, color='white', family="Cairo, Arial, sans-serif")
            ),
        ],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=20, t=20, l=20, r=20),
            width=300,
            height=300,
            paper_bgcolor=plot_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font=dict(family="Cairo, Arial, sans-serif", color='white'),
            annotations=[
                go.layout.Annotation(
                    text=f"<b style='color:{level_color}; font-size:16px;'>{max_tweets_per_minute}</b><br><span style='color:#6c757d; font-size:12px;'>تغريدة/دقيقة</span><br><span style='color:{level_color}; font-size:11px;'>{activity_level}</span>",
                    x=0.5, xanchor="center", xref="paper",
                    y=0.5, yanchor="middle", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="white", family="Cairo, Arial, sans-serif")
                )
            ]
        )
    )
    
    # Add needle/pointer
    hand_length = 0.4
    hand_angle = np.pi * (1 - (current_value - min_value) / (max_value - min_value))
    
    fig.add_shape(
        type="line",
        x0=0.5, y0=0.5,
        x1=0.5 + hand_length * np.cos(hand_angle),
        y1=0.5 + hand_length * np.sin(hand_angle),
        line=dict(color=level_color, width=4),
        xref="paper", yref="paper"
    )
    
    # Add center circle
    fig.add_shape(
        type="circle",
        x0=0.48, y0=0.48, x1=0.52, y1=0.52,
        fillcolor=level_color,
        line=dict(color=level_color, width=2),
        xref="paper", yref="paper"
    )
    
    return fig

def create_engagement_scatter(df):
    """
    Create engagement scatter plot showing followers vs retweets with sentiment color coding
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for engagement analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Find relevant columns
    follower_cols = [col for col in df.columns if 'follower' in col.lower() or 'follow' in col.lower()]
    retweet_cols = [col for col in df.columns if 'retweet' in col.lower() or 'rt' in col.lower() or 'share' in col.lower()]
    sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
    
    if not follower_cols or not retweet_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No follower or retweet data found for engagement analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Use first available columns
    follower_col = follower_cols[0]
    retweet_col = retweet_cols[0]
    
    # Prepare data
    df_clean = df[[follower_col, retweet_col]].copy()
    df_clean = df_clean.dropna()
    
    # Convert to numeric
    df_clean[follower_col] = pd.to_numeric(df_clean[follower_col], errors='coerce')
    df_clean[retweet_col] = pd.to_numeric(df_clean[retweet_col], errors='coerce')
    df_clean = df_clean.dropna()
    
    if df_clean.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid numeric data for engagement analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Add sentiment data if available
    if sentiment_cols and len(cached_ai_results.get('sentiments', [])) > 0:
        sentiments = cached_ai_results['sentiments'][:len(df_clean)]
        # Pad with 'neutral' if needed
        while len(sentiments) < len(df_clean):
            sentiments.append('neutral')
        df_clean['sentiment'] = sentiments[:len(df_clean)]
        
        # Create color mapping
        color_map = {'positive': '#28a745', 'negative': '#dc3545', 'neutral': '#6c757d'}
        colors = [color_map.get(s, '#6c757d') for s in df_clean['sentiment']]
        
        fig = px.scatter(
            df_clean, 
            x=follower_col, 
            y=retweet_col,
            color='sentiment',
            color_discrete_map=color_map,
            title="Engagement Analysis: Followers vs Retweets by Sentiment",
            labels={follower_col: 'Followers', retweet_col: 'Retweets'},
            hover_data=[follower_col, retweet_col, 'sentiment']
        )
    else:
        fig = px.scatter(
            df_clean, 
            x=follower_col, 
            y=retweet_col,
            title="Engagement Analysis: Followers vs Retweets",
            labels={follower_col: 'Followers', retweet_col: 'Retweets'},
            color_discrete_sequence=['#17a2b8']
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Followers",
        yaxis_title="Retweets",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        showlegend=True if sentiment_cols else False
    )
    
    # Add trend line
    fig.add_traces(
        px.scatter(
            df_clean, 
            x=follower_col, 
            y=retweet_col, 
            trendline="ols"
        ).data[1:]
    )
    
    return fig

def create_time_series_chart(df, time_period):
    """
    Create time series chart for tweet counts based on selected time period
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for time series analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Find date column
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower()]
    
    if not date_columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No date column found in the data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    date_col = date_columns[0]
    
    try:
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        
        if df_copy.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid dates found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Group by time period
        if time_period == 'minute':
            df_copy['time_group'] = df_copy[date_col].dt.floor('T')  # Minute
            title = "عدد التغريدات لكل دقيقة"
            x_title = "الوقت (دقيقة)"
        elif time_period == 'day':
            df_copy['time_group'] = df_copy[date_col].dt.date
            title = "عدد التغريدات لكل يوم"
            x_title = "التاريخ (يوم)"
        elif time_period == 'week':
            df_copy['time_group'] = df_copy[date_col].dt.to_period('W').dt.start_time
            title = "عدد التغريدات لكل أسبوع"
            x_title = "التاريخ (أسبوع)"
        else:
            df_copy['time_group'] = df_copy[date_col].dt.date
            title = "عدد التغريدات لكل يوم"
            x_title = "التاريخ (يوم)"
        
        # Count tweets per time period
        time_counts = df_copy.groupby('time_group').size().reset_index(name='count')
        
        # Create the chart with proper line visualization
        fig = go.Figure()
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=time_counts['time_group'],
            y=time_counts['count'],
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3, shape='spline'),
            marker=dict(size=8, color='#00d4ff', symbol='circle'),
            fill='tonexty',
            fillcolor='rgba(0, 212, 255, 0.1)',
            name='عدد التغريدات',
            hovertemplate='<b>%{x}</b><br>عدد التغريدات: %{y}<extra></extra>'
        ))
        
        # Dark theme layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=18, color='white'),
                x=0.5
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(size=12, color='white'),
            xaxis=dict(
                 title=x_title,
                 title_font=dict(size=14, color='white'),
                 tickfont=dict(color='white'),
                 gridcolor='#404040',
                 showgrid=True
             ),
             yaxis=dict(
                 title='عدد التغريدات',
                 title_font=dict(size=14, color='white'),
                 tickfont=dict(color='white'),
                 gridcolor='#404040',
                 showgrid=True
             ),
            hovermode='x unified',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating time series chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating time series chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig

def create_sentiment_pie_chart(df):
    """
    Create modern flat pie chart for sentiment distribution with AI cache priority and fallback
    """
    global cached_ai_results
    
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="لا توجد بيانات متاحة لتحليل المشاعر",
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
    
    try:
        sentiments = []
        
        # Use cached results if available, otherwise perform basic analysis
        if cached_ai_results['sentiments'] and len(cached_ai_results['sentiments']) > 0:
            print(f"Using cached AI sentiment results: {len(cached_ai_results['sentiments'])} sentiments")
            sentiments = cached_ai_results['sentiments']
        else:
            print("No cached results found, performing basic sentiment analysis...")
            
            # Find text column using your helper function
            text_col = find_text_column(df)
            
            if not text_col or text_col not in df.columns:
                print("No suitable text column found")
                fig = go.Figure()
                fig.add_annotation(
                    text="لم يتم العثور على عمود النص في البيانات",
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
            
            print(f"Processing {len(df[text_col].dropna())} texts for basic sentiment analysis...")
            
            # Perform basic sentiment analysis
            for text in df[text_col].dropna():
                if isinstance(text, str) and len(text.strip()) >= 3:
                    sentiment_result = analyze_sentiment_basic(str(text))
                    sentiment = sentiment_result[0] if isinstance(sentiment_result, tuple) else sentiment_result
                    sentiments.append(sentiment)
                else:
                    sentiments.append('neutral')
            
            # Update cache with basic analysis results
            cached_ai_results['sentiments'] = sentiments
            print(f"Generated {len(sentiments)} basic sentiment results and updated cache")
        
        # Create pie chart from sentiment counts
        sentiment_counts = Counter(sentiments)
        print(f"Final sentiment counts: {sentiment_counts}")
        
        if not sentiment_counts:
            fig = go.Figure()
            fig.add_annotation(
                text="لا توجد بيانات مشاعر متاحة",
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
        
        # Prepare data for pie chart
        arabic_labels = []
        values = []
        colors = []
        
        # Arabic labels and modern colors
        label_mapping = {
            'positive': 'إيجابي',
            'negative': 'سلبي', 
            'neutral': 'محايد'
        }
        
        # Modern flat color scheme
        color_mapping = {
            'positive': '#2DD4BF',  # Modern teal
            'negative': '#F87171',  # Modern coral red
            'neutral': '#60A5FA'    # Modern blue
        }
        
        for sentiment, count in sentiment_counts.items():
            if count > 0:
                arabic_labels.append(label_mapping.get(sentiment, sentiment))
                values.append(count)
                colors.append(color_mapping.get(sentiment, '#60A5FA'))
        
        if not values:
            fig = go.Figure()
            fig.add_annotation(
                text="لا توجد بيانات صالحة للعرض",
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
        
        # Create modern flat pie chart
        fig = go.Figure(data=[go.Pie(
            labels=arabic_labels,
            values=values,
            hole=0.3,  # Small hole for modern donut look
            marker=dict(
                colors=colors,
                line=dict(color='white', width=3)  # Clean white borders
            ),
            textinfo='percent',
            textposition='inside',
            textfont=dict(size=11, color='white', family='Arial'),
            pull=[0.05, 0.05, 0.05],  # Slightly separate all slices
            hovertemplate='<b>%{label}</b><br>العدد: %{value}<br>النسبة: %{percent}<extra></extra>',
            sort=False  # Keep original order
        )])
        
        # Modern clean layout
        fig.update_layout(
            title=dict(
                text="توزيع المشاعر",
                font=dict(size=16, color='white', family='Arial', weight='bold'),
                x=0.5,
                y=0.95
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14, color='white', family='Arial'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
                font=dict(size=10, color='white'),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=5, r=5, t=40, b=30),
            height=280,
            annotations=[
                dict(
                    text=f"إجمالي: {sum(values)}",
                    x=0.5, y=0.5,
                    font=dict(size=14, color='white', weight='bold'),
                    showarrow=False
                )
            ]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating sentiment pie chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"خطأ في إنشاء مخطط المشاعر: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=14, color='white')
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

def create_emotions_bar_chart(df):
    """
    Create bar chart for emotions distribution
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for emotions analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Find text column
    text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower() or 'tweet' in col.lower()]
    
    if not text_columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No text column found in the data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    text_col = text_columns[0]
    
    try:
        # Analyze emotions
        all_emotions = []
        for text in df[text_col].dropna():
            emotions = analyze_emotions(str(text))
            all_emotions.extend(emotions)
        
        # Count emotions
        emotion_counts = Counter(all_emotions)
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=list(emotion_counts.keys()),
            y=list(emotion_counts.values()),
            marker_color='#00d4ff'
        )])
        
        fig.update_layout(
            title=dict(
                text="توزيع المشاعر",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(size=12, color='white'),
            xaxis=dict(
                title='المشاعر',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            ),
            yaxis=dict(
                title='العدد',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating emotions bar chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating emotions bar chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig

def create_places_bar_chart(df):
    """
    Create bar chart for places/locations distribution
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for places analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Find location/place columns
    location_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['location', 'place', 'city', 'country', 'geo'])]
    
    if not location_columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No location column found in the data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    location_col = location_columns[0]
    
    try:
        # Count locations
        location_counts = df[location_col].value_counts().head(10)  # Top 10 locations
        
        # Create bar chart
        fig = go.Figure(data=[go.Bar(
            x=location_counts.index,
            y=location_counts.values,
            marker_color='#ff6b6b'
        )])
        
        fig.update_layout(
            title=dict(
                text="توزيع الأماكن",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(size=12, color='white'),
            xaxis=dict(
                title='الأماكن',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            ),
            yaxis=dict(
                title='العدد',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating places bar chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating places bar chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig

def create_tweets_timeline_chart(df):
    """
    Create timeline chart for tweets per minute
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for timeline analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    # Find date column
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower()]
    
    if not date_columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No date column found in the data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig
    
    date_col = date_columns[0]
    
    try:
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        
        if df_copy.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid dates found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            return fig
        
        # Group by minute
        df_copy['minute'] = df_copy[date_col].dt.floor('T')
        minute_counts = df_copy.groupby('minute').size().reset_index(name='count')
        
        # Create timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=minute_counts['minute'],
            y=minute_counts['count'],
            mode='lines+markers',
            line=dict(color='#4ecdc4', width=2),
            marker=dict(size=6, color='#4ecdc4'),
            name='تغريدات لكل دقيقة'
        ))
        
        fig.update_layout(
            title=dict(
                text="الجدول الزمني للتغريدات (لكل دقيقة)",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(size=12, color='white'),
            xaxis=dict(
                title='الوقت',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            ),
            yaxis=dict(
                title='عدد التغريدات',
                title_font=dict(size=14, color='white'),
                tickfont=dict(color='white'),
                gridcolor='#404040'
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating tweets timeline chart: {e}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating tweets timeline chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return fig

# Professional Arabic Dashboard Layout with RTL Design
# Custom CSS for Arabic RTL and Professional Styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            }
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
            }
            .rtl {
                direction: rtl;
                text-align: right;
            }
            .arabic-title {
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            .arabic-subtitle {
                font-weight: 400;
                opacity: 0.8;
            }
            .card-header-arabic {
                direction: rtl;
                text-align: right;
            }
            .dropdown-arabic .Select-control {
                direction: rtl;
                text-align: right;
            }
            .form-control-arabic {
                direction: rtl;
                text-align: right;
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

app.layout = html.Div([
    
    dbc.Container([
        # Header Section - Arabic Style
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1("📊 منصة تحليل وسائل التواصل الاجتماعي", 
                           className="text-white mb-2 arabic-title rtl",
                           style={'fontSize': '2.5rem', 'fontWeight': '700'}),
                    html.P("منصة متقدمة لتحليل المشاعر والذكاء الاجتماعي",
                          className="text-white mb-0 arabic-subtitle rtl",
                          style={'fontSize': '1.1rem'})
                ], style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'padding': '50px 40px',
                    'borderRadius': '20px',
                    'marginBottom': '40px',
                    'boxShadow': '0 15px 35px rgba(102, 126, 234, 0.3)',
                    'border': '1px solid rgba(255, 255, 255, 0.1)'
                })
            ])
        ]),
    
    # KPI Cards Row - Professional Metrics Matrix
    html.Div(id='kpi-cards-section', style={'marginBottom': '30px'}),
    
    # Control Panel Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.Span("📁 ", style={'color': '#00d4ff', 'marginRight': '8px'}),
                        html.Span("رفع البيانات والإعدادات", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    # Upload Section
                    dbc.Row([
                        dbc.Col([
                            html.Label("📁 رفع ملفات البيانات", className="fw-bold mb-2 rtl", style={'color': 'white'}),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.Span("☁️📤", style={'fontSize': '2rem', 'color': '#667eea', 'marginBottom': '8px'}),
                                    html.Br(),
                                    'اسحب وأفلت ملفات Excel أو انقر للتصفح'
                                ], style={'textAlign': 'center', 'color': '#cccccc', 'direction': 'rtl'}),
                                style={
                                    'width': '100%',
                                    'height': '100px',
                                    'lineHeight': '100px',
                                    'borderWidth': '2px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '10px',
                                    'borderColor': '#444444',
                                    'textAlign': 'center',
                                    'background': '#2d2d2d',
                                    'cursor': 'pointer',
                                    'transition': 'all 0.3s ease'
                                },
                                multiple=True
                            ),
                            html.Div(id='upload-status', className="mt-3"),
            html.Div(id='ai-processing-status', className="mt-3"),  # Processing status indicator
            dcc.Interval(
                id='processing-interval',
                interval=2000,  # Check every 2 seconds
                n_intervals=0
            )
                        ])
                    ], className="mb-4"),
                    
                    # Advanced Filters Matrix
                    html.Hr(),
                    html.Label("🎛️ المرشحات المتقدمة", className="fw-bold mb-3 rtl", style={'color': 'white'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("الحد الأدنى للمتابعين", className="small fw-bold rtl", style={'color': '#cccccc'}),
                            dcc.Input(
                                id='min-followers',
                                type='number',
                                placeholder='١٠٠٠',
                                className='form-control',
                                style={'borderRadius': '8px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("الحد الأدنى لإعادة التغريد", className="small fw-bold rtl", style={'color': '#cccccc'}),
                            dcc.Input(
                                id='min-retweets',
                                type='number',
                                placeholder='١٠',
                                className='form-control',
                                style={'borderRadius': '8px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("الشبكة الاجتماعية", className="small fw-bold rtl", style={'color': '#cccccc'}),
                            dcc.Dropdown(
                                id='social-network-filter',
                                options=[],
                                placeholder='جميع الشبكات',
                                multi=True,
                                style={'borderRadius': '8px'}
                            )
                        ], width=3),
                        dbc.Col([
                            html.Label("نوع التفاعل", className="small fw-bold rtl", style={'color': '#cccccc'}),
                            dcc.Dropdown(
                                id='engagement-filter',
                                options=[],
                                placeholder='جميع الأنواع',
                                multi=True,
                                style={'borderRadius': '8px'}
                            )
                        ], width=3)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='high-influence-filter',
                                options=[{'label': '⭐ أصحاب التأثير العالي فقط (أكثر من ٢٥ ألف متابع)', 'value': 'high_influence'}],
                                value=[],
                                className="mt-2"
                            )
                        ])
                    ])
                ])
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ])
    ], className="mb-4"),
    
    # Analytics Matrix - 2x2 Grid
    dbc.Row([
        # Left Column - Charts
        dbc.Col([
            # Hashtag Analysis
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-hashtag me-2", style={'color': '#28a745'}),
                        html.Span("اتجاهات الهاشتاغ", style={'fontWeight': '600'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='hashtag-chart', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'marginBottom': '25px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'}),
            
            # Emotion Analysis
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-brain me-2", style={'color': '#6f42c1'}),
                        html.Span("ذكاء المشاعر", style={'fontWeight': '600'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='emotion-chart', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6),
        
        # Right Column - Analytics
        dbc.Col([
            # Sentiment Analysis
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-chart-pie me-2", style={'color': '#dc3545'}),
                        html.Span("توزيع المشاعر", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='sentiment-chart', style={'height': '180px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'marginBottom': '25px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'}),
            
            # Data Source Distribution
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-network-wired me-2", style={'color': '#fd7e14'}),
                        html.Span("تحليل المصادر", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='source-chart', style={'height': '180px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'marginBottom': '25px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'}),
            

            
            # AI Summary Panel
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-robot me-2", style={'color': '#20c997'}),
                        html.Span("رؤى الذكاء الاصطناعي", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    html.Div(id='analysis-summary', style={'maxHeight': '200px', 'overflowY': 'auto'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6)
    ], className="mb-4"),
    
    # Time Series Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-chart-line me-2", style={'color': '#007bff'}),
                        html.Span("التحليل الزمني للتغريدات", style={'fontWeight': '600', 'color': 'white'}),
                        html.Div([
                            dcc.Dropdown(
                                id='time-series-selector',
                                options=[
                                    {'label': 'دقيقة', 'value': 'minute'},
                                    {'label': 'يوم', 'value': 'day'},
                                    {'label': 'أسبوع', 'value': 'week'}
                                ],
                                value='day',
                                style={'width': '150px', 'fontSize': '14px', 'direction': 'rtl'}
                            )
                        ], style={'marginLeft': 'auto'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'direction': 'rtl'})
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='time-series-chart', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ])
    ], className="mb-4"),
    
    # Engagement Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-users me-2", style={'color': '#17a2b8'}),
                        html.Span("تحليل التفاعل", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='engagement-scatter', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6),

    ], className="mb-4"),
    
    # Content Analysis Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-thumbs-up me-2", style={'color': '#28a745'}),
                        html.Span("المحتوى الإيجابي", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody(id='positive-tweets-content', style={'maxHeight': '400px', 'overflowY': 'auto'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-thumbs-down me-2", style={'color': '#dc3545'}),
                        html.Span("المحتوى السلبي", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #dc3545 0%, #e74c3c 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody(id='negative-tweets-content', style={'maxHeight': '400px', 'overflowY': 'auto'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6)
    ], className="mb-4"),
    

    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-map-marker-alt me-2", style={'color': '#e83e8c'}),
                        html.Span("التوزيع الجغرافي", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='places-bar-chart', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Div([
                        html.I(className="fas fa-clock me-2", style={'color': '#6610f2'}),
                        html.Span("الجدول الزمني للتغريدات", style={'fontWeight': '600', 'color': 'white'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none', 'direction': 'rtl'}),
                dbc.CardBody([
                    dcc.Graph(id='tweets-timeline-chart', style={'height': '400px'})
                ], style={'padding': '20px'})
            ], style={'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)', 'border': '1px solid rgba(255, 255, 255, 0.1)', 'borderRadius': '16px', 'background': 'rgba(255, 255, 255, 0.95)', 'backdropFilter': 'blur(10px)'})
        ], width=6)
    ], className="mb-4")
    ], fluid=True, style={'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)', 'minHeight': '100vh', 'padding': '30px'})
])

# Callback
# Callback for AI processing - only runs when data is uploaded
# AI processing is now handled within the main callback
@app.callback(
    Output('ai-processing-status', 'children'),
    [Input('processing-interval', 'n_intervals')]
)
def update_processing_status(n_intervals):
    """Update processing status based on AI analysis completion"""
    global cached_ai_results, df1, df2
    
    # Only show processing status if data has been uploaded
    if df1 is None:
        return ""
    
    if cached_ai_results['sentiments']:
        return dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            "✅ AI Analysis completed successfully!"
        ], color="success", className="mb-0")
    else:
        return dbc.Alert([
            html.I(className="fas fa-spinner fa-spin me-2"),
            "🤖 AI Analysis in progress... This may take a few minutes."
        ], color="info", className="mb-0")


# Separate callback for time series chart to prevent full dashboard reload
@app.callback(
    Output('time-series-chart', 'figure'),
    [Input('time-series-selector', 'value')]
)
def update_time_series_chart(time_period):
    """Update only the time series chart when time period changes"""
    global df1, df2
    
    # Check if data is available
    if df1 is None:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data available for time series analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False
        )
        return empty_fig
    
    # Combine dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True) if df2 is not None and not df2.empty else df1
    
    # Create time series chart
    return create_time_series_chart(combined_df, time_period)

@app.callback(
    [Output('upload-status', 'children'),
      Output('kpi-cards-section', 'children'),
      Output('hashtag-chart', 'figure'),
      Output('sentiment-chart', 'figure'),
      Output('source-chart', 'figure'),
      Output('emotion-chart', 'figure'),
      Output('engagement-scatter', 'figure'),
      Output('places-bar-chart', 'figure'),
      Output('tweets-timeline-chart', 'figure'),
      Output('analysis-summary', 'children'),
      Output('social-network-filter', 'options'),
      Output('engagement-filter', 'options'),
      Output('positive-tweets-content', 'children'),
      Output('negative-tweets-content', 'children')],
    [Input('upload-data', 'contents'),
     Input('min-followers', 'value'),
     Input('min-retweets', 'value'),
     Input('social-network-filter', 'value'),
     Input('engagement-filter', 'value'),
     Input('high-influence-filter', 'value')],
    [State('upload-data', 'filename')]
)
def update_dashboard(contents, min_followers, min_retweets, social_networks, engagement_types, high_influence, filenames):
    global df1, df2
    
    print(f"Callback triggered: contents={contents is not None}, filenames={filenames}")
    print(f"Filters - min_followers: {min_followers}, min_retweets: {min_retweets}, social_networks: {social_networks}, engagement_types: {engagement_types}, high_influence: {high_influence}")
    
    if contents is None:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="No data uploaded yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        empty_fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white'
        )
        
        # Empty KPI cards for no data state
        empty_kpi_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-upload fa-2x mb-2", style={'color': '#bdc3c7'}),
                            html.H4("Upload Data", className="text-center", style={'color': '#7f8c8d'}),
                            html.P("No files uploaded yet", className="text-center text-muted mb-0")
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ])
                ], style={
                    'background': '#f8f9fa',
                    'border': '2px dashed #dee2e6',
                    'borderRadius': '15px',
                    'height': '140px'
                })
            ], width=12)
        ], className="mb-4")
        
        return (
                dbc.Alert("Please upload Excel files to begin analysis.", color="info"),
                empty_kpi_cards,
                empty_fig,  # hashtag chart
                empty_fig,  # sentiment chart
                empty_fig,  # source chart
                empty_fig,  # emotion chart
                empty_fig,  # engagement scatter
                empty_fig,  # places bar chart
                empty_fig,  # tweets timeline chart
                html.P("No data available for summary."),  # analysis summary
                [],  # social network options
                [],  # engagement options
                html.P("No positive tweets to display."),  # positive tweets
                html.P("No negative tweets to display.")   # negative tweets
            )
    
    try:
        # Process uploaded files (support 1 or more files)
        if len(contents) < 1:
            # Empty KPI cards for insufficient files
            empty_kpi_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle fa-2x mb-2", style={'color': '#f39c12'}),
                                html.H4("Need More Files", className="text-center", style={'color': '#7f8c8d'}),
                                html.P("Please upload at least 1 Excel file", className="text-center text-muted mb-0")
                            ], style={'textAlign': 'center', 'padding': '20px'})
                        ])
                    ], style={
                        'background': '#fff3cd',
                        'border': '2px solid #ffeaa7',
                        'borderRadius': '15px',
                        'height': '140px'
                    })
                ], width=12)
            ], className="mb-4")
            
            return (
                dbc.Alert("Please upload at least 1 Excel file.", color="warning"),
                empty_kpi_cards,
                go.Figure(),  # hashtag chart
                go.Figure(),  # sentiment chart
                go.Figure(),  # source chart
                go.Figure(),  # emotion chart
                go.Figure(),  # engagement scatter
                go.Figure(),  # sentiment pie chart
                go.Figure(),  # emotions bar chart
                go.Figure(),  # places bar chart
                go.Figure(),  # tweets timeline chart
                html.P("No data available for summary."),  # analysis summary
                [],  # social network options
                [],  # engagement options
                html.P("No positive tweets to display."),  # positive tweets
                html.P("No negative tweets to display.")   # negative tweets
            )
        
        # Parse the uploaded files
        dfs = []
        print(f"Processing {len(contents)} files...")
        
        for i, (content, filename) in enumerate(zip(contents, filenames)):
            print(f"Processing file {i+1}: {filename}")
            try:
                content_type, content_string = content.split(',')
                print(f"Content type: {content_type}")
                
                decoded = base64.b64decode(content_string)
                print(f"Decoded size: {len(decoded)} bytes")
                
                if filename.endswith('.xlsx') or filename.endswith('.xls'):
                    print("Reading as Excel file...")
                    try:
                        # Try xlrd engine first for better compatibility
                        print("Attempting xlrd engine reading...")
                        df = pd.read_excel(
                            io.BytesIO(decoded), 
                            engine='xlrd' if filename.endswith('.xls') else None,
                            sheet_name=0,
                            nrows=1500  # Conservative row limit
                        )
                        print(f"Successfully read Excel file: {df.shape[0]} rows, {df.shape[1]} columns")
                    except Exception as excel_error:
                        print(f"Error with xlrd reading: {excel_error}")
                        print("Trying openpyxl with minimal rows...")
                        try:
                            df = pd.read_excel(
                                io.BytesIO(decoded), 
                                engine='openpyxl',
                                nrows=500  # Very conservative limit
                            )
                            print(f"Successfully read with openpyxl: {df.shape[0]} rows, {df.shape[1]} columns")
                        except Exception as fallback_error:
                            print(f"All Excel reading methods failed: {fallback_error}")
                            # Create a sample dataframe to prevent complete failure
                            print("Creating sample data for demonstration...")
                            df = pd.DataFrame({
                                'text': ['Sample Arabic text #hashtag1', 'Another text #hashtag2'],
                                'hashtags': ['#hashtag1', '#hashtag2'],
                                'sentiment': ['positive', 'neutral']
                            })
                            print("Using sample data for demonstration")
                else:
                    print("Reading as CSV file...")
                    df = pd.read_csv(io.BytesIO(decoded))
                
                print(f"Successfully read file: {df.shape[0]} rows, {df.shape[1]} columns")
                dfs.append(df)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                return (
                    dbc.Alert(f"Error reading {filename}: {str(e)}", color="danger"),
                    html.Div(),  # empty kpi cards
                    go.Figure(),  # hashtag chart
                    go.Figure(),  # sentiment chart
                    go.Figure(),  # source chart
                    go.Figure(),  # emotion chart
                    go.Figure(),  # engagement scatter
                    go.Figure(),  # places bar chart
                    go.Figure(),  # tweets timeline chart
                    html.P("No data available for summary."),  # analysis summary
                    [],  # social network options
                    [],  # engagement options
                    html.P("No positive tweets to display."),  # positive tweets
                    html.P("No negative tweets to display.")   # negative tweets
                )
        
        # Combine all dataframes
        print("Combining dataframes...")
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataframe shape: {combined_df.shape}")
        
        # Assign dataframes (support single or multiple files)
        df1 = dfs[0] if len(dfs) > 0 else None
        df2 = dfs[1] if len(dfs) > 1 else None
        
        # Apply filters to combined dataframe
        filtered_df = combined_df.copy()
        print(f"Starting with {len(filtered_df)} rows before filtering")
        
        # Map column names to actual data columns
        follower_cols = ['Sender Followers Count', 'Follower Count', 'followers_count']
        retweet_cols = ['Retweets', 'retweet_count']
        social_network_cols = ['SocialNetwork', 'social_network']
        engagement_cols = ['Engagement Type', 'engagement_type']
        
        # Apply follower filter using safe column detection
        if min_followers is not None:
            follower_col = find_column_safely(filtered_df, follower_cols)
            
            if follower_col:
                # Convert to numeric, handling any non-numeric values
                filtered_df[follower_col] = pd.to_numeric(filtered_df[follower_col], errors='coerce')
                filtered_df = filtered_df[filtered_df[follower_col] >= min_followers]
                print(f"After follower filter (>={min_followers}): {len(filtered_df)} rows")
        
        # Apply retweet filter using safe column detection
        if min_retweets is not None:
            retweet_col = find_column_safely(filtered_df, retweet_cols)
            
            if retweet_col:
                # Convert to numeric, handling any non-numeric values
                filtered_df[retweet_col] = pd.to_numeric(filtered_df[retweet_col], errors='coerce')
                filtered_df = filtered_df[filtered_df[retweet_col] >= min_retweets]
                print(f"After retweet filter (>={min_retweets}): {len(filtered_df)} rows")
        
        # Apply social network filter using safe column detection
        if social_networks:
            social_col = find_column_safely(filtered_df, social_network_cols)
            
            if social_col:
                filtered_df = filtered_df[filtered_df[social_col].isin(social_networks)]
                print(f"After social network filter: {len(filtered_df)} rows")
        
        # Apply engagement type filter using safe column detection
        if engagement_types:
            engagement_col = find_column_safely(filtered_df, engagement_cols)
            
            if engagement_col:
                filtered_df = filtered_df[filtered_df[engagement_col].isin(engagement_types)]
                print(f"After engagement type filter: {len(filtered_df)} rows")
        
        # Apply high influence filter (>25k followers)
        if high_influence and 'high_influence' in high_influence:
            follower_col = None
            for col in follower_cols:
                if col in filtered_df.columns:
                    follower_col = col
                    break
            
            if follower_col:
                filtered_df[follower_col] = pd.to_numeric(filtered_df[follower_col], errors='coerce')
                filtered_df = filtered_df[filtered_df[follower_col] > 25000]
                print(f"After high influence filter (>25k): {len(filtered_df)} rows")
        
        # Detect spam in all tweets
        text_cols = ['Tweet', 'Text', 'text', 'Content', 'content']
        text_col = None
        for col in text_cols:
            if col in filtered_df.columns:
                text_col = col
                break
        
        total_tweets = len(filtered_df)
        
        # Spam calculation removed - now using tweets per minute analysis
        
        # Generate filter options from original data
        social_network_options = []
        engagement_options = []
        
        # Social network options
        for col in social_network_cols:
            if col in combined_df.columns:
                unique_networks = combined_df[col].dropna().unique()
                social_network_options = [{'label': str(net), 'value': str(net)} for net in unique_networks if str(net) != 'nan']
                break
        
        # Engagement type options
        for col in engagement_cols:
            if col in combined_df.columns:
                unique_engagements = combined_df[col].dropna().unique()
                engagement_options = [{'label': str(eng), 'value': str(eng)} for eng in unique_engagements if str(eng) != 'nan']
                break
        
        # Data overview
        total_rows = len(combined_df)
        total_columns = len(combined_df.columns)
        print(f"Data overview: {total_rows} rows, {total_columns} columns")
        print(f"Columns: {list(combined_df.columns)}")
        
        # Smart column mapping - find text and hashtag columns
        text_column = None
        hashtag_column = None
        
        # Look for text-like columns (prioritize exact matches first)
        text_candidates = ['Message', 'text', 'message', 'content', 'tweet', 'post', 'description']
        
        # First, try exact matches
        for candidate in text_candidates:
            if candidate in combined_df.columns:
                text_column = candidate
                break
        
        # If no exact match, try partial matches but exclude ID columns
        if not text_column:
            for candidate in text_candidates:
                for col in combined_df.columns:
                    if (candidate.lower() in col.lower() and 
                        'id' not in col.lower() and 
                        'link' not in col.lower() and
                        'url' not in col.lower()):
                        text_column = col
                        break
                if text_column:
                    break
        
        # Look for hashtag-like columns
        hashtag_candidates = ['hashtags', 'hashtag', 'tags', 'tag']
        for candidate in hashtag_candidates:
            for col in combined_df.columns:
                if candidate.lower() in col.lower():
                    hashtag_column = col
                    break
            if hashtag_column:
                break
        
        print(f"Column mapping: text_column='{text_column}', hashtag_column='{hashtag_column}'")
        print(f"DataFrame columns: {list(combined_df.columns)}")
        print(f"DataFrame shape: {combined_df.shape}")
        if len(combined_df) > 0:
            print(f"Sample row: {combined_df.iloc[0].to_dict()}")
        
        # If no text column found, use the first string column or create dummy data
        if not text_column:
            string_cols = combined_df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_column = string_cols[0]
                print(f"Using '{text_column}' as text column")
            else:
                # Create a dummy text column if none exists
                combined_df['text'] = 'Sample text data'
                text_column = 'text'
                print("Created dummy text column")
        
        # If no hashtag column found, create one from text content
        if not hashtag_column:
            print("No hashtag column found, extracting from text")
            combined_df['hashtags'] = combined_df[text_column].astype(str).apply(lambda x: ' '.join(re.findall(r'#\w+', x)) if pd.notna(x) else '')
            hashtag_column = 'hashtags'
        
        # Standardize column names for processing
        if text_column != 'text':
            combined_df['text'] = combined_df[text_column]
        if hashtag_column != 'hashtags':
            combined_df['hashtags'] = combined_df[hashtag_column]
        
        # Professional KPI Cards Matrix with robust data handling
        print("Creating KPI dashboard...")
        
        # Calculate key metrics with safe data access
        total_posts = len(filtered_df) if filtered_df is not None else 0
        total_files = len(dfs) if dfs else 0
        
        # Safe engagement calculation
        engagement_metrics = calculate_engagement_metrics(filtered_df) if filtered_df is not None else {}
        avg_engagement = engagement_metrics.get('retweets', 0)
        
        # Safe follower count calculation
        follower_col = find_column_safely(filtered_df, ['followers_count', 'follower_count', 'followers']) if filtered_df is not None else None
        high_influence_count = 0
        if follower_col and filtered_df is not None:
            try:
                follower_series = pd.to_numeric(filtered_df[follower_col], errors='coerce')
                high_influence_count = len(follower_series[follower_series > 25000])
            except Exception as e:
                print(f"Error calculating high influence count: {e}")
                high_influence_count = 0
        
        # Add spam detection integration
        spam_count = 0
        spam_percentage = 0
        text_col = find_text_column(filtered_df) if filtered_df is not None else None
        if text_col and filtered_df is not None:
            try:
                spam_count = sum(1 for text in filtered_df[text_col].dropna() 
                               if detect_spam(str(text)))
                spam_percentage = (spam_count / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
            except Exception as e:
                print(f"Error calculating spam metrics: {e}")
                spam_count = 0
                spam_percentage = 0
        
        # Create KPI cards
        kpi_cards = dbc.Row([
            # Total Posts Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-file-alt fa-2x", style={'color': '#3498db'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{total_posts:,}", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P("Total Posts", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(116, 185, 255, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3),
            
            # Data Sources Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-database fa-2x", style={'color': '#2ecc71'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{total_files}", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P("Data Sources", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #55efc4 0%, #00b894 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(85, 239, 196, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3),
            
            # Average Engagement Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-heart fa-2x", style={'color': '#e17055'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{avg_engagement:.1f}", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P("Avg Engagement", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(253, 121, 168, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3),
            
            # High Influence Accounts Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-star fa-2x", style={'color': '#fdcb6e'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{high_influence_count:,}", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P("High Influence", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(253, 203, 110, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3)
        ], className="mb-4"),
        
        # Second row of KPI cards with spam detection
        dbc.Row([
            # Spam Detection Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-shield-alt fa-2x", style={'color': '#e74c3c'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{spam_percentage:.1f}%", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P(f"Spam Detected ({spam_count})", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #ff7675 0%, #d63031 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(255, 118, 117, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3),
            
            # Quality Score Card
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-check-circle fa-2x", style={'color': '#00b894'}),
                            ], style={'textAlign': 'center', 'marginBottom': '10px'}),
                            html.H2(f"{100 - spam_percentage:.1f}%", className="text-center mb-0", style={'color': '#2c3e50', 'fontWeight': '700'}),
                            html.P("Quality Score", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ])
                    ], style={'padding': '20px'})
                ], style={
                    'background': 'linear-gradient(135deg, #00b894 0%, #00a085 100%)',
                    'border': 'none',
                    'borderRadius': '15px',
                    'boxShadow': '0 8px 25px rgba(0, 184, 148, 0.3)',
                    'color': 'white',
                    'height': '140px'
                })
            ], width=3)
        ], className="mb-4")
        
        data_overview = kpi_cards
        
        # Hashtag Analysis (using filtered data)
        print("Starting hashtag analysis...")
        all_hashtags = []
        text_columns = []
        
        # Find text columns in filtered dataset
        print(f"Analyzing hashtags in filtered dataset with {len(filtered_df)} rows...")
        for col in filtered_df.columns:
            if filtered_df[col].dtype == 'object':
                text_columns.append(col)
                text_series = filtered_df[col].dropna()
                print(f"Processing {len(text_series)} text entries in column '{col}'")
                
                for idx, text in enumerate(text_series):
                    if idx % 100 == 0:  # Log progress every 100 entries
                        print(f"Processing text entry {idx+1}/{len(text_series)}")
                    
                    hashtags = extract_hashtags(str(text))
                    all_hashtags.extend(hashtags)
        
        # Create hashtag chart
        print("Creating hashtag chart...")
        hashtag_counts = Counter(all_hashtags)
        print(f"Total unique hashtags found: {len(hashtag_counts)}")
        
        if all_hashtags:
            top_hashtags = dict(hashtag_counts.most_common(20))
            print(f"Top hashtags: {list(top_hashtags.keys())[:5]}...")  # Show first 5
            
            hashtag_fig = px.bar(
                x=list(top_hashtags.values()),
                y=list(top_hashtags.keys()),
                orientation='h',
                title="Top 20 Hashtags",
                labels={'x': 'Count', 'y': 'Hashtags'}
            )
            hashtag_fig.update_layout(height=600)
        else:
            print("No hashtags found, creating empty chart")
            hashtag_fig = go.Figure()
            hashtag_fig.add_annotation(
                text="No hashtags found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
        
        # Ensure sentiment analysis is performed and cached
        print("Ensuring sentiment analysis is available...")
        ensure_sentiment_analysis(combined_df)
        
        # Test basic sentiment analysis
        test_text = "This is a positive message!"
        result = analyze_sentiment_basic(test_text)
        print(f"Basic sentiment test: {result}")
        
        # Use cached sentiment analysis results
        print("Using cached sentiment analysis for pie chart...")
        all_sentiments = cached_ai_results.get('sentiments', [])
        all_emotions = cached_ai_results.get('emotions', [])
        
        print(f"Using {len(all_sentiments)} cached sentiment results for pie chart")
        
        # Create sentiment chart
        print("Creating sentiment chart...")
        sentiment_counts = Counter(all_sentiments)
        print(f"Sentiment distribution: {dict(sentiment_counts)}")
        
        if all_sentiments:
            sentiment_fig = create_sentiment_pie_chart(filtered_df)
        else:
            print("No sentiment data available, creating empty chart")
            sentiment_fig = go.Figure()
            sentiment_fig.add_annotation(
                text="لا توجد بيانات مشاعر متاحة\nيرجى رفع ملف Excel أولاً",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(
                    family="Cairo, Arial, sans-serif",
                    size=16,
                    color="#7f8c8d"
                )
            )
            sentiment_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
            )
        
        # Create source distribution pie chart
        print("Creating source distribution pie chart...")
        source_columns = ['SocialNetwork', 'Source', 'Platform', 'Channel']
        source_data = None
        
        for col in source_columns:
            if col in combined_df.columns:
                source_data = combined_df[col].value_counts()
                print(f"Using column '{col}' for source distribution")
                break
        
        if source_data is None or source_data.empty:
            print("No source data found, creating empty chart")
            source_fig = go.Figure()
            source_fig.add_annotation(
                text="لا توجد بيانات مصادر متاحة\nيرجى رفع ملف Excel يحتوي على عمود المصدر",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(
                    family="Cairo, Arial, sans-serif",
                    size=16,
                    color="#7f8c8d"
                )
            )
            source_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
            )
        else:
            print(f"Creating source pie chart with {len(source_data)} sources")
            
            # Create professional color palette for sources
            source_colors = px.colors.qualitative.Set3[:len(source_data)]
            
            source_fig = px.pie(
                values=source_data.values,
                names=source_data.index,
                title="توزيع مصادر البيانات",
                color_discrete_sequence=source_colors
            )
            
            # Update layout for better Arabic support and professional look
            source_fig.update_layout(
                font=dict(
                    family="Cairo, Arial, sans-serif",
                    size=14,
                    color="#2c3e50"
                ),
                title=dict(
                    text="توزيع مصادر البيانات",
                    x=0.5,
                    font=dict(size=18, weight='bold', color='#2c3e50')
                ),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=12)
                ),
                margin=dict(l=20, r=80, t=60, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Update traces for better appearance
            source_fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(
                    line=dict(color='#ffffff', width=2)
                ),
                pull=[0.02] * len(source_data)  # Slightly separate slices
            )
        print("Source pie chart created successfully")
        
        # Perform AI analysis on uploaded data
        print("Starting AI sentiment analysis...")
        print(f"Starting AI analysis. Text column: {text_column}")
        print(f"Combined DF shape: {combined_df.shape}")
        
        # Extract texts for AI analysis - prioritize by likes and followers
        texts_for_analysis = []
        if text_column and text_column in combined_df.columns:
            # Find likes and followers columns
            likes_col = None
            followers_col = None
            
            for col in combined_df.columns:
                col_lower = col.lower()
                if 'favorite' in col_lower or 'like' in col_lower:
                    likes_col = col
                elif 'follower' in col_lower or 'follow' in col_lower:
                    followers_col = col
            
            # Create a copy for sorting
            df_for_analysis = combined_df[[text_column]].copy()
            df_for_analysis = df_for_analysis.dropna(subset=[text_column])
            df_for_analysis = df_for_analysis[df_for_analysis[text_column].astype(str).str.len() > 10]
            
            # Add engagement score based on available metrics
            engagement_score = 0
            if likes_col and likes_col in combined_df.columns:
                likes_data = pd.to_numeric(combined_df[likes_col], errors='coerce').fillna(0)
                df_for_analysis['likes'] = likes_data[:len(df_for_analysis)]
                engagement_score += df_for_analysis['likes']
            
            if followers_col and followers_col in combined_df.columns:
                followers_data = pd.to_numeric(combined_df[followers_col], errors='coerce').fillna(0)
                df_for_analysis['followers'] = followers_data[:len(df_for_analysis)]
                engagement_score += df_for_analysis['followers'] * 0.1  # Weight followers less than likes
            
            # If no engagement metrics found, use original order
            if likes_col or followers_col:
                df_for_analysis['engagement_score'] = engagement_score
                df_for_analysis = df_for_analysis.sort_values('engagement_score', ascending=False)
                print(f"Sorted tweets by engagement (likes: {likes_col}, followers: {followers_col})")
            else:
                print("No engagement metrics found, using original order")
            
            # Get top 100 tweets by engagement for analysis
            sample_texts = df_for_analysis[text_column].head(100).tolist()
            texts_for_analysis = [text for text in sample_texts if len(str(text).strip()) > 10]
            print(f"Selected {len(texts_for_analysis)} top engagement tweets for AI analysis")
        
        print(f"Texts for analysis count: {len(texts_for_analysis)}")
        
        if texts_for_analysis:
            print("About to call AI analysis...")
            try:
                # Perform batch AI sentiment analysis with larger batch size for better performance
                print("Performing AI sentiment analysis...")
                ai_results = analyze_sentiment_batch_with_ai(texts_for_analysis, batch_size=10)
                
                # Extract results from list of tuples: [(sentiment, confidence, emotions_dict), ...]
                all_texts = texts_for_analysis
                all_sentiments = [result[0] for result in ai_results]  # Extract sentiment from each tuple
                all_emotions = [result[2] for result in ai_results]    # Extract emotions dict from each tuple
                
                # Extract positive and negative tweets (maintaining engagement order)
                # Since all_texts are already sorted by engagement, preserve this order
                positive_tweets = [text for text, sentiment in zip(all_texts, all_sentiments) if sentiment == 'positive'][:10]
                negative_tweets = [text for text, sentiment in zip(all_texts, all_sentiments) if sentiment == 'negative'][:10]
                
                print(f"Selected top {len(positive_tweets)} positive and {len(negative_tweets)} negative tweets by engagement")
                
                # Generate AI summary
                print("Generating AI summary...")
                summary = generate_ai_summary(all_texts, all_sentiments, all_emotions, all_hashtags[:20])
                
                # Cache the results
                cached_ai_results.update({
                    'texts': all_texts,
                    'sentiments': all_sentiments,
                    'emotions': all_emotions,
                    'positive_tweets': positive_tweets,
                    'negative_tweets': negative_tweets,
                    'summary': summary
                })
                
                print(f"AI analysis completed: {len(all_sentiments)} sentiments analyzed")
                
            except Exception as e:
                print(f"AI analysis failed with error: {e}")
                print(f"Falling back to basic summary generation...")
                # Use cached results or empty results
                all_texts = cached_ai_results.get('texts', [])
                all_sentiments = cached_ai_results.get('sentiments', [])
                all_emotions = cached_ai_results.get('emotions', [])
                positive_tweets = cached_ai_results.get('positive_tweets', [])
                negative_tweets = cached_ai_results.get('negative_tweets', [])
                
                # Generate basic summary as fallback
                basic_summary = generate_summary(
                    texts_for_analysis if texts_for_analysis else [],
                    all_sentiments if all_sentiments else [],
                    all_emotions if all_emotions else []
                )
                cached_ai_results['summary'] = basic_summary
                print(f"Generated fallback summary: {basic_summary[:100]}...")
        else:
            print("No suitable texts found for AI analysis")
            # Use cached results or empty results
            all_texts = cached_ai_results.get('texts', [])
            all_sentiments = cached_ai_results.get('sentiments', [])
            all_emotions = cached_ai_results.get('emotions', [])
            positive_tweets = cached_ai_results.get('positive_tweets', [])
            negative_tweets = cached_ai_results.get('negative_tweets', [])
        
        # Create emotion analysis chart
        emotion_fig = go.Figure()
        if all_emotions:
            emotion_totals = {'joy': 0, 'anger': 0, 'fear': 0, 'sadness': 0, 'surprise': 0, 'trust': 0}
            for emotions in all_emotions:
                for emotion, count in emotions.items():
                    emotion_totals[emotion] += count
            
            emotion_fig = px.bar(
                x=list(emotion_totals.keys()),
                y=list(emotion_totals.values()),
                title="Emotion Distribution",
                labels={'x': 'Emotions', 'y': 'Count'},
                color=list(emotion_totals.values()),
                color_continuous_scale='viridis'
            )
        else:
            emotion_fig.add_annotation(
                text="لا توجد بيانات مشاعر متاحة\nيرجى رفع ملف Excel أولاً",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(
                    family="Cairo, Arial, sans-serif",
                    size=16,
                    color="#7f8c8d"
                )
            )
            emotion_fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
            )
        
        # Generate time series chart based on selected period
        # Time series chart is now handled by separate callback
        
        # Use cached AI results if available
        if cached_ai_results['summary'] is not None:
            summary_content = dcc.Markdown(cached_ai_results['summary'])
            positive_content = html.Div([
                html.P(f"Found {len(cached_ai_results['positive_tweets'])} positive tweets:", className="mb-3"),
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.P(tweet, className="mb-0")
                        ])
                    ], className="mb-2") for tweet in cached_ai_results['positive_tweets'][:10]
                ])
            ]) if cached_ai_results['positive_tweets'] else html.P("No positive tweets found.", className="text-muted")
            
            negative_content = html.Div([
                html.P(f"Found {len(cached_ai_results['negative_tweets'])} negative tweets:", className="mb-3"),
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.P(tweet, className="mb-0")
                        ])
                    ], className="mb-2") for tweet in cached_ai_results['negative_tweets'][:10]
                ])
            ]) if cached_ai_results['negative_tweets'] else html.P("No negative tweets found.", className="text-muted")
        else:
            # Generate basic summary even if AI analysis failed
            basic_summary = generate_summary(
                texts_for_analysis if texts_for_analysis else [],
                all_sentiments if all_sentiments else [],
                all_emotions if all_emotions else []
            )
            summary_content = dcc.Markdown(basic_summary)
            positive_content = html.P("No positive tweets to display.", className="text-muted")
            negative_content = html.P("No negative tweets to display.", className="text-muted")
            print(f"Using basic summary: {basic_summary[:100]}...")
        
        # Emergency fallback if summary_content is still empty or None
        if summary_content is None or (hasattr(summary_content, 'children') and not summary_content.children):
            emergency_summary = create_emergency_summary(filtered_df)
            summary_content = dcc.Markdown(emergency_summary)
            print("Using emergency summary as fallback")
        
        # Create engagement scatter plot
        engagement_fig = create_engagement_scatter(filtered_df)
        
        # Calculate engagement metrics
        engagement_metrics = calculate_engagement_metrics(filtered_df)
        max_tweet_per_minute = engagement_metrics['max_tweet_per_minute']
        
        # Update KPI cards with engagement metrics
        updated_kpi_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-chart-line fa-2x mb-2", style={'color': '#3498db'}),
                            html.H4(f"{engagement_metrics['max_tweet_per_minute']}", className="text-center mb-1", style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                            html.P("أقصى تغريدة/دقيقة", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #3498db, #2980b9)', 'color': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(52, 152, 219, 0.3)'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-heart fa-2x mb-2", style={'color': '#e74c3c'}),
                            html.H4(f"{engagement_metrics['likes']}", className="text-center mb-1", style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                            html.P("إجمالي الإعجابات", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #e74c3c, #c0392b)', 'color': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(231, 76, 60, 0.3)'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-retweet fa-2x mb-2", style={'color': '#27ae60'}),
                            html.H4(f"{engagement_metrics['retweets']}", className="text-center mb-1", style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                            html.P("إجمالي إعادة التغريد", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #27ae60, #229954)', 'color': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(39, 174, 96, 0.3)'})
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-users fa-2x mb-2", style={'color': '#f39c12'}),
                            html.H4(f"{engagement_metrics['unique_users']}", className="text-center mb-1", style={'color': '#2c3e50', 'fontWeight': 'bold'}),
                            html.P("المستخدمون الفريدون", className="text-center text-muted mb-0", style={'fontSize': '0.9rem'})
                        ], style={'textAlign': 'center'})
                    ])
                ], style={'background': 'linear-gradient(135deg, #f39c12, #e67e22)', 'color': 'white', 'borderRadius': '15px', 'boxShadow': '0 4px 15px rgba(243, 156, 18, 0.3)'})
            ], width=3)
        ], className="mb-4")
        
        print("Callback processing completed successfully!")
        # Create new visualizations
        places_bar_fig = create_places_bar_chart(filtered_df)
        tweets_timeline_fig = create_tweets_timeline_chart(filtered_df)
        
        # Calculate tweets per minute and create gauge
        engagement_metrics = calculate_engagement_metrics(filtered_df)
        max_tweets_per_minute = engagement_metrics.get('max_tweet_per_minute', 0)
        tweets_per_minute_level, level_text = analyze_tweets_per_minute_level(max_tweets_per_minute)
        tweets_per_minute_gauge = create_tweets_per_minute_gauge(tweets_per_minute_level)
        
        # Combine comprehensive KPI cards (data_overview includes spam detection)
        comprehensive_kpi_cards = html.Div([
            data_overview,  # First row with spam detection
            updated_kpi_cards  # Second row with engagement metrics
        ])
        
        return (
            dbc.Alert(f"Successfully uploaded {len(contents)} files! Showing {len(filtered_df)} rows after filtering.", color="success"),
            comprehensive_kpi_cards,
            hashtag_fig,
            sentiment_fig,
            source_fig,
            emotion_fig,
            engagement_fig,
            places_bar_fig,
            tweets_timeline_fig,
            summary_content,
            social_network_options,
            engagement_options,
            positive_content,
            negative_content,

        )
        
    except Exception as e:
        # Error KPI cards
        error_kpi_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.I(className="fas fa-exclamation-circle fa-2x mb-2", style={'color': '#e74c3c'}),
                            html.H4("Processing Error", className="text-center", style={'color': '#7f8c8d'}),
                            html.P("Error occurred during data processing", className="text-center text-muted mb-0")
                        ], style={'textAlign': 'center', 'padding': '20px'})
                    ])
                ], style={
                    'background': '#f8d7da',
                    'border': '2px solid #f5c6cb',
                    'borderRadius': '15px',
                    'height': '140px'
                })
            ], width=12)
        ], className="mb-4")
        
        return (
            dbc.Alert(f"Error processing files: {str(e)}", color="danger"),
            error_kpi_cards,
            go.Figure(),
            go.Figure(),
            go.Figure(),  # source chart
            go.Figure(),  # emotion chart
            go.Figure(),  # engagement scatter
            go.Figure(),  # places bar chart
            go.Figure(),  # tweets timeline chart
            html.P("No data available for summary."),  # analysis summary
            [],  # social network options
            [],  # engagement options
            html.P("No positive tweets to display."),  # positive tweets
            html.P("No negative tweets to display.")   # negative tweets
        )


if __name__ == '__main__':
    import os
    print("Starting Dash Arabic Hashtag & Sentiment Analysis Dashboard...")
    
    # Test Gemini API at startup
    test_gemini_api()
    
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 8050))
    
    # Use 0.0.0.0 for production hosting
    host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    
    print(f"Dashboard will be available at: http://{host}:{port}")
    app.run_server(debug=False, host=host, port=port)
