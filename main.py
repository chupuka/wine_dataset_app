"""
Wine Recommendation System
Flask Web Interface
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from flask import Flask, request, session, jsonify
import uuid

# ==================== FLASK APP ====================
app = Flask(__name__)
app.secret_key = 'wine-recommendation-secret-key'

# ==================== ЗАГРУЗКА ДВИЖКА ====================
from engine import (
    prepare_data, train_sentiment_models, predict_sentiment,
    top_products_by_positive_ratio, top_negative_words, get_wine_stats,
    forecast_reviews, get_user_recommendations, anonymize_data,
    save_review, get_user_reviews, search_wines, recommend_wines,
    render_wine_results, render_stats_html, render_top_positive
)

# ==================== INITIALIZATION ====================
print("\n" + "="*50)
print("> Initializing system...")
print("="*50)

prepare_data()
train_sentiment_models()

print("\nSystem ready!")

# ==================== HELPER FUNCTIONS ====================

def get_current_user():
    """Get current user ID"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    return session['user_id']


def render_page(results_html='', sentiment_result='', stats_html='', 
                top_positive_html='', forecast_html='', recommendations_html=''):
    """Render page with template replacement"""
    user_id = get_current_user()
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            page = f.read()
    except:
        page = get_default_page()
    
    page = page.replace('{{results}}', results_html)
    page = page.replace('{{sentiment_result}}', sentiment_result)
    page = page.replace('{{stats_html}}', stats_html)
    page = page.replace('{{top_positive}}', top_positive_html)
    page = page.replace('{{forecast}}', forecast_html)
    page = page.replace('{{recommendations}}', recommendations_html)
    page = page.replace('{{user_id}}', user_id)
    
    return page


def get_default_page():
    """Default page if index.html not found"""
    return '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Wine Analysis System</title>
    <style>
        body { font-family: Arial; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; }
        h1 { color: #722f37; }
        .search-box { display: flex; gap: 10px; margin: 20px 0; }
        .search-box input { flex: 1; padding: 12px; font-size: 16px; }
        .search-box button { padding: 12px 24px; background: #722f37; color: white; border: none; cursor: pointer; }
        .tabs { display: flex; gap: 10px; margin: 20px 0; border-bottom: 1px solid #ddd; }
        .tab { padding: 10px 20px; cursor: pointer; border: none; background: none; }
        .tab.active { border-bottom: 2px solid #722f37; font-weight: bold; }
        .content { display: none; }
        .content.active { display: block; }
        .wine-card { display: inline-block; width: 200px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; vertical-align: top; }
        .wine-name { font-weight: bold; margin: 10px 0; }
        .wine-info { font-size: 12px; color: #666; }
        .sentiment-box { padding: 15px; margin: 15px 0; border-radius: 8px; }
        .sentiment-positive { background: #d4edda; }
        .sentiment-negative { background: #f8d7da; }
    </style>
</head>
<body>
    <div class="container">
        <<h1>🍷 Wine Recommendation System</h1>
        <p>Review analysis & personalized recommendations</p>
        
        <form method="POST" action="/search">
            <div class="search-box">
                <input type="text" name="query" placeholder="Search wines (red, France, cheap...)">
                <button type="submit">Search</button>
            </div>
        </form>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('search', this)">Search</button>
            <button class="tab" onclick="showTab('sentiment', this)">Sentiment</button>
            <button class="tab" onclick="showTab('stats', this)">Statistics</button>
            <button class="tab" onclick="showTab('recommend', this)">Recommendations</button>
        </div>
        
        <div class="content active" id="search">{{results}}</div>
        <div class="content" id="sentiment">
            <h3>Review Sentiment Analysis</h3>
            <form method="POST" action="/analyze">
                <textarea name="text" rows="4" style="width:100%;padding:10px;" placeholder="Enter review text..."></textarea>
                <button type="submit" class="search-box button">Analyze</button>
            </form>
            {{sentiment_result}}
        </div>
        <div class="content" id="stats">{{stats_html}}</div>
        <div class="content" id="recommend">{{recommendations}}</div>
    </div>
    <script>
        function showTab(id, btn) {
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            btn.classList.add('active');
        }
    </script>
</body>
</html>'''


# ==================== ROUTES ====================

@app.route('/', methods=['GET', 'POST'])
def index():
    results_html = render_wine_results([])
    stats_html = render_stats_html()
    top_positive_html = render_top_positive()
    
    # Прогноз
    forecast = forecast_reviews(7)
    forecast_html = '<div class="forecast"><h4>7-day forecast:</h4><p>' + ' → '.join(map(str, forecast)) + '</p></div>'
    
    # Recommendations for current user
    user_id = get_current_user()
    recs = get_user_recommendations(user_id, 5)
    recommendations_html = '<div class="recommendations-grid">'
    for i, r in enumerate(recs, 1):
        recommendations_html += f'''
        <div class="wine-card">
            <div class="wine-header">
                <span class="wine-number">#{i}</span>
                <span class="wine-rating">★★★★☆</span>
            </div>
            <div class="wine-icon">🍷</div>
            <div class="wine-name">{r}</div>
            <div class="wine-info">Personal pick for you</div>
            <span class="match-score">Based on your preferences</span>
        </div>
        '''
    recommendations_html += '</div>'
    
    return render_page(results_html, '', stats_html, top_positive_html, forecast_html, recommendations_html)


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.form.get('query', '')
    
    if query:
        results = recommend_wines(query, 10)
        results_html = render_wine_results(results)
    else:
        results_html = render_wine_results([])
    
    stats_html = render_stats_html()
    top_positive_html = render_top_positive()
    
    return render_page(results_html, '', stats_html, top_positive_html)


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.form.get('text', '')
    
    if text:
        result, pred, proba = predict_sentiment(text)
        
        if pred == 1:
            sentiment_class = 'sentiment-positive'
            icon = '✅'
        else:
            sentiment_class = 'sentiment-negative'
            icon = '❌'
        
        sentiment_html = f'''
        <div class="sentiment-box {sentiment_class}">
            <h3>{icon} Result: {result}</h3>
            <p>Text: "{text[:100]}..."</p>
        </div>
        '''
        
        # Save review
        user_id = get_current_user()
        product = 'Unknown'
        rating = 3
        save_review(user_id, product, text, rating, pred)
    else:
        sentiment_html = '<p>Enter text to analyze</p>'
    
    stats_html = render_stats_html()
    top_positive_html = render_top_positive()
    
    return render_page('', sentiment_html, stats_html, top_positive_html)


@app.route('/add_review', methods=['POST'])
def add_review():
    product = request.form.get('product', '')
    text = request.form.get('text', '')
    rating = int(request.form.get('rating', 3))
    
    if product and text:
        pred, sentiment, _ = predict_sentiment(text)
        user_id = get_current_user()
        save_review(user_id, product, text, rating, sentiment)
        
        review_message = f'<p style="color:green;margin-top:15px;"><strong>Review added for {product}!</strong></p>'
    else:
        review_message = '<p style="color:red;">Please fill all fields</p>'
    
    stats_html = render_stats_html()
    top_positive_html = render_top_positive()
    
    return render_page('', review_message, stats_html, top_positive_html)


@app.route('/recommendations')
def recommendations():
    user_id = get_current_user()
    recs = get_user_recommendations(user_id, 10)
    
    results = []
    for prod in recs:
        results.append({
            'name': prod, 
            'product': prod, 
            'rating': 4, 
            'score': 90,
            'description': 'Recommended based on similar users',
            'region': ''
        })
    
    results_html = render_wine_results(results)
    stats_html = render_stats_html()
    top_positive_html = render_top_positive()
    
    # Recommendations for current user
    recommendations_html = '<div class="recommendations-grid">'
    for i, r in enumerate(recs, 1):
        recommendations_html += f'''
        <div class="wine-card">
            <div class="wine-header">
                <span class="wine-number">#{i}</span>
                <span class="wine-rating">★★★★☆</span>
            </div>
            <div class="wine-icon">🍷</div>
            <div class="wine-name">{r}</div>
            <div class="wine-info">Personal pick for you</div>
            <span class="match-score">Based on your preferences</span>
        </div>
        '''
    recommendations_html += '</div>'
    
    return render_page(results_html, '', stats_html, top_positive_html, '', recommendations_html)


# ==================== API ENDPOINTS ====================

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    data = request.json
    text = data.get('text', '')
    
    result, pred, proba = predict_sentiment(text)
    
    return jsonify({
        'result': result,
        'sentiment': int(pred),
        'confidence': float(proba)
    })


@app.route('/api/recommendations')
def api_recommendations():
    user_id = get_current_user()
    recs = get_user_recommendations(user_id, 10)
    
    return jsonify({
        'user_id': user_id,
        'recommendations': recs
    })


@app.route('/api/search_wines')
def api_search_wines():
    """API для автодополнения поиска вин"""
    query = request.args.get('q', '').strip().lower()
    
    if len(query) < 2:
        return jsonify({'wines': []})
    
    from engine import DF
    if DF is None or len(DF) == 0:
        return jsonify({'wines': []})
    
    # Проверяем, есть ли колонка rating (данные подготовлены)
    if 'rating' not in DF.columns:
        # Используем оригинальные данные
        search_cols = ['variety', 'country', 'winery']
        if 'variety' in DF.columns:
            mask = (
                DF['variety'].str.lower().str.contains(query, na=False) |
                DF.get('country', pd.Series([], dtype=str)).str.lower().str.contains(query, na=False) |
                DF.get('winery', pd.Series([], dtype=str)).str.lower().str.contains(query, na=False)
            )
        else:
            mask = pd.Series([False] * len(DF))
    else:
        # Данные уже подготовлены
        mask = (
            DF['product'].str.lower().str.contains(query, na=False) |
            DF.get('country', pd.Series()).str.lower().str.contains(query, na=False) |
            DF.get('winery', pd.Series()).str.lower().str.contains(query, na=False)
        )
    
    results = DF[mask].head(20)
    
    wines = []
    for _, row in results.iterrows():
        # Определяем название вина
        if 'winery' in row and pd.notna(row.get('winery')):
            name = row['winery']
        elif 'product' in row:
            name = str(row['product'])
        elif 'variety' in row:
            name = str(row['variety'])
        else:
            name = 'Unknown'
        
        # Определяем сорт
        product = row.get('product', row.get('variety', 'Unknown'))
        
        # Определяем страну
        country = row.get('country', 'Unknown')
        if pd.isna(country):
            country = 'Unknown'
        
        # Определяем рейтинг
        if 'rating' in row and pd.notna(row.get('rating')):
            rating = int(row['rating'])
        elif 'points' in row and pd.notna(row.get('points')):
            rating = int(row['points'] / 20)
        else:
            rating = 3
        
        wines.append({
            'name': name,
            'product': str(product),
            'country': str(country),
            'rating': rating
        })
    
    # Убираем дубликаты по названию
    seen = set()
    unique_wines = []
    for w in wines:
        if w['name'] not in seen:
            seen.add(w['name'])
            unique_wines.append(w)
    
    return jsonify({'wines': unique_wines[:10]})


# ==================== RUN ====================

if __name__ == '__main__':
    print("\nВыберите режим:")
    print("  1 - Консольный режим")
    print("  2 - Веб-интерфейс (Flask)")
    choice = input("Ваш выбор (1/2): ").strip()
    
    if choice == '2':
        print("\n" + "="*50)
        print("> Запуск веб-интерфейса...")
        print("Откройте в браузере: http://127.0.0.1:5000")
        print("="*50)
        app.run(debug=False, port=5000, host='0.0.0.0')
    else:
        print("\nЗапустите снова с выбором 2 для веб-интерфейса")