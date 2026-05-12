"""
Движок рекомендательной системы вин
ML логика, функции и обработка данных
"""

import pandas as pd
import numpy as np
import random
import os
import sqlite3
from datetime import datetime, timedelta
from collections import Counter
import re

# ==================== ЗАГРУЗКА ДАННЫХ ====================
WINE_CSV = "winemag-data_first150k.csv"
DB_FILE = "reviews.db"

print("[Engine] Загружаем данные о винах...")
try:
    DF = pd.read_csv(WINE_CSV)
    print(f"[Engine] Загружено вин: {len(DF)}")
except Exception as e:
    print(f"[Engine] Ошибка загрузки: {e}")
    DF = pd.DataFrame()

# ==================== ПОДГОТОВКА ДАННЫХ ====================
def prepare_data():
    """Подготовка данных: конвертация полей"""
    global DF
    
    if len(DF) == 0:
        return
    
    # Переименование колонок
    df = DF.copy()
    df = df.rename(columns={
        'description': 'text',
        'variety': 'product'
    })
    
    # Конвертация points в rating (1-5) и sentiment (0/1)
    def points_to_rating(points):
        if pd.isna(points):
            return 3
        if points < 82:
            return 1
        elif points < 86:
            return 2
        elif points < 90:
            return 3
        elif points < 95:
            return 4
        else:
            return 5
    
    def points_to_sentiment(points):
        if pd.isna(points):
            return 1
        return 1 if points >= 90 else 0
    
    df['rating'] = df['points'].apply(points_to_rating)
    df['sentiment'] = df['points'].apply(points_to_sentiment)
    
    # Генерация дат (случайные за последние 2 года)
    np.random.seed(42)
    base_date = datetime.now()
    df['date'] = [base_date - timedelta(days=np.random.randint(0, 730)) 
                  for _ in range(len(df))]
    
    # Генерация user_id из taster_name или случайный
    if 'taster_name' in df.columns and df['taster_name'].notna().any():
        df['user_id'] = df['taster_name'].fillna('Anonymous').factorize()[0]
        df['user_name'] = df['taster_name'].fillna('Anonymous')
    else:
        df['user_id'] = np.random.randint(1, 101, size=len(df))
        df['user_name'] = [f"Пользователь_{i}" for i in df['user_id']]
    
    # Очистка от null в важных полях
    df = df.dropna(subset=['text', 'rating', 'product'])
    
    DF = df
    print(f"[Engine] Подготовлено записей: {len(DF)}")
    return DF


# ==================== SENTIMENT ANALYSIS ====================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

VECTORIZER = None
MODELS = {}
METRICS = None
BEST_MODEL = None

def train_sentiment_models():
    """Обучение 3 моделей классификации"""
    global VECTORIZER, MODELS, METRICS, BEST_MODEL
    
    if len(DF) == 0:
        prepare_data()
    
    print("[Engine] Обучаем модели классификации...")
    
    # Сэмплируем для скорости
    df_sample = DF.sample(n=min(20000, len(DF)), random_state=42)
    
    X = df_sample['text'].values
    y = df_sample['sentiment'].values
    
    VECTORIZER = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
    X_vec = VECTORIZER.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)
    
    # 3 модели
    models_dict = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    best_f1 = 0
    
    for name, model in models_dict.items():
        print(f"  Обучаем {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }
        
        if f1 > best_f1:
            best_f1 = f1
            BEST_MODEL = model
            BEST_NAME = name
            BEST_Y_TEST = y_test
            BEST_Y_PRED = y_pred
        
        MODELS[name] = model
    
    METRICS = pd.DataFrame(results).T
    print(f"[Engine] Лучшая модель: {BEST_NAME} (F1={best_f1:.4f})")
    return METRICS


def predict_sentiment(text):
    """Предсказание тональности с % уверенности"""
    global VECTORIZER, BEST_MODEL
    
    if VECTORIZER is None or BEST_MODEL is None:
        train_sentiment_models()
    
    X_new = VECTORIZER.transform([text])
    pred = BEST_MODEL.predict(X_new)[0]
    proba = BEST_MODEL.predict_proba(X_new)[0][int(pred)]
    
    label = 'позитивный' if pred == 1 else 'негативный'
    return f"{label} ({proba*100:.1f}%)", pred, proba


# ==================== АНАЛИТИКА ====================
def top_products_by_positive_ratio(n=10):
    """Топ товаров по доле позитивных отзывов"""
    if len(DF) == 0:
        prepare_data()
    
    product_stats = DF.groupby('product').agg({
        'sentiment': ['mean', 'count']
    }).round(3)
    product_stats.columns = ['positive_ratio', 'reviews_count']
    product_stats = product_stats[product_stats['reviews_count'] >= 30]
    return product_stats.nlargest(n, 'positive_ratio')


def top_negative_words(n=5):
    """Топ-N слов в негативных отзывах"""
    if len(DF) == 0:
        prepare_data()
    
    negative_texts = DF[DF['sentiment'] == 0]['text'].str.lower()
    words = ' '.join(negative_texts).split()
    
    stop_words = {'the', 'and', 'of', 'a', 'in', 'is', 'with', 'this', 'that', 'it', 
                  'for', 'wine', 'but', 'not', 'has', 'have', 'from', 'are', 'was',
                  'on', 'to', 'is', 'an', 'as', 'be', 'by', 'at'}
    words = [w for w in words if len(w) > 3 and w not in stop_words and w.isalpha()]
    
    return Counter(words).most_common(n)


def get_wine_stats():
    """Статистика по винам"""
    if len(DF) == 0:
        prepare_data()
    
    stats = DF.groupby('product').agg({
        'rating': ['mean', 'count'],
        'sentiment': 'mean'
    }).round(2)
    stats.columns = ['avg_rating', 'review_count', 'positive_ratio']
    stats = stats[stats['review_count'] >= 10].sort_values('avg_rating', ascending=False)
    return stats.head(50)


# ==================== ВРЕМЕННЫЕ РЯДЫ ====================
from sklearn.linear_model import LinearRegression

def forecast_reviews(days=7):
    """Прогнозирование количества отзывов"""
    if len(DF) == 0:
        prepare_data()
    
    daily = DF.groupby(DF['date'].dt.date).size().reset_index(name='count')
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    
    if len(daily) < 10:
        return [3] * days
    
    for lag in [1, 2, 3, 7]:
        daily[f'lag_{lag}'] = daily['count'].shift(lag)
    
    daily = daily.dropna()
    
    if len(daily) < 10:
        return [3] * days
    
    X = daily[['lag_1', 'lag_2', 'lag_3', 'lag_7']].values
    y = daily['count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Прогноз
    last = daily.tail(7)['count'].values
    predictions = []
    
    for _ in range(days):
        features = np.array([[last[-1], last[-2], last[-3], last[-7] if len(last) >= 7 else last[-1]]])
        pred = max(1, model.predict(features)[0])
        predictions.append(int(pred))
        last = np.append(last[1:], pred)
    
    return predictions


# ==================== РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ====================
from sklearn.metrics.pairwise import cosine_similarity

# User-based collaborative filtering
def get_user_recommendations(user_id, top_n=5):
    """Персональные рекомендации для пользователя"""
    if len(DF) == 0:
        prepare_data()
    
    print(f"[Recommendations] User ID: {user_id}")
    
    # Проверяем отзывы пользователя в БД
    user_reviews_df = get_user_reviews(user_id)
    print(f"[Recommendations] Found {len(user_reviews_df)} user reviews in DB")
    
    if len(user_reviews_df) > 0:
        print(f"[Recommendations] User products: {list(user_reviews_df['product'].unique())[:5]}")
        
        user_rated_products = set(user_reviews_df['product'].unique())
        user_high_rated = user_reviews_df[user_reviews_df['rating'] >= 4]
        
        if len(user_high_rated) > 0:
            # Получаем топ вина из основной базы (которых пользователь не оценивал)
            top_wines = DF[~DF['product'].isin(user_rated_products)]
            
            # Группируем и сортируем по рейтингу
            top_wines_grouped = top_wines.groupby('product').agg({
                'rating': 'mean',
                'sentiment': 'mean'
            }).reset_index()
            
            top_wines_grouped = top_wines_grouped.sort_values(['rating', 'sentiment'], ascending=[False, False])
            
            # Возвращаем топ вина с хорошими оценками
            top_products = top_wines_grouped[top_wines_grouped['rating'] >= 4]['product'].head(top_n).tolist()
            print(f"[Recommendations] Returning: {top_products}")
            
            if len(top_products) >= top_n:
                return top_products
    
    # Новый пользователь или нет похожих - популярные вина
    popular = DF.groupby('product').agg({'rating': 'mean', 'sentiment': 'mean'}).reset_index()
    popular = popular.sort_values(['rating', 'sentiment'], ascending=[False, False])
    result = list(popular.head(top_n)['product'])
    print(f"[Recommendations] Returning popular: {result[:3]}")
    return result


# ==================== ЗАЩИТА ПЕРСОНАЛЬНЫХ ДАННЫХ ====================
def anonymize_data(data):
    """Маскирование персональных данных"""
    df = data.copy()
    
    if 'user_name' in df.columns:
        df['user_name'] = df['user_name'].apply(lambda x: x[:2] + '***' + x[-2:] if len(str(x)) > 4 else 'U***')
    
    # Генерируем анонимные user_id для отображения
    if 'user_id' in df.columns:
        df['anon_id'] = 'User_' + (df['user_id'] % 1000).astype(str)
    
    return df


# ==================== РАБОТА С SQLite ====================
def init_db():
    """Инициализация базы данных SQLite"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        product TEXT,
        text TEXT,
        rating INTEGER,
        sentiment INTEGER,
        date TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS user_sessions (
        session_id TEXT PRIMARY KEY,
        user_id TEXT,
        created_at TEXT
    )''')
    
    conn.commit()
    conn.close()


def save_review(user_id, product, text, rating, sentiment):
    """Сохранение отзыва в SQLite"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''INSERT INTO reviews (user_id, product, text, rating, sentiment, date) 
                  VALUES (?, ?, ?, ?, ?, ?)''',
              (user_id, product, text, rating, sentiment, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()


def get_user_reviews(user_id):
    """Получение отзывов пользователя"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM reviews WHERE user_id = ? ORDER BY date DESC", (str(user_id),))
    rows = c.fetchall()
    conn.close()
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows, columns=['id', 'user_id', 'product', 'text', 'rating', 'sentiment', 'date'])


# ==================== РЕНДЕРИНГ HTML ====================
def render_result_card(i, r):
    """Рендеринг карточки вина"""
    rating_stars = '★' * r.get('rating', 3) + '☆' * (5 - r.get('rating', 3))
    country = r.get('country', 'Unknown')
    price = r.get('price', 0)
    price_str = f"${price:.2f}" if price > 0 else "N/A"
    description = r.get('description', '')[:60] + '...' if r.get('description') else ''
    region = r.get('region', '')
    
    return f'''
    <div class="wine-card">
        <div class="wine-header">
            <span class="wine-number">#{i}</span>
            <span class="wine-rating">{rating_stars}</span>
        </div>
        <div class="wine-icon"></div>
        <div class="wine-name">{r.get('name', 'Unknown')}</div>
        <div class="wine-info">{r.get('product', 'Unknown')}</div>
        <div class="wine-country">{country}{" | " + region if region else ""}</div>
        <div class="wine-price">{price_str}</div>
        <div class="wine-description">{description}</div>
        <span class="match-score">Совм: {r.get('score', 0):.0f}%</span>
    </div>
    '''


def search_wines(query, filters=None):
    """Поиск вин по запросу"""
    if len(DF) == 0:
        prepare_data()
    
    results = DF.copy()
    
    # Поиск по названию, сорту, стране
    if query:
        mask = (results['product'].str.lower().str.contains(query.lower(), na=False) |
                results.get('country', '').str.lower().str.contains(query.lower(), na=False) |
                results.get('winery', '').str.lower().str.contains(query.lower(), na=False))
        results = results[mask]
    
    # Фильтры
    if filters:
        if 'min_rating' in filters:
            results = results[results['rating'] >= filters['min_rating']]
        if 'country' in filters and filters['country']:
            results = results[results['country'] == filters['country']]
        if 'max_price' in filters:
            results = results[results['price'] <= filters['max_price']]
    
    return results.head(20)


def recommend_wines(query, top_k=5):
    """Рекомендация вин по описанию"""
    if len(DF) == 0:
        prepare_data()
    
    query_lower = query.lower()
    scores = []
    
    # Ключевые слова для типов вин
    type_keywords = {
        'красное': ['red', 'cabernet', 'merlot', 'pinot noir', 'syrah', 'shiraz'],
        'белое': ['white', 'chardonnay', 'sauvignon', 'riesling', 'pinot grigio'],
        'розовое': ['rosé', 'rose', 'pink'],
        'игристое': ['sparkling', 'champagne', 'prosecco'],
    }
    
    for _, row in DF.iterrows():
        score = 0
        text_lower = str(row.get('text', '')).lower()
        
        # Тип вина
        for wine_type, keywords in type_keywords.items():
            if wine_type in query_lower:
                if any(kw in text_lower for kw in keywords):
                    score += 30
        
        # Рейтинг
        score += row.get('rating', 3) * 8
        
        # Цена (бюджет)
        if 'дешевле' in query_lower or 'недорогой' in query_lower:
            if row.get('price', 100) < 20:
                score += 20
        elif 'дорогой' in query_lower or 'премиум' in query_lower:
            if row.get('price', 0) > 50:
                score += 20
        
        # Страна
        for country in ['france', 'italy', 'spain', 'usa', 'argentina']:
            if country in query_lower and country in str(row.get('country', '')).lower():
                score += 15
        
        # Случайный фактор
        score += random.uniform(0, 10)
        
        scores.append((row, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {
            'name': row['winery'] if pd.notna(row.get('winery')) else row['product'],
            'product': row['product'],
            'country': row.get('country', 'Unknown'),
            'price': row.get('price', 0) if pd.notna(row.get('price')) else 0,
            'rating': row['rating'],
            'score': score,
            'description': row.get('text', '')[:100],
            'region': row.get('region', '')
        }
        for row, score in scores[:top_k]
    ]


def render_wine_results(results):
    """Рендеринг результатов поиска"""
    if not results:
        return '<div class="no-results"><p>Введите запрос, чтобы найти вина</p></div>'
    
    html = ''
    for i, r in enumerate(results, 1):
        html += render_result_card(i, r)
    return html


def render_stats_html():
    """Рендеринг статистики"""
    stats = get_wine_stats()
    
    html = '<table class="stats-table"><thead><tr><th>Вино</th><th>Средний рейтинг</th><th>Отзывов</th><th>Доля позитива</th></tr></thead><tbody>'
    
    for idx, row in stats.iterrows():
        html += f'<tr><td><strong>{idx}</strong></td><td>{row["avg_rating"]:.1f}/5</td><td>{int(row["review_count"])}</td><td>{row["positive_ratio"]*100:.0f}%</td></tr>'
    
    html += '</tbody></table>'
    return html


def render_top_positive():
    """Топ вин по позитиву"""
    top = top_products_by_positive_ratio(10)
    
    html = '<div class="top-list">'
    for i, (product, row) in enumerate(top.iterrows(), 1):
        html += f'<div class="top-item"><span class="top-num">{i}</span><span class="top-name">{product}</span><span class="top-score">{row["positive_ratio"]*100:.0f}%</span></div>'
    html += '</div>'
    return html


# ==================== ИНИЦИАЛИЗАЦИЯ ====================
init_db()
if len(DF) > 0:
    prepare_data()