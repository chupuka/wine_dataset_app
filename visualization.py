"""
Генерация визуализации для отчёта
student_report.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== ЗАГРУЗКА ДАННЫХ ====================
print("Загрузка данных...")
df = pd.read_csv("winemag-data_first150k.csv")
print(f"Загружено записей: {len(df)}")

# ==================== ПОДГОТОВКА ====================
print("Подготовка данных...")

# Переименование
df = df.rename(columns={'description': 'text', 'variety': 'product'})

# Конвертация points в rating и sentiment
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

# Очистка
df = df.dropna(subset=['text', 'rating', 'product'])

print(f"После очистки: {len(df)} записей")

# ==================== SENTIMENT ANALYSIS ====================
print("Обучение моделей...")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Сэмпл для скорости
df_sample = df.sample(n=min(15000, len(df)), random_state=42)

X = df_sample['text'].values
y = df_sample['sentiment'].values

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# 3 модели
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    if results[name]['F1-Score'] > best_f1:
        best_f1 = results[name]['F1-Score']
        best_model = model
        best_pred = y_pred

results_df = pd.DataFrame(results).T
print("\n=== МЕТРИКИ МОДЕЛЕЙ ===")
print(results_df.round(4))

# ==================== АНАЛИТИКА ====================
print("\nАналитика...")

# Топ товаров по позитиву
product_stats = df.groupby('product').agg({
    'sentiment': ['mean', 'count']
}).round(3)
product_stats.columns = ['positive_ratio', 'reviews_count']
product_stats = product_stats[product_stats['reviews_count'] >= 30]
top_products = product_stats.nlargest(10, 'positive_ratio')

# Топ слов в негативных отзывах
negative_texts = df[df['sentiment'] == 0]['text'].str.lower()
words = ' '.join(negative_texts).split()
stop_words = {'the', 'and', 'of', 'a', 'in', 'is', 'with', 'this', 'that', 'it', 
              'for', 'wine', 'but', 'not', 'has', 'have', 'from', 'are', 'was',
              'on', 'to', 'is', 'an', 'as', 'be', 'by', 'at'}
words = [w for w in words if len(w) > 3 and w not in stop_words and w.isalpha()]
top_neg_words = Counter(words).most_common(10)

# ==================== ВИЗУАЛИЗАЦИЯ ====================
print("\nСоздание графиков...")

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Анализ винных отзывов', fontsize=18, fontweight='bold', y=0.98)

# 1. Распределение оценок
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(df['rating'], bins=5, color='#722f37', alpha=0.8, edgecolor='white')
ax1.set_title('Распределение оценок (рейтинг 1-5)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Рейтинг')
ax1.set_ylabel('Количество')
ax1.set_xticks([1, 2, 3, 4, 5])

# 2. Pie chart тональности
ax2 = fig.add_subplot(3, 3, 2)
sent_counts = df['sentiment'].value_counts()
colors = ['#e74c3c', '#27ae60']
labels = ['Негативные', 'Позитивные']
ax2.pie(sent_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', 
        startangle=90, explode=[0.05, 0])
ax2.set_title('Соотношение тональности', fontsize=12, fontweight='bold')

# 3. Топ стран по количеству отзывов
ax3 = fig.add_subplot(3, 3, 3)
top_countries = df['country'].value_counts().head(10)
ax3.barh(top_countries.index[::-1], top_countries.values[::-1], color='#8b5a6b', alpha=0.8)
ax3.set_title('Топ-10 стран по отзывам', fontsize=12, fontweight='bold')
ax3.set_xlabel('Количество отзывов')

# 4. Матрица ошибок
ax4 = fig.add_subplot(3, 3, 4)
cm = confusion_matrix(y_test, best_pred)
im = ax4.imshow(cm, cmap='Blues')
ax4.set_title('Матрица ошибок (лучшая модель)', fontsize=12, fontweight='bold')
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Негатив', 'Позитив'])
ax4.set_yticklabels(['Негатив', 'Позитив'])
for i in range(2):
    for j in range(2):
        ax4.text(j, i, cm[i, j], ha='center', va='center', fontsize=20, 
                color='white' if cm[i, j] > cm.max()/2 else 'black', fontweight='bold')

# 5. Облако слов
ax5 = fig.add_subplot(3, 3, 5)
all_text = ' '.join(df['text'].str.lower().values[:10000])
wordcloud = WordCloud(width=600, height=400, background_color='white', 
                      colormap='Reds', max_words=100).generate(all_text)
ax5.imshow(wordcloud, interpolation='bilinear')
ax5.axis('off')
ax5.set_title('Облако слов (все отзывы)', fontsize=12, fontweight='bold')

# 6. Топ вин по доле позитивных отзывов
ax6 = fig.add_subplot(3, 3, 6)
top_prod_names = [p[:18]+'...' if len(p) > 18 else p for p in top_products.index]
ax6.barh(range(len(top_prod_names)), top_products['positive_ratio'].values, color='#27ae60', alpha=0.8)
ax6.set_yticks(range(len(top_prod_names)))
ax6.set_yticklabels(top_prod_names)
ax6.set_title('Топ вин по доле позитива', fontsize=12, fontweight='bold')
ax6.set_xlabel('Доля позитивных')
ax6.invert_yaxis()

# 7. Таблица метрик
ax7 = fig.add_subplot(3, 3, 7)
ax7.axis('off')
table_data = []
for model_name, metrics in results.items():
    row = [f"{metrics['Accuracy']:.3f}", f"{metrics['Precision']:.3f}", 
           f"{metrics['Recall']:.3f}", f"{metrics['F1-Score']:.3f}"]
    table_data.append(row)

table = ax7.table(cellText=table_data,
                  rowLabels=list(results.keys()),
                  colLabels=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                  loc='center',
                  cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
ax7.set_title('Сравнение моделей', fontsize=12, fontweight='bold', pad=20)

# 8. Топ слов в негативных отзывах
ax8 = fig.add_subplot(3, 3, 8)
if top_neg_words:
    words_list = [w[0] for w in top_neg_words[:8]]
    counts_list = [w[1] for w in top_neg_words[:8]]
    ax8.barh(words_list[::-1], counts_list[::-1], color='#e74c3c', alpha=0.8)
    ax8.set_title('Топ слов в негативных отзывах', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Частота')

# 9. Распределение тональности по странам (топ-5)
ax9 = fig.add_subplot(3, 3, 9)
top5_countries = df['country'].value_counts().head(5).index
country_sentiment = df[df['country'].isin(top5_countries)].groupby('country')['sentiment'].mean()
country_sentiment = country_sentiment.sort_values(ascending=True)
colors9 = ['#27ae60' if v >= 0.5 else '#e74c3c' for v in country_sentiment.values]
ax9.barh(country_sentiment.index, country_sentiment.values, color=colors9, alpha=0.8)
ax9.set_title('Доля позитива по странам (топ-5)', fontsize=12, fontweight='bold')
ax9.set_xlabel('Доля позитивных')
ax9.set_xlim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('student_report.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n[OK] Otchet sohranen: student_report.png")

# ==================== КОНСОЛЬНЫЙ ВЫВОД ====================
print("\n" + "="*60)
print("ИТОГОВЫЙ ОТЧЁТ")
print("="*60)

print("\nMETRIKI MODELEY KLASSIFIKATSII:")
print(results_df.round(4).to_string())

print("\nLUCHSHAYA MODEL:", results_df['F1-Score'].idxmax())
print("   F1-Score:", f"{results_df['F1-Score'].max():.4f}")

print("\nSTATISTIKA DANNYH:")
print("   Vsego zapisey:", len(df))
print("   Pozitivnyh otzyvov:", df['sentiment'].sum(), f"({df['sentiment'].mean()*100:.1f}%)")
print("   Unikalnyh vin:", df['product'].nunique())
print("   Stran:", df['country'].nunique())

print("\nTOP-5 VIN PO DOLE POZITIVA:")
for i, (prod, row) in enumerate(top_products.head(5).iterrows(), 1):
    print("   ", i, ".", prod, ":", f"{row['positive_ratio']*100:.1f}%", f"({int(row['reviews_count'])} otzyvov)")

print("\nTOP-5 SLOV V NEGATIVNYH OTZYVAH:")
for word, count in top_neg_words[:5]:
    print("   -", word, ":", count)

print("\n" + "="*60)
print("Gotovo! Fayl: student_report.png")
print("="*60)