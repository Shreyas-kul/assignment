# -*- coding: utf-8 -*-
"""coldstart_recommendation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eDEiHPfST2ysocJ69REOJ_rNQFgVOH2C
"""

pip install scikit-surprise

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.model_selection import train_test_split as sk_train_test_split

customer_data = pd.DataFrame({
    'customer_id': [1, 2, 1, 3, 2, 3, 1],
    'product_id': [101, 102, 103, 101, 104, 102, 105],
    'rating': [5, 4, 5, 3, 4, 4, 5]
})

product_data = pd.DataFrame({
    'product_id': [101, 102, 103, 104, 105],
    'description': ['Fresh basil leaves', 'Tomato paste', 'Mozzarella cheese', 'Olive oil', 'Garlic cloves'],
    'category': ['Herb', 'Sauce', 'Cheese', 'Oil', 'Herb']
})

customer_data.fillna(0, inplace=True)
product_data.fillna('Unknown', inplace=True)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
product_vectors = tfidf_vectorizer.fit_transform(product_data['description'])

product_similarity = cosine_similarity(product_vectors, product_vectors)

product_similarity_df = pd.DataFrame(product_similarity, index=product_data['product_id'], columns=product_data['product_id'])
print("Product Similarity Matrix:")
print(product_similarity_df)

def content_based_recommendation(product_id, top_n=5):
    product_idx = product_data[product_data['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(product_similarity[product_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar = sim_scores[1:top_n+1]  # Exclude itself
    recommended_product_ids = [product_data.iloc[i[0]]['product_id'] for i in top_similar]
    return recommended_product_ids

print("Content-Based Recommendations for Product 101:", content_based_recommendation(101))

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(customer_data[['customer_id', 'product_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

svd_model = SVD()
svd_model.fit(trainset)

predictions = svd_model.test(testset)
print("RMSE for Collaborative Filtering:", accuracy.rmse(predictions))

def hybrid_recommendation(customer_id, product_id, top_n=5):
    content_recs = content_based_recommendation(product_id, top_n)

    collaborative_recs = []
    for prod in content_recs:
        est_rating = svd_model.predict(customer_id, prod).est
        collaborative_recs.append((prod, est_rating))

    collaborative_recs.sort(key=lambda x: x[1], reverse=True)
    top_hybrid_recs = [rec[0] for rec in collaborative_recs[:top_n]]
    return top_hybrid_recs

print("Hybrid Recommendations for Customer 1 and Product 101:", hybrid_recommendation(1, 101))

def precision_at_n(testset, n=5):
    hits = 0
    total = 0
    for uid, iid, true_r in testset:
        recommendations = hybrid_recommendation(uid, iid, top_n=n)
        if iid in recommendations:
            hits += 1
        total += 1
    return hits / total

print("Precision@5 for Hybrid System:", precision_at_n(testset, n=5))

