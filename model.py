import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1️⃣ LOAD DATA (STREAMLIT SAFE)
# ==========================================================

DATA_FILE = "online_course_data.csv"   # Use CSV (Recommended)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found. Make sure it exists in the repo root."
    )

df = pd.read_csv(DATA_FILE)

# ==========================================================
# 2️⃣ PREPROCESSING
# ==========================================================

# Binary encoding
df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

# Ordinal encoding
df['difficulty_level'] = df['difficulty_level'].map({
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3
})

# Fill possible NaNs after mapping
df.fillna(0, inplace=True)

# Scale numeric columns
scale_cols = [
    'course_duration_hours',
    'course_price',
    'enrollment_numbers',
    'feedback_score',
    'time_spent_hours',
    'previous_courses_taken',
    'rating'
]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# ==========================================================
# 3️⃣ CONTENT-BASED MATRIX
# ==========================================================

content_features = [
    'course_duration_hours',
    'course_price',
    'difficulty_level',
    'certification_offered',
    'study_material_available',
    'feedback_score',
    'enrollment_numbers',
    'time_spent_hours',
    'previous_courses_taken',
    'rating'
]

X_content = df[content_features].values
course_index = {cid: idx for idx, cid in enumerate(df['course_id'])}

# ==========================================================
# 4️⃣ COLLABORATIVE FILTERING
# ==========================================================

user_item = df.pivot_table(
    index='user_id',
    columns='course_id',
    values='rating'
).fillna(0)

user_similarity = cosine_similarity(user_item)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item.index,
    columns=user_item.index
)

# ==========================================================
# 5️⃣ CONTENT SCORE FUNCTION
# ==========================================================

def content_scores(reference_course_id):

    if reference_course_id not in course_index:
        return {}

    idx = course_index[reference_course_id]
    ref_vector = X_content[idx].reshape(1, -1)

    scores = cosine_similarity(ref_vector, X_content)[0]

    return dict(zip(df['course_id'], scores))


# ==========================================================
# 6️⃣ COLLABORATIVE SCORE FUNCTION
# ==========================================================

def collaborative_scores(user_id, top_k=5):

    if user_id not in user_similarity_df.index:
        return {}

    similar_users = (
        user_similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1:top_k+1]
    )

    scores = {}

    for sim_user, sim_value in similar_users.items():

        ratings = user_item.loc[sim_user]

        for cid, rating in ratings.items():
            if rating > 0:
                scores[cid] = scores.get(cid, 0) + sim_value * rating

    return scores


# ==========================================================
# 7️⃣ HYBRID RECOMMENDATION
# ==========================================================

def hybrid_recommendation(
    user_id=None,
    reference_course_id=None,
    top_n=5,
    alpha=0.5
):

    content_dict = {}
    collaborative_dict = {}

    if reference_course_id is not None:
        content_dict = content_scores(reference_course_id)

    if user_id is not None:
        collaborative_dict = collaborative_scores(user_id)

    final_scores = {}

    for cid in set(content_dict) | set(collaborative_dict):
        final_scores[cid] = (
            alpha * content_dict.get(cid, 0) +
            (1 - alpha) * collaborative_dict.get(cid, 0)
        )

    if not final_scores:
        return pd.DataFrame()

    top_courses = sorted(
        final_scores,
        key=final_scores.get,
        reverse=True
    )[:top_n]

    return (
        df[df['course_id'].isin(top_courses)]
        [['course_id', 'course_name', 'instructor', 'rating']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
