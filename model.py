import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# LOAD DATA
# ==========================================================

DATA_FILE = "online_course_data.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in project folder.")

df = pd.read_csv(DATA_FILE)

# ==========================================================
# PREPROCESSING
# ==========================================================

df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

df['difficulty_level'] = df['difficulty_level'].map({
    'Beginner': 1,
    'Intermediate': 2,
    'Advanced': 3
})

df.fillna(0, inplace=True)

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
# CONTENT MATRIX
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
# HYBRID RECOMMENDATION (Optimized)
# ==========================================================

def hybrid_recommendation(user_id=None, reference_course_id=None, top_n=5, alpha=0.7):

    final_scores = {}

    # -------------------------
    # CONTENT-BASED
    # -------------------------
    if reference_course_id in course_index:

        idx = course_index[reference_course_id]
        ref_vector = X_content[idx].reshape(1, -1)

        content_scores = cosine_similarity(ref_vector, X_content)[0]

        for cid, score in zip(df['course_id'], content_scores):
            final_scores[cid] = alpha * score

    # -------------------------
    # LIGHT COLLABORATIVE
    # -------------------------
    if user_id is not None and user_id in df['user_id'].values:

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

        similar_users = (
            user_similarity_df[user_id]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        for sim_user, sim_value in similar_users.items():

            ratings = user_item.loc[sim_user]

            for cid, rating in ratings.items():
                if rating > 0:
                    final_scores[cid] = final_scores.get(cid, 0) + (1 - alpha) * sim_value * rating

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
