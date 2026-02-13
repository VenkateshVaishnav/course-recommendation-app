import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "online_course_data.xlsx"


# -------------------------------------------------
# LOAD + PREPROCESS (CACHED)
# -------------------------------------------------

def load_and_prepare_data():

    df = pd.read_excel(DATA_FILE)

    # Binary encoding
    df['certification_offered'] = df['certification_offered'].map({'Yes': 1, 'No': 0})
    df['study_material_available'] = df['study_material_available'].map({'Yes': 1, 'No': 0})

    # Ordinal encoding
    df['difficulty_level'] = df['difficulty_level'].map({
        'Beginner': 1,
        'Intermediate': 2,
        'Advanced': 3
    })

    # Scale numerical columns
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

    return df


# -------------------------------------------------
# BUILD MATRICES (ONLY WHEN NEEDED)
# -------------------------------------------------

def build_content_matrix(df):

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

    return X_content, course_index


def build_user_similarity(df):

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

    return user_item, user_similarity_df


# -------------------------------------------------
# HYBRID RECOMMENDATION
# -------------------------------------------------

def hybrid_recommendation(user_id=None, reference_course_id=None, top_n=5, alpha=0.5):

    df = load_and_prepare_data()

    X_content, course_index = build_content_matrix(df)
    user_item, user_similarity_df = build_user_similarity(df)

    # ---- Content Scores ----
    content_dict = {}

    if reference_course_id is not None and reference_course_id in course_index:

        idx = course_index[reference_course_id]
        ref_vector = X_content[idx].reshape(1, -1)
        scores = cosine_similarity(ref_vector, X_content)[0]

        content_dict = dict(zip(df['course_id'], scores))

    # ---- Collaborative Scores ----
    collaborative_dict = {}

    if user_id is not None and user_id in user_similarity_df.index:

        similar_users = (
            user_similarity_df[user_id]
            .sort_values(ascending=False)
            .iloc[1:6]
        )

        for sim_user, sim_value in similar_users.items():

            ratings = user_item.loc[sim_user]

            for cid, rating in ratings.items():
                if rating > 0:
                    collaborative_dict[cid] = collaborative_dict.get(cid, 0) + sim_value * rating

    # ---- Hybrid Combine ----
    final_scores = {}

    for cid in set(content_dict) | set(collaborative_dict):
        final_scores[cid] = (
            alpha * content_dict.get(cid, 0) +
            (1 - alpha) * collaborative_dict.get(cid, 0)
        )

    top_courses = sorted(
        final_scores,
        key=final_scores.get,
        reverse=True
    )[:top_n]

    return df[df['course_id'].isin(top_courses)][
        ['course_id', 'course_name', 'instructor', 'rating']
    ].drop_duplicates()
