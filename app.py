import streamlit as st
from model import hybrid_recommendation

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="centered"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🎓 Online Course Recommendation System")

st.markdown("""
This app uses a **Hybrid Recommendation Model**  
(Content-Based + Collaborative Filtering)
to suggest personalized courses.
""")

st.markdown("---")

# -------------------------------------------------
# USER INPUTS
# -------------------------------------------------
st.subheader("🔍 Get Course Recommendations")

user_id = st.number_input(
    "Enter User ID (0 for new user)",
    min_value=0,
    step=1
)

reference_course_id = st.number_input(
    "Enter Reference Course ID (0 if none)",
    min_value=0,
    step=1
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=5
)

st.markdown("---")

# -------------------------------------------------
# RECOMMEND BUTTON
# -------------------------------------------------
if st.button("🎯 Recommend Courses"):

    try:
        # Handle cold-start logic
        uid = None if user_id == 0 else user_id
        cid = None if reference_course_id == 0 else reference_course_id

        results = hybrid_recommendation(
            user_id=uid,
            reference_course_id=cid,
            top_n=top_n
        )

        if results is None or results.empty:
            st.warning("⚠️ No recommendations found. Try different inputs.")
        else:
            st.success("✅ Top Recommended Courses")
            st.dataframe(results.reset_index(drop=True), width="stretch")

    except Exception as e:
        st.error("❌ Error generating recommendations.")
        st.exception(e)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Python, Pandas, Scikit-learn & Streamlit | "
    "Hybrid Recommendation Engine"
)
