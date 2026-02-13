import sys
import os
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))
import streamlit as st
from model import hybrid_recommendation

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="centered"
)

# --------------------------------------
# TITLE
# --------------------------------------
st.title("🎓 Online Course Recommendation System")

st.markdown("""
This application uses a **Hybrid Recommendation System**
combining:

- 📌 Content-Based Filtering  
- 👥 Collaborative Filtering
""")

st.markdown("---")

# --------------------------------------
# INPUT SECTION
# --------------------------------------
st.subheader("🔍 Get Course Recommendations")

user_id = st.number_input(
    "Enter User ID (0 for new user)",
    min_value=0,
    step=1
)

reference_course_id = st.number_input(
    "Enter Reference Course ID",
    min_value=1,
    step=1
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)

st.markdown("---")

# --------------------------------------
# BUTTON ACTION
# --------------------------------------
if st.button("🎯 Recommend Courses"):

    with st.spinner("Generating recommendations..."):

        results = hybrid_recommendation(
            user_id=None if user_id == 0 else user_id,
            reference_course_id=reference_course_id,
            top_n=top_n
        )

    if results is not None and not results.empty:
        st.success("✅ Recommended Courses")
        st.dataframe(results.reset_index(drop=True), use_container_width=True)
    else:
        st.warning("⚠️ No recommendations found. Try different inputs.")

# --------------------------------------
# FOOTER
# --------------------------------------
st.markdown("---")
st.caption("Built with Python, Scikit-learn & Streamlit | Hybrid Recommendation System")

