import streamlit as st
import pandas as pd

from src.utils.feedback import load_feedback

st.set_page_config(page_title="Feedback Dashboard")

st.title("Feedback Dashboard")

feedback = load_feedback()

if not feedback:
    st.info("No feedback received yet.")
    st.stop()

df = pd.DataFrame(feedback)

total = len(df)
positive = len(df[df["rating"] == "up"])
negative = len(df[df["rating"] == "down"])

st.metric("Total Feedback", total)
st.metric("Positive", positive)
st.metric("Negative", negative)

st.divider()

st.subheader("Most Recent Feedback")

st.dataframe(
    df.sort_values("timestamp", ascending=False),
    use_container_width=True,
)