# app/app.py
# This is the Streamlit UI entrypoint for the project.

import streamlit as st  # Streamlit is the framework that renders this script as a web app.

# Configure basic page settings before you render any UI elements.
# - page_title sets the browser tab title
# - layout="wide" gives us more horizontal space, which is useful for showing retrieved context later
st.set_page_config(page_title="RAG Eval Lab", layout="wide")

# Render a big title at the top of the page.
st.title("RAG Eval Lab")

# Render a short message (placeholder) so we can verify the UI is running.
# We'll replace this with chat input + retrieval + answers later.
st.write("Day 1 skeleton: UI is alive.")
