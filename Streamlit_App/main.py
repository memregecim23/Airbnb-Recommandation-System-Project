import streamlit as st

st.set_page_config(layout="wide", page_title="Airbnb", page_icon="air.png")

# Klasör yapın düz olduğu için "page/" kısmını sildik
home_page = st.Page(page="homepage.py", title="Tanıtım", icon="🌎")
analytics_page = st.Page(page="analyticspage.py", title="Veriseti", icon="📊")
recommender_page = st.Page(page="recommender.py", title="Öneri Sistemi", icon="🏖️")

pg = st.navigation([home_page, analytics_page, recommender_page])

pg.run()
