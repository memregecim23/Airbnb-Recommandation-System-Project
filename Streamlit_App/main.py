import Streamlit_App as st

st.set_page_config(layout="wide", page_title="Airbnb", page_icon="air.png")

home_page = st.Page(page="page/homepage.py",title="Tanıtım", icon="🌎")
analytics_page = st.Page(page="page/analyticspage.py" , title="Veriseti" ,icon= "📊" )
recommender_page = st.Page(page="page/recommender.py" , title="recommender" ,icon= "🏖️" )

pg = st.navigation([home_page , analytics_page,recommender_page])

pg.run()

