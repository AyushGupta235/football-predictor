import streamlit as st
import pandas as pd

from data import teams, team_codes, team_ratings, prepare_data, model_run

def set_bg_hack_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url("https://wallpapercave.com/wp/wp6592753.jpg");
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

st.title("Premier League")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select the your team", sorted(teams))
with col2:
    team2 = st.selectbox("Select the opposing team", sorted(teams))

target_score = st.selectbox("Is your team at home or away?", ["Home","Away"])

with col1:
    goalsf1 = st.number_input("Enter the goals scored by your team in the last 10 games", min_value=0, value=0)
with col2:
    goalsa1 = st.number_input("Enter the goals conceded by your team in the last 10 games", min_value=0, value=0)

with col1:
    goalsf2 = st.number_input("Enter the goals scored by opposing team in the last 10 games", min_value=0, value=0)
with col2:
    goalsa2 = st.number_input("Enter the goals conceded by opposing team in the last 10 games", min_value=0, value=0)


if st.button("Predict Probability"):
    team1_ = team_codes[team1]
    team2_ = team_codes[team2]
    team1_rating = team_ratings[team1]
    team2_rating = team_ratings[team2]

    goalsf1 = goalsf1/3
    goalsa1 = goalsa1/3

    df = pd.read_csv('matches_5yr.csv')

    X,labels = prepare_data(df, team1=team1, team2=team2, goalsf1=goalsf1, goalsa1=goalsa1, goalsf2=goalsa2, goalsa2=goalsa2)
    pred, prob = model_run(X, labels)

    st.header("Predicted result: " + pred[0])
    if st.button("Show all probabilities"):
        st.write(prob)