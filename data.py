import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier

teams = ['Manchester City', 
         'Arsenal', 
         'Manchester United',
         'Newcastle United', 
         'Liverpool', 
         'Brighton and Hove Albion',
         'Aston Villa', 
         'Tottenham Hotspur', 
         'Brentford', 
         'Fulham',
         'Crystal Palace', 
         'Chelsea', 
         'Wolverhampton Wanderers',
         'West Ham United', 
         'Bournemouth', 
         'Nottingham Forest', 
         'Everton',
         'Leicester City', 
         'Leeds United', 
         'Southampton', 
         'Burnley',
         'Watford', 
         'Norwich City', 
         'West Bromwich Albion',
         'Sheffield United', 
         'Cardiff City', 
         'Huddersfield Town']

team_codes = {'Manchester City': 0,
 'Arsenal': 1,
 'Manchester United': 2,
 'Newcastle United': 3,
 'Liverpool': 4,
 'Brighton and Hove Albion': 5,
 'Aston Villa': 6,
 'Tottenham Hotspur': 7,
 'Brentford': 8,
 'Fulham': 9,
 'Crystal Palace': 10,
 'Chelsea': 11,
 'Wolverhampton Wanderers': 12,
 'West Ham United': 13,
 'Bournemouth': 14,
 'Nottingham Forest': 15,
 'Everton': 16,
 'Leicester City': 17,
 'Leeds United': 18,
 'Southampton': 19,
 'Burnley': 20,
 'Watford': 21,
 'Norwich City': 22,
 'West Bromwich Albion': 23,
 'Sheffield United': 24,
 'Cardiff City': 25,
 'Huddersfield Town': 26}

team_ratings = {'Arsenal': 80,
 'Aston Villa': 77,
 'Brighton and Hove Albion': 75,
 'Burnley': 76,
 'Chelsea': 82,
 'Crystal Palace': 76,
 'Everton': 79,
 'Fulham': 75,
 'Leeds United': 76,
 'Leicester City': 80,
 'Liverpool': 85,
 'Manchester City': 85,
 'Manchester United': 82,
 'Newcastle United': 76,
 'Sheffield United': 73,
 'Southampton': 76,
 'Tottenham Hotspur': 82,
 'West Bromwich Albion': 73,
 'West Ham United': 78,
 'Wolverhampton Wanderers': 79,
 'Bournemouth': 74,
 'Brentford': 72,
 'Cardiff City': 71,
 'Huddersfield Town': 69,
 'Norwich City': 73,
 'Nottingham Forest': 71,
 'Watford': 74}

name_diff = {
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Tottenham": "Tottenham Hotspur",
    "Brighton": "Brighton and Hove Albion",
    "Newcastle Utd": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
    "Nott'ham Forest": "Nottingham Forest",
    "West Brom": "West Bromwich Albion",
    "Huddersfield": "Huddersfield Town",
    "Manchester Utd": "Manchester United"
}

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def prepare_data(matches: pd.DataFrame, team1, team2, goalsf1, goalsf2, goalsa1, goalsa2):
    # Fixing names
    matches["opponent"] = matches["opponent"].replace(name_diff)
    
    # Add team ratings
    fifa21 = pd.read_csv('teams_fifa21.csv')
    fifa21["Name"] = fifa21["Name"].replace("Brighton & Hove Albion", "Brighton and Hove Albion")
    fifa_rating = fifa21[fifa21["Name"].isin(teams)]
    fifa_dict = dict(zip(fifa_rating["Name"], fifa_rating["Overall"]))
    matches["team_rating"] = matches["team"].map(fifa_dict)
    matches["opp_rating"] = matches["opponent"].map(fifa_dict)

    # Data cleaning and preprocessing
    matches["date"] = pd.to_datetime(matches["date"])
    matches[["hours", "mins"]] = matches["time"].str.split(':', n=1, expand=True)
    matches["hours"] = matches["hours"].astype(int)
    matches["mins"] = matches["mins"].astype(int)
    matches["time"] = matches["hours"] + (matches["mins"] / 60)

    matches.drop(columns = ["hours", "mins"], axis = 1, inplace=True)

    matches["venue_code"] = matches["venue"].astype("category").cat.codes # Home and away
    matches["team_code"] = matches["team"].map(team_codes)
    matches["opp_code"] = matches["opponent"].map(team_codes) # Converting opponents into categorical codes
    matches["form_code"] = matches["formation"].astype('category').cat.codes
    matches["day_code"] = matches["date"].dt.dayofweek # Numbering days of the week 0-6
    matches.sort_values(by='date', inplace=True)

    # Fixing the index
    matches.index = range(matches.shape[0])
    matches = matches.fillna(method='ffill')

    labels_dict = {
        0:'Draw',
        1:'Loss',
        2:'Win'
    }
    matches["target"] = matches["result"].astype("category").cat.codes 

    cols = ["gf", "ga", "xg", "xga", "poss", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    new_cols = [f"{c}_rolling" for c in cols]

    predictors = [
        'team_code',
        'time',
        'form_code',
        'venue_code',
        'opp_code',
        'day_code'
    ]

    predictors = predictors + new_cols

    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    x = matches_rolling.droplevel('team')

    last_value = x[predictors].loc[x['team'] == team1].iloc[-1]
    last_value["opp_code"] = team_codes[team2]
    last_value["gf_rolling"] = goalsf1/10
    last_value["ga_rolling"] = goalsa1/10
    x = last_value.values.reshape(1, -1)

    return x, labels_dict

def model_run(x, labels):
    # Load the model
    model = pickle.load(open('premierleague.pkl', 'rb'))

    # Make predictions
    prediction = model.predict(x)
    prediction = [labels[p] for p in prediction]

    proba = model.predict_proba(x)

    probabilities = {
        labels[0]:str(round(proba[0][0],2)*100) + "%",
        labels[1]:str(round(proba[0][1],2)*100) + "%",
        labels[2]:str(round(proba[0][2],2)*100) + "%"
    }

    return prediction, probabilities