import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set professional plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_data(filepath="data/results.csv"):
    """Loads and performs initial preprocessing on the dataset."""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    # Pre-calculate common metrics
    df["total_goals"] = df["home_score"] + df["away_score"]
    
    def get_result(row):
        if row["home_score"] > row["away_score"]:
            return "Home Win"
        elif row["home_score"] < row["away_score"]:
            return "Away Win"
        return "Draw"
    
    df["result"] = df.apply(get_result, axis=1)
    return df

def get_basic_stats(df):
    """Returns a dictionary of basic dataset statistics."""
    return {
        "total_matches": df.shape[0],
        "earliest_year": df['date'].min().year,
        "latest_year": df['date'].max().year,
        "unique_countries": df['country'].nunique(),
        "most_freq_home": df['home_team'].value_counts().idxmax(),
        "most_freq_home_count": df['home_team'].value_counts().max()
    }

def analyze_goals(df):
    """Performs deep analysis on goal statistics."""
    avg_goals = df["total_goals"].mean()
    highest_score_idx = df['total_goals'].idxmax()
    highest_score_match = df.loc[highest_score_idx]
    
    home_sum = df['home_score'].sum()
    away_sum = df['away_score'].sum()
    
    return {
        "avg_goals": avg_goals,
        "highest_scoring_match": {
            "home": highest_score_match['home_team'],
            "away": highest_score_match['away_team'],
            "goals": highest_score_match['total_goals']
        },
        "total_home_goals": home_sum,
        "total_away_goals": away_sum,
        "most_common_total": df['total_goals'].mode()[0]
    }

def get_match_outcomes(df):
    """Analyzes win/loss/draw percentages and home advantage."""
    outcome_pct = df['result'].value_counts(normalize=True) * 100
    return outcome_pct.to_dict()

def get_historical_leaders(df):
    """Identifies top winning teams in history."""
    home_wins = df[df['result'] == 'Home Win']['home_team']
    away_wins = df[df['result'] == 'Away Win']['away_team']
    all_wins = pd.concat([home_wins, away_wins])
    
    top_teams = all_wins.value_counts().head(10)
    return top_teams

def plot_goals_distribution(df):
    """Visualizes the distribution of total goals."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['total_goals'], bins=range(0, 20), kde=True, color='teal')
    plt.title('Distribution of Total Goals per Match (1872-2024)', fontsize=16, pad=20)
    plt.xlabel('Total Goals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 15)
    plt.tight_layout()
    plt.show()

def plot_outcome_distribution(df):
    """Visualizes the distribution of match outcomes using a bar chart."""
    plt.figure(figsize=(10, 6))
    outcomes = df['result'].value_counts()
    sns.barplot(x=outcomes.index, y=outcomes.values, hue=outcomes.index, palette="viridis", legend=False)
    plt.title('Match Outcome Distribution: Identifying Home Advantage', fontsize=16, pad=20)
    plt.xlabel('Match Result', fontsize=12)
    plt.ylabel('Number of Matches', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_top_winners(df):
    """Visualizes the top 10 teams by total wins."""
    leaders = get_historical_leaders(df)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=leaders.values, y=leaders.index, hue=leaders.index, palette="magma", legend=False)
    plt.title('Top 10 International Football Teams by Total Wins', fontsize=16, pad=20)
    plt.xlabel('Number of Wins', fontsize=12)
    plt.ylabel('Team', fontsize=12)
    plt.tight_layout()
    plt.show()
