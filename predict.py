import pandas as pd
import numpy as np
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load assets once at import time
fighters_df = pd.read_csv(os.path.join(DATA_DIR, 'fighters.csv'))

lr_odds = joblib.load(os.path.join(MODEL_DIR, 'lr_with_odds.pkl'))
lr_no = joblib.load(os.path.join(MODEL_DIR, 'lr_without_odds.pkl'))
scaler_odds = joblib.load(os.path.join(MODEL_DIR, 'scaler_with_odds.pkl'))
scaler_no = joblib.load(os.path.join(MODEL_DIR, 'scaler_without_odds.pkl'))
feat_odds = joblib.load(os.path.join(MODEL_DIR, 'feature_cols_with_odds.pkl'))
feat_no = joblib.load(os.path.join(MODEL_DIR, 'feature_cols_without_odds.pkl'))


def get_fighter(name: str) -> pd.Series:
    match = fighters_df[fighters_df['fighter'].str.lower() == name.lower()]
    if match.empty:
        raise ValueError(f"Fighter '{name}' not found in database.")
    return match.iloc[0]


def safe_rate(num, den):
    return num / den if den > 0 else 0.0


def compute_features(r: pd.Series, b: pd.Series,
                     r_odds: float = None, b_odds: float = None) -> dict:
    numeric_stats = [
        'Height_cms', 'Reach_cms', 'Weight_lbs', 'age',
        'avg_TD_landed', 'avg_TD_pct',
        'current_lose_streak', 'current_win_streak',
        'draw', 'longest_win_streak', 'losses',
        'total_rounds_fought', 'total_title_bouts',
        'win_by_Decision_Majority', 'win_by_Decision_Split',
        'win_by_Decision_Unanimous', 'win_by_KO/TKO',
        'win_by_Submission', 'win_by_TKO_Doctor_Stoppage', 'wins'
    ]

    row = {}

    for stat in numeric_stats:
        r_val = float(r.get(stat, 0) or 0)
        b_val = float(b.get(stat, 0) or 0)
        row[f'diff_{stat}'] = r_val - b_val

    row['diff_odds'] = (r_odds - b_odds) if (r_odds is not None and b_odds is not None) else 0.0
    row['diff_ev'] = 0.0

    r_wins = float(r.get('wins', 0) or 0)
    b_wins = float(b.get('wins', 0) or 0)
    r_losses = float(r.get('losses', 0) or 0)
    b_losses = float(b.get('losses', 0) or 0)

    r_finish = float((r.get('win_by_KO/TKO', 0) or 0) + (r.get('win_by_Submission', 0) or 0))
    b_finish = float((b.get('win_by_KO/TKO', 0) or 0) + (b.get('win_by_Submission', 0) or 0))

    r_dec = float((r.get('win_by_Decision_Unanimous', 0) or 0) +
                  (r.get('win_by_Decision_Split', 0) or 0) +
                  (r.get('win_by_Decision_Majority', 0) or 0))
    b_dec = float((b.get('win_by_Decision_Unanimous', 0) or 0) +
                  (b.get('win_by_Decision_Split', 0) or 0) +
                  (b.get('win_by_Decision_Majority', 0) or 0))

    row['diff_finish_rate'] = safe_rate(r_finish, r_wins) - safe_rate(b_finish, b_wins)
    row['diff_decision_rate'] = safe_rate(r_dec, r_wins) - safe_rate(b_dec, b_wins)
    row['diff_win_rate'] = safe_rate(r_wins, r_wins + r_losses) - safe_rate(b_wins, b_wins + b_losses)

    r_stance = str(r.get('Stance', 'Orthodox') or 'Orthodox')
    b_stance = str(b.get('Stance', 'Orthodox') or 'Orthodox')

    if r_stance == b_stance:
        stance = 'same'
    elif 'Southpaw' in [r_stance, b_stance]:
        stance = 'southpaw_involved'
    else:
        stance = 'other'

    row['stance_matchup_other'] = int(stance == 'other')
    row['stance_matchup_same'] = int(stance == 'same')
    row['stance_matchup_southpaw_involved'] = int(stance == 'southpaw_involved')

    return row


def build_input(row: dict, feature_cols: list, weight_class: str, gender: str) -> pd.DataFrame:
    X = pd.DataFrame([row])

    wc_cols = [c for c in feature_cols if c.startswith('weight_class_')]
    for col in wc_cols:
        X[col] = int(col == f'weight_class_{weight_class}')

    X['gender_FEMALE'] = int(gender == 'FEMALE')
    X['gender_MALE'] = int(gender == 'MALE')

    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0

    return X[feature_cols]


def predict(red_name: str, blue_name: str,
            weight_class: str, gender: str,
            r_odds: float = None, b_odds: float = None) -> dict:

    r = get_fighter(red_name)
    b = get_fighter(blue_name)
    row = compute_features(r, b, r_odds, b_odds)

    X_odds = build_input(row, feat_odds, weight_class, gender)
    X_odds_scaled = scaler_odds.transform(X_odds)
    prob_red_odds = lr_odds.predict_proba(X_odds_scaled)[0][1]

    X_no = build_input(row, feat_no, weight_class, gender)
    X_no_scaled = scaler_no.transform(X_no)
    prob_red_no = lr_no.predict_proba(X_no_scaled)[0][1]

    return {
        'red_fighter': red_name,
        'blue_fighter': blue_name,
        'with_odds': {
            'red_win_prob': round(prob_red_odds, 3),
            'blue_win_prob': round(1 - prob_red_odds, 3),
            'predicted_winner': red_name if prob_red_odds > 0.5 else blue_name,
            'confidence': round(max(prob_red_odds, 1 - prob_red_odds), 3)
        },
        'without_odds': {
            'red_win_prob': round(prob_red_no, 3),
            'blue_win_prob': round(1 - prob_red_no, 3),
            'predicted_winner': red_name if prob_red_no > 0.5 else blue_name,
            'confidence': round(max(prob_red_no, 1 - prob_red_no), 3)
        }
    }


def get_fighter_names() -> list:
    return sorted(fighters_df['fighter'].dropna().tolist())