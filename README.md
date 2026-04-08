# 🥊 UFC Fight Predictor

A machine learning system that predicts UFC fight outcomes using historical fighter statistics.
Trained on 7,169 fights spanning 1993–2026.

## Live Demo

```bash
streamlit run app.py
```

---

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression (with odds) | 67.8% | 0.729 |
| XGBoost (with odds) | 67.6% | 0.725 |
| Random Forest (with odds) | 66.9% | 0.722 |
| Logistic Regression (no odds) | 63.0% | 0.653 |
| Baseline (majority class) | 57.8% | 0.500 |

**Key finding:** Two betting odds features account for +4.8% accuracy and +7.6 AUC points.
The no-odds model represents what pure fight analytics can achieve independently of the market.

---

## Methodology

### Data
- Source: [Ultimate UFC Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) (Kaggle)
- 7,177 raw fights → 7,169 after removing draws and no-contests
- 118 raw columns → 43 engineered features

### Feature Engineering

Raw fighter stats exist as `R_` and `B_` prefixed columns. Rather than feeding these directly
(which would teach the model corner-color bias), all features are computed as **Red minus Blue differentials**.
A positive value means the red corner fighter has the advantage.

| Feature Group | Count | Examples |
|---|---|---|
| Differential stats | 22 | wins, losses, reach, age, takedowns, win streaks |
| Ratio features | 3 | finish rate, decision rate, win rate |
| Categorical | 3 | stance matchup, weight class, gender |

**Ratio features** use zero-safe division — debut fighters with 0 wins receive a rate of 0
rather than producing division-by-zero artifacts.

**Stance matchup** is encoded as three categories: same stance, southpaw involved, other.
Southpaw fighters appear in ~31% of fights and carry a known statistical edge.

### Model Selection

All three models were evaluated with and without betting odds features:

- **Logistic Regression** performed best overall despite being the simplest model.
  Differential features produce largely linear decision boundaries — more complex models
  overfit the noise without gaining signal.
- **XGBoost** and **Random Forest** performed within 1% of Logistic Regression,
  confirming that additional complexity is not warranted for this feature set.

### The Odds Question

Betting odds (`diff_odds`, `diff_ev`) are the two most predictive features, accounting for
31% of Random Forest's total feature importance. This raises a question any honest
ML practitioner should ask: are we predicting fights, or replicating what bookmakers already know?

Both model versions are available in the app — with and without odds — so the user can
see exactly how much the market is doing the work.

### Data Leakage Prevention

- Train/test split performed before any scaling
- `StandardScaler` fitted on training data only, applied to test
- Ranking columns (97–99% missing) dropped — not imputed
- No post-fight statistics used as input features
- Stratified split maintains the 57.8/42.2 Red/Blue win ratio in both sets

---

## Project Structure

```
ufc-predictor/
├── app.py                  # Streamlit web app
├── predict.py              # Inference engine
├── requirements.txt
├── Data/
│   └── fighters.csv        # Fighter database (2,241 fighters)
├── models/
│   ├── lr_with_odds.pkl
│   ├── lr_without_odds.pkl
│   ├── scaler_with_odds.pkl
│   ├── scaler_without_odds.pkl
│   ├── feature_cols_with_odds.pkl
│   └── feature_cols_without_odds.pkl
├── notebooks/
│   └── 01_eda.ipynb        # EDA, feature engineering, model training
└── outputs/                # Plots and visualizations
```

---

## Setup

```bash
git clone https://github.com/AhmedkhalilDAOUDI/UFC-Predictor.git
cd UFC-Predictor
pip install -r requirements.txt
```

Download [ufc-master.csv](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset)
and place it in the `Data/` folder.

To regenerate models from scratch:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

To run the app directly using pre-trained models:

```bash
streamlit run app.py
```

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · joblib · Matplotlib · Seaborn
