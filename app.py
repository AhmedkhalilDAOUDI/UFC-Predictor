import streamlit as st
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict as predictor
importlib.reload(predictor)

# Page config
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="centered"
)

# Header
st.title("🥊 UFC Fight Predictor")
st.markdown("Machine learning model trained on 7,169 UFC fights (1993–2026)")
st.divider()

# Fighter selection
fighter_names = predictor.get_fighter_names()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Red Corner")
    red_fighter = st.selectbox(
        "Select fighter",
        fighter_names,
        key="red",
        index=fighter_names.index("Conor McGregor") if "Conor McGregor" in fighter_names else 0
    )

with col2:
    st.markdown("### 🔵 Blue Corner")
    blue_fighter = st.selectbox(
        "Select fighter",
        fighter_names,
        key="blue",
        index=fighter_names.index("Khabib Nurmagomedov") if "Khabib Nurmagomedov" in fighter_names else 1
    )

st.divider()

# Fight context
col3, col4 = st.columns(2)

with col3:
    weight_class = st.selectbox("Weight Class", [
        "Heavyweight", "Light Heavyweight", "Middleweight",
        "Welterweight", "Lightweight", "Featherweight",
        "Bantamweight", "Flyweight", "Catch Weight",
        "Women's Strawweight", "Women's Flyweight",
        "Women's Bantamweight", "Women's Featherweight"
    ])

with col4:
    gender = st.radio("Gender", ["MALE", "FEMALE"], horizontal=True)

# Optional odds
with st.expander("Add betting odds (optional)"):
    st.markdown("Enter American odds (e.g. -150 for favourite, +120 for underdog)")
    col5, col6 = st.columns(2)
    with col5:
        r_odds_input = st.number_input("Red corner odds", value=0, step=10)
    with col6:
        b_odds_input = st.number_input("Blue corner odds", value=0, step=10)
    use_odds = st.checkbox("Use odds in prediction", value=False)

st.divider()

# Predict button
if st.button("🔮 Predict Fight", use_container_width=True, type="primary"):
    if red_fighter == blue_fighter:
        st.error("Select two different fighters.")
    else:
        try:
            r_odds = r_odds_input if use_odds and r_odds_input != 0 else None
            b_odds = b_odds_input if use_odds and b_odds_input != 0 else None

            result = predictor.predict(
                red_name=red_fighter,
                blue_name=blue_fighter,
                weight_class=weight_class,
                gender=gender,
                r_odds=r_odds,
                b_odds=b_odds
            )

            # Display results
            st.markdown("## Prediction Results")

            # Pick which model result to show
            model_result = result['with_odds'] if (use_odds and r_odds and b_odds) else result['without_odds']
            model_label = "with betting odds" if (use_odds and r_odds and b_odds) else "analytics only"

            winner = model_result['predicted_winner']
            confidence = model_result['confidence']
            red_prob = model_result['red_win_prob']
            blue_prob = model_result['blue_win_prob']

            # Winner banner
            winner_color = "🔴" if winner == red_fighter else "🔵"
            st.success(f"{winner_color} **{winner}** wins — {confidence*100:.1f}% confidence ({model_label})")

            # Probability bars
            st.markdown("### Win Probability")
            col7, col8 = st.columns(2)
            with col7:
                st.metric(f"🔴 {red_fighter}", f"{red_prob*100:.1f}%")
                st.progress(red_prob)
            with col8:
                st.metric(f"🔵 {blue_fighter}", f"{blue_prob*100:.1f}%")
                st.progress(blue_prob)

            # Both models side by side
            st.divider()
            st.markdown("### Model Comparison")
            st.markdown("How much do the betting odds change the prediction?")

            col9, col10 = st.columns(2)
            with col9:
                st.markdown("**Analytics Only**")
                wo = result['without_odds']
                st.write(f"🔴 {red_fighter}: {wo['red_win_prob']*100:.1f}%")
                st.write(f"🔵 {blue_fighter}: {wo['blue_win_prob']*100:.1f}%")
                st.write(f"Winner: **{wo['predicted_winner']}**")

            with col10:
                st.markdown("**With Betting Odds**")
                wi = result['with_odds']
                st.write(f"🔴 {red_fighter}: {wi['red_win_prob']*100:.1f}%")
                st.write(f"🔵 {blue_fighter}: {wi['blue_win_prob']*100:.1f}%")
                st.write(f"Winner: **{wi['predicted_winner']}**")

        except ValueError as e:
            st.error(str(e))

# Footer
st.divider()
st.markdown(
    "Built with scikit-learn · Logistic Regression · 7,169 UFC fights · "
    "[GitHub](https://github.com/AhmedkhalilDAOUDI)"
)