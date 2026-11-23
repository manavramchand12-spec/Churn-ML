"""Telco Customer Churn Predictor - Streamlit app.

Run locally with:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


# ---------- Loading ----------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        # The saved model can fail to load on a different scikit-learn version, so rebuild it
        from train import train_model
        return train_model(save=False)


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


df = load_data()
model = load_model()
metrics = load_metrics()

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]


# ---------- Sidebar: customer profile ----------
st.sidebar.header("Customer profile")

with st.sidebar.expander("👤 Demographics", expanded=True):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.radio("Senior citizen", YES_NO, horizontal=True)
    partner = st.radio("Has a partner", YES_NO, horizontal=True)
    dependents = st.radio("Has dependents", YES_NO, horizontal=True)

with st.sidebar.expander("📄 Account", expanded=True):
    tenure = st.slider("Tenure (months)", 0, 72, 6)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.radio("Paperless billing", YES_NO, index=1, horizontal=True)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)",
    ])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 80.0, step=0.5)
    total = st.number_input("Total charges ($)", 0.0, 9000.0, float(round(monthly * tenure, 2)),
                            help="Defaults to monthly charges × tenure.")

with st.sidebar.expander("📶 Services", expanded=False):
    phone = st.radio("Phone service", YES_NO, index=1, horizontal=True)
    multiple_lines = st.selectbox(
        "Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"]
    )
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addon_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
    security = st.selectbox("Online security", addon_options)
    backup = st.selectbox("Online backup", addon_options)
    device = st.selectbox("Device protection", addon_options)
    support = st.selectbox("Tech support", addon_options)
    tv = st.selectbox("Streaming TV", addon_options)
    movies = st.selectbox("Streaming movies", addon_options)

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": int(senior == "Yes"),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}])


# ---------- Risk factors found in the EDA ----------
def risk_factors(c: pd.Series) -> list[tuple[str, str]]:
    """(why it matters, suggested action) pairs for this customer, based on notebook findings."""
    factors = []
    if c["Contract"] == "Month-to-month":
        factors.append(("Month-to-month contract (43% churn vs 3% on two-year)",
                        "Offer a discount for upgrading to a 1- or 2-year plan."))
    if c["tenure"] <= 12:
        factors.append(("In the first year as a customer, when churn is highest",
                        "Add to an early-life onboarding / check-in programme."))
    if c["InternetService"] == "Fiber optic":
        factors.append(("Fiber optic customer (42% churn vs 19% on DSL)",
                        "Check service quality and price against competitors."))
    if c["TechSupport"] == "No":
        factors.append(("No tech support (42% churn vs 15% with it)",
                        "Offer a free tech-support trial."))
    if c["OnlineSecurity"] == "No":
        factors.append(("No online security (42% churn vs 15% with it)",
                        "Bundle online security at a discount."))
    if c["PaymentMethod"] == "Electronic check":
        factors.append(("Pays by electronic check (45% churn)",
                        "Encourage automatic bank or card payments."))
    if c["MonthlyCharges"] >= 80:
        factors.append(("High monthly bill (churners pay a median of about $80)",
                        "Review the plan for a better-value bundle."))
    return factors


# ---------- Main page ----------
st.title("📉 Telco Customer Churn Predictor")
st.caption("Predicts whether a telecom customer is likely to leave, using a model trained on 7,043 "
           "IBM Telco customers. Fill in the profile on the left.")

tab_predict, tab_insights, tab_model = st.tabs(["🔮 Prediction", "📊 Churn insights", "🧠 About the model"])

with tab_predict:
    probability = float(model.predict_proba(customer)[0, 1])
    will_churn = probability >= 0.5

    if probability >= 0.7:
        level, colour = "High", "🔴"
    elif probability >= 0.4:
        level, colour = "Medium", "🟠"
    else:
        level, colour = "Low", "🟢"

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{probability:.0%}")
    c2.metric("Risk level", f"{colour} {level}")
    c3.metric("Prediction", "Likely to churn" if will_churn else "Likely to stay")
    st.progress(probability)

    if will_churn:
        st.error("This customer is predicted to **churn**. Consider a retention offer.")
    else:
        st.success("This customer is predicted to **stay**.")

    factors = risk_factors(customer.iloc[0])
    st.subheader("Why this customer may be at risk")
    if factors:
        st.table(pd.DataFrame(factors, columns=["Risk factor", "Suggested action"]))
    else:
        st.write("None of the main churn risk factors from the analysis apply to this customer.")

with tab_insights:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Overall churn rate", f"{df['ChurnFlag'].mean():.1%}")
    k3.metric("Median tenure of churners", f"{df.loc[df.ChurnFlag == 1, 'tenure'].median():.0f} months")
    k4.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")

    feature = st.selectbox("Show churn rate by", [
        "Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
        "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents", "gender",
    ])
    rates = (df.groupby(feature)["ChurnFlag"].mean().mul(100).round(1)
             .sort_values(ascending=False).rename("Churn rate (%)"))
    st.bar_chart(rates, horizontal=True)

    st.subheader("Churn rate by tenure")
    tenure_rates = (df.assign(TenureGroup=pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                                                 labels=["0-12", "13-24", "25-48", "49-72"]))
                    .groupby("TenureGroup", observed=True)["ChurnFlag"].mean().mul(100).round(1)
                    .rename("Churn rate (%)"))
    st.bar_chart(tenure_rates)

    with st.expander("Browse the raw data"):
        st.dataframe(df.drop(columns="ChurnFlag"))

with tab_model:
    if metrics:
        st.markdown(f"**Model:** {metrics['model_name']} with balanced class weights, inside a "
                    "scikit-learn pipeline (standard scaling + one-hot encoding).")
        m = metrics["test_metrics"]
        a, b, c, d, e = st.columns(5)
        a.metric("Churn recall", f"{m['Churn recall']:.0%}")
        b.metric("Churn precision", f"{m['Churn precision']:.0%}")
        c.metric("Churn F1", f"{m['Churn F1']:.2f}")
        d.metric("ROC AUC", f"{m['ROC AUC']:.2f}")
        e.metric("Accuracy", f"{m['Accuracy']:.0%}")
        st.caption(f"Scored on {metrics['test_rows']:,} held-out customers.")
    st.markdown(
        "- Four models were compared (Logistic Regression, Decision Tree, Random Forest, SVM) and tuned "
        "with 5-fold cross-validation on churn F1.\n"
        "- The model is tuned to **catch as many churners as possible**: it finds about 4 in 5 customers "
        "who leave, at the cost of some false alarms.\n"
        "- Full analysis: `notebooks/telco_churn_analysis.ipynb`."
    )
