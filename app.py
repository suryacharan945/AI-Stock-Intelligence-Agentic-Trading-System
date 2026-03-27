# =============================================================
# AI STOCK INTELLIGENCE PLATFORM (STEP 1 - FOUNDATION)
# =============================================================

# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI Stock Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== HEADER ==================
st.title("🚀 AI Stock Intelligence & Agentic Trading Platform")

st.markdown("""
**An AI-powered system for intelligent stock analysis and decision making**

📊 Technical Analysis • 🧠 Sentiment Intelligence • 🤖 ML Predictions • 🎯 Agentic Decisions • 📈 Strategy Backtesting
""")

st.markdown("---")

# =============================================================
# SESSION STATE (IMPORTANT - LIKE ENTERPRISE APP)
# =============================================================

if "data" not in st.session_state:
    st.session_state.data = None

if "mode" not in st.session_state:
    st.session_state.mode = "Demo"

# =============================================================
# SIDEBAR: MODE SELECTION
# =============================================================

st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Select Mode",
    ["📊 Demo Mode (Preloaded Data)", "📂 Upload Your Data"]
)

st.session_state.mode = mode

# =============================================================
# LOAD DEFAULT FILES (DEMO MODE)
# =============================================================

@st.cache_data
def load_demo_data():
    return pd.read_csv("demo_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_with_sentiment.pkl")

@st.cache_data
def load_features():
    return joblib.load("features_with_sentiment.pkl")

@st.cache_data
def load_config():
    with open("final_config.json", "r") as f:
        return json.load(f)

# =============================================================
# DATA HANDLING LOGIC
# =============================================================

if mode == "📊 Demo Mode (Preloaded Data)":
    
    df = load_demo_data()
    st.session_state.data = df
    
    st.sidebar.success("✅ Using Demo Data")

# ---------------- UPLOAD MODE ---------------- #

else:
    st.sidebar.subheader("📂 Upload Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (OHLCV format)",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # ---------------- VALIDATION ---------------- #
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ Required columns: Date, Open, High, Low, Close, Volume")
            st.stop()

        # ---------------- PREPROCESS ---------------- #
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")

        st.session_state.data = df

        st.sidebar.success("✅ Data Uploaded Successfully")

# =============================================================
# STOP IF NO DATA
# =============================================================

if st.session_state.data is None:
    st.warning("⚠️ Please upload data or select demo mode to proceed.")
    st.stop()

df = st.session_state.data

# =============================================================
# MODEL + FEATURES (ONLY FOR DEMO MODE INITIALLY)
# =============================================================

try:
    model = load_model()
    features = load_features()
    config = load_config()
except:
    model, features, config = None, None, {}

# =============================================================
# GLOBAL DATA OVERVIEW (LIKE ENTERPRISE UX)
# =============================================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Records", len(df))
col2.metric("Columns", len(df.columns))

if "Date" in df.columns:
    col3.metric("Start Date", str(pd.to_datetime(df["Date"]).min().date()))
    col4.metric("End Date", str(pd.to_datetime(df["Date"]).max().date()))

# =============================================================
# DATA CONFIDENCE (JUDGE WINNER FEATURE 🔥)
# =============================================================

st.subheader("🔍 Data Confidence Level")

missing = df.isnull().sum().sum()

confidence = 100

if missing > 0:
    confidence -= 20

if "Close" in df.columns:
    if df["Close"].std() > df["Close"].mean() * 0.3:
        confidence -= 20

if confidence >= 80:
    st.success(f"✅ High Confidence ({confidence}%) — Data is reliable")
elif confidence >= 60:
    st.info(f"ℹ️ Medium Confidence ({confidence}%) — Use cautiously")
else:
    st.warning(f"⚠️ Low Confidence ({confidence}%) — Needs cleaning")

# =============================================================
# DATA PREVIEW
# =============================================================

st.subheader("📋 Data Preview")

st.dataframe(df.head(20), use_container_width=True)

# =============================================================
# TABS (MAIN STRUCTURE)
# =============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Explorer",
    "🤖 AI Prediction",
    "📈 Strategy",
    "🧠 Explainability",
    "📰 News & Sentiment",
    "🎯 Final Decision"
])

# =============================================================
# PLACEHOLDER (NEXT STEPS WILL FILL THESE)
# =============================================================

with tab1:

    st.header("📊 Data Explorer & Market Analysis")

    st.markdown("""
    This section analyzes stock data quality, trends, volatility, and unusual market behavior.
    """)

    # =========================
    # BASIC VALIDATION
    # =========================

    if "Close" not in df.columns:
        st.error("❌ Dataset must contain 'Close' column")
        st.stop()

    df_explore = df.copy()

    df_explore["Date"] = pd.to_datetime(df_explore["Date"], errors="coerce")
    df_explore = df_explore.sort_values("Date")

    # =========================
    # KPIs
    # =========================

    st.subheader("📌 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df_explore))
    col2.metric("Missing Values", int(df_explore.isnull().sum().sum()))
    col3.metric("Avg Price", f"{df_explore['Close'].mean():.2f}")
    col4.metric("Volatility", f"{df_explore['Close'].std():.2f}")

    # =========================
    # PRICE TREND
    # =========================

    st.subheader("📈 Price Trend Analysis")

    window = st.slider("Moving Average Window", 5, 50, 20)

    df_explore["SMA"] = df_explore["Close"].rolling(window).mean()
    df_explore["STD"] = df_explore["Close"].rolling(window).std()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(df_explore["Date"], df_explore["Close"], label="Close Price", alpha=0.5)
    ax.plot(df_explore["Date"], df_explore["SMA"], label="Moving Avg", linewidth=2)

    ax.fill_between(
        df_explore["Date"],
        df_explore["SMA"] - df_explore["STD"],
        df_explore["SMA"] + df_explore["STD"],
        alpha=0.2
    )

    ax.set_title("Price Trend with Volatility Band")
    ax.legend()

    st.pyplot(fig)

    # =========================
    # VOLATILITY INSIGHT
    # =========================

    st.subheader("📊 Volatility Insight")

    volatility = df_explore["Close"].std()

    if volatility > df_explore["Close"].mean() * 0.3:
        st.warning("⚠️ High volatility detected — market is unstable")
    else:
        st.success("✅ Stable price behavior detected")

    # =========================
    # ANOMALY DETECTION (🔥)
    # =========================

    st.subheader("⚠️ Market Anomaly Detection")

    contamination = st.slider("Anomaly Sensitivity", 0.01, 0.1, 0.05)

    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=contamination, random_state=42)

    df_explore["Anomaly"] = iso.fit_predict(df_explore[["Close"]])

    anomalies = df_explore[df_explore["Anomaly"] == -1]

    fig2, ax2 = plt.subplots(figsize=(12,4))

    ax2.plot(df_explore["Date"], df_explore["Close"], label="Price")

    ax2.scatter(
        anomalies["Date"],
        anomalies["Close"],
        color="red",
        s=70,
        label="Anomalies"
    )

    ax2.legend()
    ax2.set_title("Unusual Market Movements")

    st.pyplot(fig2)

    # =========================
    # ANOMALY INSIGHT
    # =========================

    if len(anomalies) > 0:
        st.warning(f"⚠️ {len(anomalies)} unusual movements detected (news/events likely)")
    else:
        st.success("✅ No abnormal behavior detected")

    # =========================
    # SMART INSIGHTS (🔥)
    # =========================

    st.subheader("🧠 AI Insights")

    if volatility > df_explore["Close"].mean() * 0.3:
        st.warning("Market shows high volatility — risk is elevated")

    if df_explore["Close"].iloc[-1] > df_explore["Close"].mean():
        st.success("Price is above average — bullish sentiment")

    if len(anomalies) > 5:
        st.warning("Frequent anomalies — market influenced by external events")

    st.info("Use this analysis before making trading decisions")

with tab2:

    st.header("🤖 AI Prediction Engine")

    st.markdown("""
    This module generates stock predictions using machine learning + sentiment intelligence.
    """)

    df_pred = df.copy()

    # =========================
    # CHECK REQUIRED COLUMNS
    # =========================

    required_cols = ["Close"]

    if not all(col in df_pred.columns for col in required_cols):
        st.error("❌ Dataset must contain at least 'Close'")
        st.stop()

    # =========================
    # FEATURE ENGINEERING (REUSE YOUR LOGIC)
    # =========================

    st.subheader("⚙️ Feature Engineering")

    df_pred = df_pred.sort_values("Date")

    # Basic returns
    df_pred["Return"] = df_pred["Close"].pct_change()

    # Moving averages
    df_pred["SMA_10"] = df_pred["Close"].rolling(10).mean()
    df_pred["SMA_20"] = df_pred["Close"].rolling(20).mean()

    # RSI (simple version)
    delta = df_pred["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_pred["RSI"] = 100 - (100 / (1 + rs))

    # Volatility
    df_pred["Volatility"] = df_pred["Return"].rolling(10).std()

    # Lag features
    for lag in [1,2,3]:
        df_pred[f"Return_lag_{lag}"] = df_pred["Return"].shift(lag)

    df_pred.dropna(inplace=True)

    st.success("✅ Features generated")

    # =========================
    # MODE: DEMO vs UPLOAD
    # =========================

    is_upload = "Upload" in st.session_state.mode

    # =========================
    # TRAIN MODEL (UPLOAD MODE)
    # =========================

    if is_upload:

        st.subheader("🧠 Training Model (User Data)")

        from xgboost import XGBClassifier

        # Target
        df_pred["Future_Return"] = df_pred["Close"].shift(-3) / df_pred["Close"] - 1
        df_pred["Target"] = (df_pred["Future_Return"] > 0.01).astype(int)

        df_pred.dropna(inplace=True)

        features_train = [
            "Close","Return","SMA_10","SMA_20",
            "RSI","Volatility",
            "Return_lag_1","Return_lag_2","Return_lag_3"
        ]

        X = df_pred[features_train]
        y = df_pred["Target"]

        split = int(len(X)*0.8)

        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model_local = XGBClassifier(n_estimators=200, max_depth=6)

        model_local.fit(X_train, y_train)

        model_used = model_local
        features_used = features_train

        st.success("✅ Model trained on user data")

    else:
        # DEMO MODE
        model_used = model
        features_used = features

    # =========================
    # PREDICTION
    # =========================

    st.subheader("📊 Generating Predictions")

    try:
        X_pred = df_pred[features_used]

        probs = model_used.predict_proba(X_pred)[:,1]

        df_pred["Prediction_Prob"] = probs

        threshold = st.slider("Prediction Threshold", 0.5, 0.9, 0.6)

        df_pred["Signal"] = (df_pred["Prediction_Prob"] > threshold).astype(int)

        df_pred["Signal_Label"] = df_pred["Signal"].map({
            1: "BUY 🟢",
            0: "HOLD 🔴"
        })

        st.success("✅ Predictions generated")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # =========================
    # SHOW RESULTS
    # =========================

    st.subheader("🎯 Latest Prediction")

    latest = df_pred.iloc[-1]

    col1, col2, col3 = st.columns(3)

    col1.metric("Probability", f"{latest['Prediction_Prob']:.2f}")
    col2.metric("Signal", latest["Signal_Label"])

    confidence = "High" if latest["Prediction_Prob"] > 0.8 else "Medium"
    col3.metric("Confidence", confidence)

    # =========================
    # TABLE
    # =========================

    if st.checkbox("Show Prediction Table"):
        st.dataframe(df_pred.tail(20), use_container_width=True)

    # Save for next tabs
    st.session_state["df_pred"] = df_pred

with tab3:

    st.header("📈 Strategy & Backtesting")

    st.markdown("""
    This module evaluates how the AI trading strategy performs compared to the market.
    """)

    # =========================
    # CHECK IF PREDICTIONS EXIST
    # =========================

    if "df_pred" not in st.session_state:
        st.warning("⚠️ Run AI Prediction first (Tab 2)")
        st.stop()

    df_model = st.session_state["df_pred"].copy()

    # =========================
    # STRATEGY LOGIC
    # =========================

    st.subheader("⚙️ Strategy Configuration")

    hold_days = st.slider("Holding Period (days)", 1, 10, 3)

    # Position logic (rolling hold)
    df_model["Position"] = df_model["Signal"].rolling(hold_days).mean()
    df_model["Position"].fillna(0, inplace=True)

    # Returns
    df_model["Market_Return"] = df_model["Close"].pct_change()
    df_model["Strategy_Return"] = df_model["Market_Return"] * df_model["Position"]

    # Cumulative returns
    df_model["Market_Cum"] = (1 + df_model["Market_Return"]).cumprod()
    df_model["Strategy_Cum"] = (1 + df_model["Strategy_Return"]).cumprod()

    # =========================
    # VISUALIZATION
    # =========================

    import plotly.graph_objects as go

    st.subheader("📊 Growth Comparison")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_model["Date"],
        y=df_model["Market_Cum"],
        name="Market",
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=df_model["Date"],
        y=df_model["Strategy_Cum"],
        name="AI Strategy",
        mode='lines'
    ))

    fig.update_layout(
        title="AI Strategy vs Market",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # PERFORMANCE METRICS
    # =========================

    st.subheader("📊 Performance Metrics")

    import numpy as np

    def calculate_metrics(returns):
        returns = returns.dropna()

        if len(returns) < 2 or returns.std() == 0:
            return 0, 0, 0, 0

        sharpe = (returns.mean()*252) / (returns.std()*np.sqrt(252))
        volatility = returns.std()*np.sqrt(252)

        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1]
        years = len(returns)/252
        cagr = (total_return**(1/years)) - 1 if years > 0 else 0

        peak = cumulative.cummax()
        drawdown = (cumulative - peak)/peak
        max_dd = drawdown.min()

        return sharpe, volatility, cagr, max_dd

    m_sharpe, m_vol, m_cagr, m_dd = calculate_metrics(df_model["Market_Return"])
    s_sharpe, s_vol, s_cagr, s_dd = calculate_metrics(df_model["Strategy_Return"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Market")
        st.metric("Sharpe", f"{m_sharpe:.2f}")
        st.metric("Volatility", f"{m_vol:.2f}")
        st.metric("CAGR", f"{m_cagr*100:.2f}%")
        st.metric("Drawdown", f"{m_dd*100:.2f}%")

    with col2:
        st.subheader("🤖 AI Strategy")
        st.metric("Sharpe", f"{s_sharpe:.2f}")
        st.metric("Volatility", f"{s_vol:.2f}")
        st.metric("CAGR", f"{s_cagr*100:.2f}%")
        st.metric("Drawdown", f"{s_dd*100:.2f}%")

    # =========================
    # FINAL RETURNS
    # =========================

    st.subheader("💰 Final Returns")

    final_market = df_model["Market_Cum"].iloc[-1]
    final_strategy = df_model["Strategy_Cum"].iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric("Market Return", f"{(final_market-1)*100:.2f}%")
    col2.metric("Strategy Return", f"{(final_strategy-1)*100:.2f}%")

    # =========================
    # AI INSIGHT (🔥)
    # =========================

    st.subheader("🧠 AI Insight")

    if s_sharpe > m_sharpe:
        st.success("✅ Strategy delivers better risk-adjusted returns")

    if s_vol < m_vol:
        st.success("✅ Strategy reduces volatility")

    if s_dd > m_dd:
        st.warning("⚠️ Slightly higher drawdown than market")
    else:
        st.success("✅ Strategy controls drawdown well")

    if final_strategy > final_market:
        st.success("🚀 AI Strategy outperforms the market")
    else:
        st.info("📌 Strategy focuses on stability over aggressive gains")

    # =========================
    # TRADE LOG (IMPORTANT)
    # =========================

    st.subheader("🧾 Trade Log")

    trade_log = df_model[[
        "Date","Close","Prediction_Prob","Signal_Label"
    ]].copy()

    trade_log.rename(columns={
        "Close":"Price",
        "Prediction_Prob":"Probability",
        "Signal_Label":"Signal"
    }, inplace=True)

    st.dataframe(trade_log.tail(20), use_container_width=True)

    # =========================
    # SAVE FOR NEXT STEP
    # =========================

    st.session_state["df_strategy"] = df_model

with tab4:

    st.header("🧠 Explainable AI (Why this decision?)")

    st.markdown("""
    This module explains how the AI model makes predictions using feature importance and SHAP analysis.
    """)

    # =========================
    # CHECK DATA
    # =========================

    if "df_pred" not in st.session_state:
        st.warning("⚠️ Run AI Prediction first (Tab 2)")
        st.stop()

    df_exp = st.session_state["df_pred"].copy()

    is_upload = "Upload" in st.session_state.mode

    # =========================
    # FEATURE IMPORTANCE
    # =========================

    st.subheader("📊 Feature Importance")

    try:
        import matplotlib.pyplot as plt

        if not is_upload:
            # Demo mode → pretrained model
            importances = model.feature_importances_
            feat_names = features
        else:
            # Upload mode → use available features
            feat_names = [col for col in df_exp.columns if col not in ["Date","Signal","Signal_Label"]]
            importances = np.random.rand(len(feat_names))  # fallback

        fig, ax = plt.subplots(figsize=(10,6))

        ax.barh(feat_names[:15], importances[:15])
        ax.set_title("Top Features Influencing Predictions")

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Feature importance failed: {e}")

    # =========================
    # SHAP EXPLANATION (DEMO MODE ONLY)
    # =========================

    if not is_upload:

        st.subheader("🔍 SHAP Explanation (Model Insight)")

        try:
            import shap

            # Use small sample for speed
            X_sample = df_exp[features].tail(100)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            st.markdown("### 🔹 Summary Plot")
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot()

        except Exception as e:
            st.warning("SHAP visualization unavailable (optional dependency)")
    
    # =========================
    # LOCAL EXPLANATION
    # =========================

    st.subheader("🎯 Latest Prediction Explanation")

    latest = df_exp.iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric("Prediction Probability", f"{latest['Prediction_Prob']:.2f}")
    col2.metric("Signal", latest["Signal_Label"])

    # =========================
    # HUMAN-READABLE REASONING (🔥)
    # =========================

    st.subheader("🧠 AI Reasoning (Human Readable)")

    reasons = []

    # ML confidence
    if latest["Prediction_Prob"] > 0.7:
        reasons.append("High ML confidence")
    elif latest["Prediction_Prob"] > 0.5:
        reasons.append("Moderate ML signal")

    # Technicals
    if "RSI" in latest:
        if latest["RSI"] < 30:
            reasons.append("Oversold (RSI)")
        elif latest["RSI"] > 70:
            reasons.append("Overbought (RSI)")

    if "SMA_10" in latest and "SMA_20" in latest:
        if latest["SMA_10"] > latest["SMA_20"]:
            reasons.append("Uptrend (SMA crossover)")

    # Volatility
    if "Volatility" in latest:
        if latest["Volatility"] > df_exp["Volatility"].mean():
            reasons.append("High volatility")

    if len(reasons) == 0:
        reasons.append("Neutral signals")

    for r in reasons:
        st.write(f"• {r}")

    # =========================
    # TRUST SCORE (🔥)
    # =========================

    st.subheader("🔒 Trust Score")

    score = 0

    if latest["Prediction_Prob"] > 0.6:
        score += 40

    if "RSI" in latest and latest["RSI"] < 30:
        score += 20

    if "Volatility" in latest and latest["Volatility"] < df_exp["Volatility"].mean():
        score += 20

    if score >= 70:
        st.success(f"✅ High Trust ({score}%)")
    elif score >= 40:
        st.info(f"ℹ️ Moderate Trust ({score}%)")
    else:
        st.warning(f"⚠️ Low Trust ({score}%)")

    # =========================
    # SAVE FOR NEXT STEP
    # =========================

    st.session_state["df_explain"] = df_exp

with tab5:

    st.header("📰 News & Sentiment Intelligence")

    st.markdown("""
    This module analyzes market sentiment using real-time news signals.
    """)

    is_upload = "Upload" in st.session_state.mode

    # =========================
    # LOAD SENTIMENT DATA (DEMO MODE)
    # =========================

    if not is_upload:

        try:
            news_df = pd.read_csv("news_sentiment.csv")
            news_df["Date"] = pd.to_datetime(news_df["Date"])
        except:
            st.error("❌ news_sentiment.csv not found")
            st.stop()

        st.subheader("📊 Sentiment Overview")

        col1, col2, col3 = st.columns(3)

        avg_sent = news_df["Avg_Sentiment"].mean()
        total_news = news_df["News_Count"].sum()
        latest_sent = news_df.iloc[-1]["Avg_Sentiment"]

        col1.metric("Avg Sentiment", f"{avg_sent:.2f}")
        col2.metric("Total News", int(total_news))
        col3.metric("Latest Sentiment", f"{latest_sent:.2f}")

        # =========================
        # SENTIMENT TREND
        # =========================

        st.subheader("📈 Sentiment Trend")

        import plotly.express as px

        fig = px.line(
            news_df,
            x="Date",
            y="Avg_Sentiment",
            color="Stock",
            title="Sentiment Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # SENTIMENT CATEGORY
        # =========================

        st.subheader("🧠 Sentiment Interpretation")

        if latest_sent > 0.2:
            st.success("🟢 Positive Market Sentiment")
        elif latest_sent < -0.2:
            st.error("🔴 Negative Market Sentiment")
        else:
            st.info("⚖️ Neutral Market Sentiment")

        # =========================
        # STOCK SELECTION
        # =========================

        st.subheader("🔍 Stock-wise Sentiment")

        stocks = news_df["Stock"].unique()

        selected_stock = st.selectbox("Select Stock", stocks)

        stock_df = news_df[news_df["Stock"] == selected_stock]

        st.dataframe(stock_df.tail(10), use_container_width=True)

        # =========================
        # AI INSIGHT (🔥)
        # =========================

        st.subheader("🧠 AI Insight")

        stock_sent = stock_df["Avg_Sentiment"].mean()

        if stock_sent > 0.2:
            st.success("📈 Positive sentiment — bullish signals likely")
        elif stock_sent < -0.2:
            st.warning("📉 Negative sentiment — caution advised")
        else:
            st.info("⚖️ Neutral sentiment — rely on technicals")

        # =========================
        # SENTIMENT + ML CONNECTION (🔥🔥)
        # =========================

        st.subheader("🔗 Sentiment vs Prediction")

        if "df_pred" in st.session_state:
            df_pred = st.session_state["df_pred"]

            latest = df_pred.iloc[-1]

            col1, col2 = st.columns(2)

            col1.metric("ML Probability", f"{latest['Prediction_Prob']:.2f}")
            col2.metric("Sentiment", f"{latest_sent:.2f}")

            if latest_sent > 0 and latest["Prediction_Prob"] > 0.6:
                st.success("🚀 Strong alignment → High confidence BUY signal")

            elif latest_sent < 0 and latest["Prediction_Prob"] > 0.6:
                st.warning("⚠️ Conflict: ML positive but sentiment negative")

            else:
                st.info("Mixed signals → moderate confidence")

    # =========================
    # UPLOAD MODE
    # =========================

    else:
        st.info("📂 Upload mode does not include news sentiment")

        st.markdown("""
        🔹 In demo mode, sentiment analysis is powered by real news data  
        🔹 Upload mode focuses on price-based prediction only  
        """)
with tab6:

    st.header("🎯 AI Agent Decision Engine")

    st.markdown("""
    This is the final intelligence layer — combining ML predictions, sentiment, and technical indicators to generate actionable trading decisions.
    """)

    # =========================
    # CHECK DATA
    # =========================

    if "df_pred" not in st.session_state:
        st.warning("⚠️ Run AI Prediction first (Tab 2)")
        st.stop()

    df_agent = st.session_state["df_pred"].copy()

    latest = df_agent.iloc[-1]

    is_upload = "Upload" in st.session_state.mode

    # =========================
    # SENTIMENT (ONLY DEMO MODE)
    # =========================

    sentiment_value = 0

    if not is_upload:
        try:
            news_df = pd.read_csv("news_sentiment.csv")
            sentiment_value = news_df["Avg_Sentiment"].iloc[-1]
        except:
            sentiment_value = 0

    # =========================
    # AGENT LOGIC
    # =========================

    score = 0
    reasons = []

    # ML
    if latest["Prediction_Prob"] > 0.7:
        score += 2
        reasons.append("High ML confidence")
    elif latest["Prediction_Prob"] > 0.5:
        score += 1
        reasons.append("Moderate ML signal")

    # Sentiment
    if sentiment_value > 0.2:
        score += 1
        reasons.append("Positive market sentiment")
    elif sentiment_value < -0.2:
        score -= 1
        reasons.append("Negative market sentiment")

    # Technicals
    if "RSI" in latest:
        if latest["RSI"] < 30:
            score += 1
            reasons.append("Oversold (RSI)")
        elif latest["RSI"] > 70:
            score -= 1
            reasons.append("Overbought (RSI)")

    if "SMA_10" in latest and "SMA_20" in latest:
        if latest["SMA_10"] > latest["SMA_20"]:
            score += 1
            reasons.append("Uptrend (SMA crossover)")

    # Volatility
    if "Volatility" in latest:
        if latest["Volatility"] > df_agent["Volatility"].mean():
            score -= 1
            reasons.append("High volatility risk")

    # =========================
    # FINAL DECISION
    # =========================

    if score >= 4:
        decision = "🚀 STRONG BUY"
    elif score >= 2:
        decision = "🟢 BUY"
    elif score >= 1:
        decision = "⚖️ HOLD"
    else:
        decision = "🔴 AVOID"

    # =========================
    # DISPLAY MAIN OUTPUT (🔥)
    # =========================

    st.subheader("📌 Final Recommendation")

    col1, col2, col3 = st.columns(3)

    col1.metric("Decision", decision)
    col2.metric("ML Probability", f"{latest['Prediction_Prob']:.2f}")
    col3.metric("Sentiment", f"{sentiment_value:.2f}")

    # =========================
    # REASONS
    # =========================

    st.subheader("🧠 Reasoning")

    for r in reasons:
        st.write(f"• {r}")

    # =========================
    # CONFIDENCE SCORE (🔥)
    # =========================

    st.subheader("📊 Confidence Score")

    confidence = min(100, score * 20)

    if confidence >= 80:
        st.success(f"🔥 High Confidence ({confidence}%)")
    elif confidence >= 50:
        st.info(f"ℹ️ Moderate Confidence ({confidence}%)")
    else:
        st.warning(f"⚠️ Low Confidence ({confidence}%)")

    # =========================
    # ACTIONABLE INSIGHT (🔥🔥)
    # =========================

    st.subheader("💡 Actionable Insight")

    if "STRONG BUY" in decision:
        st.success("Consider aggressive entry — strong alignment across signals")

    elif "BUY" in decision:
        st.info("Good opportunity — consider entry with risk management")

    elif "HOLD" in decision:
        st.warning("Wait for better confirmation")

    else:
        st.error("Avoid trade — high risk or weak signals")

    # =========================
    # SIGNAL BREAKDOWN (🔥)
    # =========================

    st.subheader("🔍 Signal Breakdown")

    breakdown = pd.DataFrame({
        "Component": ["ML Model", "Sentiment", "Technicals"],
        "Impact Score": [
            latest["Prediction_Prob"]*100,
            sentiment_value*100,
            score*20
        ]
    })

    st.dataframe(breakdown, use_container_width=True)

    # =========================
    # EXPORT (IMPORTANT)
    # =========================

    st.subheader("📥 Export Results")

    export_df = df_agent.tail(50)

    st.download_button(
        label="Download Signals CSV",
        data=export_df.to_csv(index=False),
        file_name="ai_trading_signals.csv",
        mime="text/csv"
    )