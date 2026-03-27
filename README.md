# 🚀 AI Stock Intelligence & Agentic Trading System

## 📌 Overview

This project is an AI-powered investment intelligence platform designed for retail investors.  
It transforms raw financial data into **actionable trading decisions** using Machine Learning, Sentiment Analysis, and Agentic AI.

🔗 Live Demo: https://ai-stock-intelligence-agentic-trading-system.streamlit.app/
---

## 🎯 Problem Statement

Retail investors often rely on tips and lack tools to:
- Analyze market data
- Interpret news sentiment
- Make informed trading decisions

This project builds an **intelligence layer** that converts data into **explainable, actionable signals**.

---

## 🧠 Key Features

### 🤖 AI Prediction Engine
- Predicts stock movement using XGBoost
- Uses technical indicators + lag features

### 📰 Sentiment Analysis
- Converts news headlines into sentiment scores
- Integrates market psychology into predictions

### 🎯 Agentic AI Decision Engine (Core Innovation)
- Multi-step reasoning:
  - ML prediction
  - Sentiment validation
  - Technical confirmation
- Outputs:
  - BUY / HOLD / AVOID
  - Human-readable reasoning

### 📈 Strategy & Backtesting
- Compares AI strategy vs market
- Metrics:
  - Sharpe Ratio
  - CAGR
  - Drawdown
  - Volatility

### 🧠 Explainable AI
- Feature importance
- SHAP-based interpretation
- Trust scoring system

### 📊 Interactive Streamlit Dashboard
- Upload your own dataset OR use demo data
- Visual analytics + decision insights
- Export trading signals

---

## ⚙️ Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas / NumPy
- SHAP
- Plotly / Matplotlib

---

## 🏗️ System Architecture
Data → Feature Engineering → ML Model → Sentiment → Agent Logic → Decision

---

## 📊 Model Performance

| Metric | Value |
|------|------|
| Accuracy | ~62% |
| Sharpe Ratio | ~1.7 |
| Max Drawdown | ~-4.4% |
| Volatility | ~0.05 |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/your-username/AI-Stock-Intelligence.git
cd AI-Stock-Intelligence
```
2. Install dependencies
```
pip install -r requirements.txt
```
4. Run the app
```
streamlit run app.py
```
### 3. Project Structure
app.py                  # Main application
model_with_sentiment.pkl
features_with_sentiment.pkl
demo_data.csv
news_sentiment.csv
agent_signals.csv

🎯 Unique Selling Points
🔥 Agentic AI (multi-step reasoning)
🔥 Combines technical + sentiment signals
🔥 Explainable and transparent predictions
🔥 Real trading strategy with backtesting
🔥 Interactive UI for users

📌 Future Enhancements
Portfolio optimization
Real-time API integration
Reinforcement learning trading
LSTM / Transformer models
