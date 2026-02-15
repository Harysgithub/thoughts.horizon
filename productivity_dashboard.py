import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO  # For CSV string loading

# Config
st.set_page_config(page_title="Hary's Productivity Forecast Dashboard", layout="wide")
st.title("🚀 Productivity Forecast: Plan Your Focus Weeks Ahead")

# Sidebar for inputs
st.sidebar.header("Forecast Tweaks")
forecast_days = st.sidebar.slider("Days Ahead", 7, 28, 28)  # Default 4 weeks
caffeine_plan = st.sidebar.slider("Planned Avg Caffeine (0-2)", 0, 2, 1)  # Example tweak

# Load data (upload CSV or use local path)
uploaded_file = st.sidebar.file_uploader("Upload CSV (or use sample)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Full fallback sample (your 90-day data with diet regressors)
    sample_data = """ds,focus_hours,sleep_hours,bedtime_hour,brainfm_minutes,caffeine_level,mood_score,physical_activity,healthy_meal,sugar_level
2025-12-01,4.2,8.3,27,59,0,4,0,0,1
2025-12-02,8.316450977861956,4.6,28,81,2,4,0,1,1
2025-12-03,8.392986465532706,5.9,27,77,2,3,0,1,2
2025-12-04,8.0,4.2,28,89,1,3,0,0,2
2025-12-05,4.482439442632957,4.2,28,94,2,1,0,1,2
2025-12-06,2.0,6.4,24,0,0,2,1,0,1
2025-12-07,2.9,8.6,24,2,1,3,0,0,1
2025-12-08,2.2,9.1,24,18,0,4,0,0,1
2025-12-09,7.7,6.2,26,10,1,5,0,0,1
2025-12-10,8.0,6.5,27,40,1,4,0,0,1
2025-12-11,8.0,5.8,29,38,1,3,0,0,0
2025-12-12,6.588357169147174,6.3,27,41,2,3,0,1,1
2025-12-13,2.0,7.6,24,29,2,2,0,0,1
2025-12-14,2.0,6.3,24,3,0,2,0,0,0
2025-12-15,2.471605192586819,8.3,24,21,2,3,0,1,2
2025-12-16,6.6,8.2,25,63,1,5,1,0,0
2025-12-17,5.4208083902625495,7.2,28,43,1,4,1,1,0
2025-12-18,6.530233400413228,5.9,26,109,1,4,0,1,2
2025-12-19,4.528332888226778,8.0,24,44,2,4,1,1,1
2025-12-20,2.0,7.6,26,17,0,4,0,0,1
2025-12-21,4.6,8.1,27,14,0,5,0,0,0
2025-12-22,5.8,6.6,27,40,2,5,0,0,1
2025-12-23,6.6,6.7,26,64,2,3,1,0,2
2025-12-24,5.821356594538068,6.3,25,81,1,3,0,1,0
2025-12-25,7.195692689088129,6.9,27,98,2,4,1,1,1
2025-12-26,4.953462593290836,7.7,26,38,1,4,0,1,1
2025-12-27,2.806981580720748,5.9,26,6,0,4,0,1,0
2025-12-28,3.5,6.5,25,71,0,4,0,0,0
2025-12-29,4.044340544776681,7.2,26,23,0,4,0,1,1
2025-12-30,6.5,5.2,26,84,0,4,1,0,0
2025-12-31,8.28455643243202,5.5,25,43,0,3,0,1,2
2026-01-01,5.6,7.6,28,85,2,3,1,0,2
2026-01-02,4.6,6.9,26,26,1,3,1,0,1
2026-01-03,3.6,6.7,27,26,0,3,0,0,1
2026-01-04,4.5,7.4,26,8,0,3,0,0,1
2026-01-05,5.588682854033966,5.9,28,27,2,4,1,1,2
2026-01-06,6.820521313141166,6.2,26,105,1,3,1,1,1
2026-01-07,8.441044279115454,6.8,25,90,2,3,0,1,1
2026-01-08,8.0,4.2,27,67,2,3,0,0,2
2026-01-09,7.6,6.6,25,55,1,2,0,0,0
2026-01-10,6.0,4.2,27,96,2,4,0,0,2
2026-01-11,3.1,4.2,27,76,2,3,0,0,2
2026-01-12,5.3,5.8,27,53,1,2,0,0,0
2026-01-13,4.042050421780179,9.0,24,50,0,2,1,1,2
2026-01-14,2.5,9.5,25,0,0,3,1,0,1
2026-01-15,4.0,6.5,26,39,1,5,0,0,1
2026-01-16,5.4,6.7,26,58,2,4,0,0,1
2026-01-17,4.323785303073428,5.2,26,23,0,3,0,1,1
2026-01-18,4.9,8.4,26,20,0,4,0,0,0
2026-01-19,4.311605425737835,7.2,28,9,0,4,0,1,0
2026-01-20,7.8329238882225996,4.6,26,75,1,4,0,1,2
2026-01-21,2.0,7.2,26,0,0,3,0,0,1
2026-01-22,4.0,6.0,28,23,0,4,0,0,1
2026-01-23,2.4792271976810696,6.6,24,15,1,3,1,1,0
2026-01-24,3.3,6.2,27,71,0,3,0,0,2
2026-01-25,3.9,5.8,27,58,0,2,0,0,2
2026-01-26,4.0,6.8,26,38,1,3,1,0,1
2026-01-27,6.9,6.8,25,65,1,5,0,0,1
2026-01-28,5.7,7.7,24,64,0,3,0,0,0
2026-01-29,6.925261320337449,6.3,25,85,2,3,1,1,1
2026-01-30,5.226362862225405,6.3,25,49,1,4,0,1,0
2026-01-31,2.9,4.2,26,0,0,3,0,0,2
2026-02-01,3.4,6.4,26,24,0,2,0,0,0
2026-02-02,5.97076587200387,6.9,26,26,2,4,1,1,1
2026-02-03,5.5,7.8,27,31,2,3,1,0,1
2026-02-04,5.2,6.0,27,48,2,3,0,0,1
2026-02-05,4.4,7.4,26,49,1,3,1,0,1
2026-02-06,4.851575711734357,6.0,25,54,2,4,1,1,2
2026-02-07,3.9,7.8,27,19,0,3,0,0,0
2026-02-08,6.4960148803091835,4.5,25,120,0,4,0,1,2
2026-02-09,5.3,5.7,26,34,1,1,0,0,1
2026-02-10,5.816760503620249,4.6,28,110,1,3,0,1,1
2026-02-11,5.40325129544409,6.3,28,36,2,3,0,1,1
2026-02-12,7.9,5.6,27,57,2,3,0,0,2
2026-02-13,2.62738600303584,8.5,24,3,2,1,1,1,1
2026-02-14,2.2957940912771244,7.9,25,3,0,3,0,1,1
2026-02-15,2.0,5.3,26,13,0,2,0,0,2
2026-02-16,7.0,6.0,24,98,0,2,0,0,0
2026-02-17,8.0,4.2,28,118,2,2,0,0,2
2026-02-18,5.5,5.1,28,92,0,1,0,0,0
2026-02-19,8.272031366436986,5.9,28,64,2,3,0,1,2
2026-02-20,7.8345336565934955,6.5,27,79,2,2,1,1,2
2026-02-21,2.0,5.4,26,41,2,4,0,0,1
2026-02-22,2.0,8.3,25,12,0,2,0,0,0
2026-02-23,2.998599363610975,8.7,27,26,0,3,0,1,0
2026-02-24,3.8,6.7,24,38,0,3,0,0,1
2026-02-25,5.201755536823112,6.0,24,66,0,4,1,1,0
2026-02-26,5.1,5.3,27,30,2,3,0,0,2
2026-02-27,4.837473713117755,6.3,28,19,2,1,0,1,2
2026-02-28,3.5,6.6,24,54,1,3,0,0,1"""
    df = pd.read_csv(StringIO(sample_data))

df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace=True)

# Fit model (SARIMAX with exog) - Add try/except for errors
y = df['focus_hours']
exog_cols = ['sleep_hours', 'bedtime_hour', 'brainfm_minutes', 'caffeine_level', 'mood_score', 'physical_activity', 'healthy_meal', 'sugar_level']
exog = df[exog_cols].fillna(df[exog_cols].mean())  # Handle any NaNs

# Train/test split (80/20, but ensure enough data)
split = max(int(0.8 * len(df)), 10)  # Min 10 for model
train_y, test_y = y[:split], y[split:]
train_exog, test_exog = exog[:split], exog[split:]

@st.cache_data  # Cache for speed
def fit_model():
    try:
        model = SARIMAX(train_y, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,7))
        return model.fit(disp=False)
    except Exception as e:
        st.error(f"Model fit error: {e}. Try uploading more data.")
        return None

fit = fit_model()
if fit is None:
    st.stop()  # Halt if model fails

# Forecast
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
future_exog = pd.DataFrame({col: df[col].mean() for col in exog_cols}, index=future_dates)
future_exog['caffeine_level'] = caffeine_plan  # Tweak example
forecast = fit.get_forecast(steps=forecast_days, exog=future_exog)
forecast_df = pd.DataFrame({
    'ds': future_dates,
    'yhat': forecast.predicted_mean,
    'yhat_lower': forecast.conf_int().iloc[:, 0],
    'yhat_upper': forecast.conf_int().iloc[:, 1]
}).set_index('ds')

# Combine historical + forecast
full_df = pd.concat([df[['focus_hours']].reset_index(), forecast_df.reset_index()])

# Dashboard Tabs (rest unchanged from previous code)
tab1, tab2, tab3 = st.tabs(["📈 Forecast Overview", "📊 Weekly Breakdown", "🔍 Regressor Insights"])

with tab1:
    st.subheader("Historical Focus + Forecast")
    fig = px.line(full_df, x='ds', y='focus_hours', title="Focus Hours: Past & Predicted",
                  labels={'focus_hours': 'Hours', 'ds': 'Date'})
    forecast_mask = full_df['yhat'].notna()
    fig.add_scatter(x=full_df.loc[forecast_mask, 'ds'], y=full_df.loc[forecast_mask, 'yhat'], mode='lines', name='Forecast')
    fig.add_ribbon(x=full_df.loc[forecast_mask, 'ds'], y=full_df.loc[forecast_mask, 'yhat_lower'], 
                   yupper=full_df.loc[forecast_mask, 'yhat_upper'], name='Uncertainty', line_color='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Historical Focus", f"{y.mean():.1f} hrs")
    with col2:
        st.metric("Predicted Next Week Avg", f"{forecast_df['yhat'].head(7).mean():.1f} hrs")

with tab2:
    st.subheader("Weekly Aggregates")
    df_weekly = df.resample('W').agg({'focus_hours': 'mean', 'mood_score': 'mean', 'sleep_hours': 'mean'})
    forecast_weekly = forecast_df.resample('W').agg({'yhat': 'mean'})
    full_weekly = pd.concat([df_weekly, forecast_weekly]).reset_index()
    
    fig_week = px.bar(full_weekly, x='ds', y=['focus_hours', 'yhat'], barmode='group',
                      title="Weekly Avg Focus: History vs Forecast")
    st.plotly_chart(fig_week, use_container_width=True)
    
    # Highlight peaks/slumps
    next_week_pred = forecast_df['yhat'].head(7).mean()
    if next_week_pred > y.mean():
        st.success(f"💥 Peak Week Alert: Plan deep work (predicted {next_week_pred:.1f} hrs)")
    else:
        st.warning(f"😴 Recovery Week: Focus on rest (predicted {next_week_pred:.1f} hrs)")

with tab3:
    st.subheader("Diet & Habit Impact (Correlation Heatmap)")
    corr_matrix = df[exog_cols + ['focus_hours']].corr()
    fig_heat = px.imshow(corr_matrix, title="Regressor Correlations with Focus", aspect="auto", color_continuous_scale='RdBu')
    st.plotly_chart(fig_heat, use_container_width=True)
    
    st.info("Tweak sidebar sliders to re-forecast—e.g., higher caffeine might spike short-term but crash mood.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built for Hary's productivity biz | Refresh data via upload | Questions? Let's iterate!")
