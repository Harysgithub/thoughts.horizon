import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    # Fallback to sample (your 90-day data; paste full CSV content here if needed)
    sample_data = """ds,focus_hours,sleep_hours,bedtime_hour,brainfm_minutes,caffeine_level,mood_score,physical_activity,healthy_meal,sugar_level
2025-12-01,4.2,8.3,27,59,0,4,0,0,1
..."""  # Abbrev; replace with full CSV string or path
    df = pd.read_csv(pd.StringIO(sample_data))  # Or df = pd.read_csv('your_file.csv')

df['ds'] = pd.to_datetime(df['ds'])
df.set_index('ds', inplace=True)

# Fit model (SARIMAX with exog)
y = df['focus_hours']
exog_cols = ['sleep_hours', 'bedtime_hour', 'brainfm_minutes', 'caffeine_level', 'mood_score', 'physical_activity', 'healthy_meal', 'sugar_level']
exog = df[exog_cols].fillna(df[exog_cols].mean())  # Handle any NaNs

# Train/test split (80/20)
split = int(0.8 * len(df))
train_y, test_y = y[:split], y[split:]
train_exog, test_exog = exog[:split], exog[split:]

@st.cache_data  # Cache for speed
def fit_model():
    model = SARIMAX(train_y, exog=train_exog, order=(1,1,1), seasonal_order=(1,1,1,7))
    return model.fit(disp=False)

fit = fit_model()

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
full_df = pd.concat([df[['focus_hours']], forecast_df]).reset_index()

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📈 Forecast Overview", "📊 Weekly Breakdown", "🔍 Regressor Insights"])

with tab1:
    st.subheader("Historical Focus + 4-Week Forecast")
    fig = px.line(full_df.reset_index(), x='ds', y='focus_hours', title="Focus Hours: Past & Predicted",
                  labels={'focus_hours': 'Hours', 'ds': 'Date'})
    fig.add_scatter(x=full_df[full_df['yhat'].isna() == False]['ds'], y=full_df['yhat'], mode='lines', name='Forecast')
    fig.add_ribbon(x=full_df[full_df['yhat'].isna() == False]['ds'], y=full_df['yhat_lower'], 
                   yupper=full_df['yhat_upper'], name='Uncertainty', line_color='rgba(0,0,0,0)')
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
