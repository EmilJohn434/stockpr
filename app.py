import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd

TODAY = date.today().strftime("%Y-%m-%d")
start_date = None

def load_data(ticker):
    if ticker:
        data = yf.download(ticker, start_date, TODAY)
        data.reset_index(inplace=True)
        return data

st.title('Stock Market Predictor')

menu = ["Predict Single Stock", "Compare Stocks"]
choice = st.sidebar.selectbox("Select page", menu)

if choice == "Predict Single Stock":
    selected_stock = st.text_input('Select a stock ticker for prediction (refer to yfinance for ticker)')
    start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)
    start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")
    n_years = st.slider('How many years into the future?', 1, 4)
    period = n_years * 365

    if selected_stock:
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
        changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

        # Daily data for exponential smoothing
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        daily_data = data.resample('D').interpolate()
        daily_data['Close_rolling'] = daily_data['Close'].ewm(alpha=1 - smoothing_factor).mean()

        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name="Stock Open"))
            fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name="Stock Close"))
            fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name="Close (Exponential Smoothing)"))
            fig.update_layout(
                title_text='Stock History',
                xaxis_rangeslider_visible=True,
                height=600,
                width=900,
                yaxis=dict(tickvals=[0, 100, 200, 300, 400])
            )
            st.plotly_chart(fig)

        plot_raw_data()

        df_train = daily_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

        m = Prophet(
            growth='linear',
            changepoint_prior_scale=changepoint_prior_scale
        )

        m.fit(df_train)

        future = m.make_future_dataframe(periods=period, freq='D')

        forecast = m.predict(future)

        if n_years == 1:
            st.subheader(f'Forecast Plot for {n_years} Year')
        else:
            st.subheader(f'Forecast Plot for {n_years} Years')

        fig1 = plot_plotly(m, forecast)

        fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))

        num_data_points = len(forecast)
        marker_size = max(4, 200 // num_data_points)

        fig1.update_traces(mode='markers+lines', marker=dict(size=marker_size, color='black', opacity=0.7),
                           selector=dict(name='yhat_lower,yhat_upper'))

        fig1.update_layout(
            title_text=f'Forecast Plot for {n_years} Years',
            xaxis_rangeslider_visible=True,
            height=600,
            width=900,
            yaxis=dict(tickvals=[0, 100, 200, 300, 400]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig1)

elif choice == "Compare Stocks":
    selected_stocks = st.multiselect('Select stock tickers for comparison (refer to yfinance for tickers)',
                                     ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

    if selected_stocks:
        data_load_state = st.text('Loading data...')
        data = pd.DataFrame()
        for stock in selected_stocks:
            stock_data = load_data(stock)
            stock_data['Stock'] = stock
            data = pd.concat([data, stock_data], ignore_index=True)
        data_load_state.text('Loading data... done!')

        def plot_comparison():
            fig = go.Figure()
            for stock in selected_stocks:
                stock_data = data[data['Stock'] == stock]
                fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], name=stock))

            fig.update_layout(
                title_text='Stock Comparison',
                xaxis_rangeslider_visible=True,
                height=600,
                width=900,
                yaxis=dict(tickvals=[0, 100, 200, 300, 400])
            )
            st.plotly_chart(fig)

        plot_comparison()

footer = """
<style>
.footer {
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
    <p>Made by Emil, Adhip and Naren</p>
    <p>This app is made for educational purposes only. Data it provides is not 100% accurate.</p>
    <p>Analyze stocks before investing.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
