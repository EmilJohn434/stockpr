import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
from datetime import date
import streamlit as st

TODAY = date.today().strftime("%Y-%m-%d")
start_date = None

def load_data(ticker):
    if ticker:
        data = yf.download(ticker, start_date, TODAY)
        data.reset_index(inplace=True)
        return data

st.title('Market Predictor')

menu = ["Predict Single Stock", "Compare Stocks", "Predict Gold Prices", "Predict Silver Prices", "Predict Crude Oil Prices"]
choice = st.sidebar.selectbox("Select page", menu)

def plot_raw_data(daily_data, ticker_name, y_axis_values, y_axis_range):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Open'], name=f"{ticker_name} Open"))
    fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close'], name=f"{ticker_name} Close"))
    fig.add_trace(go.Scatter(x=daily_data.index, y=daily_data['Close_rolling'], name=f"Close (Exponential Smoothing)"))
    fig.update_layout(
        title_text=f'{ticker_name} History',
        xaxis_rangeslider_visible=True,
        height=600,
        width=900,
        yaxis=dict(
            tickvals=y_axis_values,
            range=y_axis_range,
        )
    )
    st.plotly_chart(fig)

def predict_prices(data, ticker_name, n_years, smoothing_factor, changepoint_prior_scale, y_axis_values, y_axis_range):
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    daily_data = data.resample('D').interpolate()
    daily_data['Close_rolling'] = daily_data['Close'].ewm(alpha=1 - smoothing_factor).mean()

    plot_raw_data(daily_data, ticker_name, y_axis_values, y_axis_range)

    df_train = daily_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    if df_train.dropna().shape[0] < 2:
        st.error("Not enough data to train the model. Please select a different ticker or time period.")
    else:
        m = Prophet(
            growth='linear',
            changepoint_prior_scale=changepoint_prior_scale
        )

        m.fit(df_train)

        period = n_years * 365
        future = m.make_future_dataframe(periods=period, freq='D')
        forecast = m.predict(future)

        st.subheader(f'Forecast Plot for {ticker_name} ({n_years} Years)')

        fig1 = plot_plotly(m, forecast)

        fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))

        num_data_points = len(forecast)
        marker_size = max(4, 200 // num_data_points)

        fig1.update_traces(mode='markers+lines', marker=dict(size=marker_size, color='black', opacity=0.7),
                           selector=dict(name='yhat_lower,yhat_upper'))

        fig1.update_layout(
            title_text=f'Forecast Plot for {ticker_name} ({n_years} Years)',
            xaxis_rangeslider_visible=True,
            height=600,
            width=900,
            yaxis=dict(
                tickvals=y_axis_values,
                range=y_axis_range,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig1)

if choice == "Predict Single Stock":
    selected_stock = st.text_input('Select a stock ticker for prediction (refer to yfinance for ticker)')
    start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)
    start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")
    n_years = st.slider('How many years into the future?', 1, 4)
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    if selected_stock:
        data_load_state = st.text('Loading data...')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!')

        predict_prices(data, selected_stock, n_years, smoothing_factor, changepoint_prior_scale,
                       y_axis_values=[0, 100, 200, 300, 400], y_axis_range=[0, 500])

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
                yaxis=dict(
                    tickvals=[0, 100, 200, 300, 400],
                    range=[0, 500],  # Adjust this range as needed
                )
            )
            st.plotly_chart(fig)

        plot_comparison()

elif choice == "Predict Gold Prices":
    gold_data = load_data('GC=F')  # Use the correct ticker for Gold Futures
    start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)
    start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")
    n_years = st.slider('How many years into the future?', 1, 4)
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    predict_prices(gold_data, 'Gold', n_years, smoothing_factor, changepoint_prior_scale,
                   y_axis_values=[0, 600, 1200, 1800, 2400, 3000], y_axis_range=[0, 3000])

elif choice == "Predict Silver Prices":
    silver_data = load_data('SI=F')  # Use the correct ticker for Silver Futures
    start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)
    start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")
    n_years = st.slider('How many years into the future?', 1, 4)
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    predict_prices(silver_data, 'Silver', n_years, smoothing_factor, changepoint_prior_scale,
                   y_axis_values=[0, 10, 20, 30, 40, 50], y_axis_range=[0, 50])

elif choice == "Predict Crude Oil Prices":
    crude_data = load_data('CL=F')  # Use the correct ticker for Crude Oil Futures
    start_year = st.slider('Select the start year for prediction', 2010, date.today().year - 1, 2020)
    start_date = date(start_year, 1, 1).strftime("%Y-%m-%d")
    n_years = st.slider('How many years into the future?', 1, 4)
    smoothing_factor = st.slider('Smoothing Factor (increase for smoother graph)', 0.1, 0.95, 0.9, 0.05)
    changepoint_prior_scale = st.slider('Flexibility of Trend', 0.1, 10.0, 0.5, 0.1, format="%.1f")

    predict_prices(crude_data, 'Crude Oil', n_years, smoothing_factor, changepoint_prior_scale,
                   y_axis_values=[0, 40, 80, 120, 160], y_axis_range=[0, 160])

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
