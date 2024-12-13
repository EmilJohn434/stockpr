import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
from datetime import date

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

if choice == "Predict Single Stock":
    # Your existing stock prediction code here
    pass

elif choice == "Compare Stocks":
    # Your existing stock comparison code here
    pass

elif choice == "Predict Gold Prices":
    gold_data = load_data('^GOLD')
    gold_data['Date'] = pd.to_datetime(gold_data['Date'])
    gold_data.set_index('Date', inplace=True)
    gold_data['Close_rolling'] = gold_data['Close'].ewm(alpha=0.9).mean()

    df_train = gold_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(growth='linear')
    m.fit(df_train)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))
    st.plotly_chart(fig1)

elif choice == "Predict Silver Prices":
    silver_data = load_data('^SILVER')
    silver_data['Date'] = pd.to_datetime(silver_data['Date'])
    silver_data.set_index('Date', inplace=True)
    silver_data['Close_rolling'] = silver_data['Close'].ewm(alpha=0.9).mean()

    df_train = silver_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(growth='linear')
    m.fit(df_train)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))
    st.plotly_chart(fig1)

elif choice == "Predict Crude Oil Prices":
    crude_data = load_data('CL=F')
    crude_data['Date'] = pd.to_datetime(crude_data['Date'])
    crude_data.set_index('Date', inplace=True)
    crude_data['Close_rolling'] = crude_data['Close'].ewm(alpha=0.9).mean()

    df_train = crude_data[['Close_rolling']].reset_index().rename(columns={"Date": "ds", "Close_rolling": "y"})

    m = Prophet(growth='linear')
    m.fit(df_train)

    future = m.make_future_dataframe(periods=365, freq='D')
    forecast = m.predict(future)

    fig1 = plot_plotly(m, forecast)
    fig1.update_traces(mode='lines', line=dict(color='blue', width=2), selector=dict(name='yhat'))
    st.plotly_chart(fig1)

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
