import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Customizing Streamlit style
st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    .css-1q9oajh {
        background-color: #172A3A;
        color: #FFFFFF;
    }
    .css-1aumxhk, .css-1q9oajh, .css-qbe2hs, .st-b8qgg5i, .st-dxsdo3, .st-il7t6m, .st-gabwsz, .st-oelb89, .st-ds5zsu, .st-lj9zgt, .st-i5agza, .st-b7zxnu, .st-brgnd1 {
        box-shadow: none;
    }
    .css-1aumxhk, .css-1q9oajh {
        border-radius: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set title and sidebar
st.title('Stock Price Analysis and Prediction')
st.sidebar.title('Options')

# User input for stock ticker
ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL):')

# Date range selection
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('today'))

# Function to load data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Load data
if ticker:
    data = load_data(ticker, start_date, end_date)

    # Display data
    st.subheader('Stock Data')
    st.write(data)
    
    # Display current price
    ticker_info = yf.Ticker(ticker)
    current_price = ticker_info.history(period="1d").iloc[-1]["Close"]
    st.subheader('Current Price')
    st.write(current_price)

    # Additional functionalities
    if st.sidebar.checkbox("Show Volume"):
        st.subheader('Volume')
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color='#636EFA'))
        fig_volume.update_layout(title="Volume", xaxis_title="Date", yaxis_title="Volume", showlegend=False)
        st.plotly_chart(fig_volume)

    if st.sidebar.checkbox("Show Candlestick Chart"):
        st.subheader('Candlestick Chart')
        fig_candlestick = go.Figure(data=[go.Candlestick(x=data.index,
                                                         open=data['Open'],
                                                         high=data['High'],
                                                         low=data['Low'],
                                                         close=data['Close'])])
        fig_candlestick.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_candlestick)

    if st.sidebar.checkbox("Show Price Distribution"):
        st.subheader('Price Distribution')
        fig_distribution = px.histogram(data, x='Close', nbins=50, title='Price Distribution')
        st.plotly_chart(fig_distribution)

    # Plot historical prices
    if st.sidebar.checkbox("Show Line Chart"):
        st.subheader('Historical Prices')
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data['Close'], linewidth=2, color='#1F77B4')
        plt.title('Historical Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        st.pyplot(plt)

    # Linear Regression for Predictions
    if st.sidebar.checkbox("Predict Future Prices with Linear Regression"):
        st.subheader('Linear Regression Predictions')

        # Prepare the data for linear regression
        data['Date'] = data.index
        data['Date_Ordinal'] = pd.to_datetime(data['Date']).apply(lambda date: date.toordinal())
        X = data[['Date_Ordinal']].values
        y = data['Close'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = lr_model.predict(X_train)
        y_pred_test = lr_model.predict(X_test)

        # Display the model performance metrics
        st.write('Train MSE:', mean_squared_error(y_train, y_pred_train))
        st.write('Train R2 Score:', r2_score(y_train, y_pred_train))
        st.write('Test MSE:', mean_squared_error(y_test, y_pred_test))
        st.write('Test R2 Score:', r2_score(y_test, y_pred_test))

        # Predict future prices
        future_days = st.sidebar.slider("Number of Future Days to Predict", min_value=1, max_value=30, value=7)
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D')
        future_dates_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)
        future_predictions = lr_model.predict(future_dates_ordinal)

        # Display predicted future prices in a table
        future_prices_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Price': future_predictions
        })
        st.subheader('Predicted Future Prices')
        st.write(future_prices_df)

        # Plot future predictions
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Historical Close Price', linewidth=2, color='#1F77B4')
        plt.plot(future_dates, future_predictions, label='Predicted Close Price', linestyle='--', color='#FF7F0E')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.title("Historical vs Predicted Close Price")
        plt.legend()
        st.pyplot(plt)
