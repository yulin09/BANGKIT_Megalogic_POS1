import os
from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import mysql.connector
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

def fetch_data():
    try:
        conn = mysql.connector.connect(
            host='103.219.251.246',
            user='braincor_ps01',
            password='Bangkit12345.',
            database='braincor_ps01'
        )
        cursor = conn.cursor()

        orders_query = "SELECT ID, order_date, ship_date, customer_id, product_id, product_name FROM orders"
        cursor.execute(orders_query)
        orders_data = cursor.fetchall()
        orders_df = pd.DataFrame(orders_data, columns=['ID', 'order_date', 'ship_date', 'customer_id', 'product_id', 'product_name'])

        products_query = "SELECT * FROM products"
        cursor.execute(products_query)
        products_data = cursor.fetchall()

        products_column_query = "SHOW COLUMNS FROM products"
        cursor.execute(products_column_query)
        products_columns = cursor.fetchall()
        products_columns = [column[0] for column in products_columns]
        products_df = pd.DataFrame(products_data, columns=products_columns)

        cursor.close()
        conn.close()

        return orders_df, products_df

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None, None

def preprocess_data(orders_df, products_df):
    merged_df = orders_df.merge(products_df[['ID', 'unit_price']], left_on='product_id', right_on='ID', how='left')
    merged_df = merged_df.drop(columns=['ID_y']).rename(columns={'ID_x': 'ID'})
    merged_df['order_date'] = pd.to_datetime(merged_df['order_date'], errors='coerce')
    merged_df = merged_df.dropna(subset=['order_date'])
    return merged_df

def aggregate_sales_by_date(merged_df):
    sales_data = merged_df.groupby('order_date').size().reset_index(name='sales')
    return sales_data.set_index('order_date').asfreq('D', fill_value=0)

def train_lstm_model(train_data, n_past, n_future):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(n_past, len(scaled_data) - n_future + 1):
        X_train.append(scaled_data[i - n_past:i, 0])
        y_train.append(scaled_data[i + n_future - 1:i + n_future, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=n_future))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

    return model, scaler

def evaluate_model(test_data, y_pred):
    mse = mean_squared_error(test_data, y_pred)
    mae = mean_absolute_error(test_data, y_pred)
    r2 = r2_score(test_data, y_pred)
    mape = np.mean(np.abs((test_data - y_pred) / test_data)) * 100
    return {
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'R-squared': r2,
        'Mean Absolute Percentage Error': mape
    }

def future_sales_predictions(model, scaler, train_data, n_past, n_future):
    scaled_data = scaler.transform(train_data.values.reshape(-1, 1))
    X_input = scaled_data[-n_past:]
    X_input = X_input.reshape((1, X_input.shape[0], 1))

    predictions = []
    for _ in range(n_future):
        pred = model.predict(X_input)
        predictions.append(pred[0][0])
        X_input = np.append(X_input[:, 1:, :], [[[pred[0][0]]]], axis=1)

    future_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=1), periods=n_future, freq='D')
    future_data = pd.Series(future_predictions.flatten(), index=future_dates)
    future_data_monthly = future_data.resample('M').sum()
    next_month_prediction = future_data_monthly.iloc[0].item()
    next_trimester_prediction = future_data_monthly.sum().item()
    return future_data_monthly, next_month_prediction, next_trimester_prediction

def generate_suggestions(future_data_monthly, next_month_prediction, next_trimester_prediction):
    suggestions = []
    for month, sales in future_data_monthly.items():
        if sales > future_data_monthly.shift(1).get(month, 0):
            suggestions.append(
                f"{month}: The sales trend is positive for the upcoming month. Ensure sufficient inventory and consider promotional campaigns to boost sales."
            )
        else:
            suggestions.append(
                f"{month}: The sales forecast indicates a potential dip. Increase marketing efforts and explore new advertising channels to attract more buyers."
            )
    suggestions.append(
        f"Next month: Predicted sales are {next_month_prediction}. Adjust your strategies to maximize sales."
    )
    suggestions.append(
        f"Next trimester: Predicted sales are {next_trimester_prediction}. Plan your marketing and inventory strategies accordingly."
    )
    return suggestions

def plot_sales(test_data, y_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data['sales'], label='Actual Sales')
    plt.plot(test_data.index, y_pred, label='Predicted Sales')
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

def monthly_sales_latest_year(merged_df):
    merged_df['year'] = merged_df['order_date'].dt.year
    latest_year = merged_df['year'].max()
    merged_df['month'] = merged_df['order_date'].dt.strftime('%b')
    
    month_order = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    merged_df['month'] = pd.Categorical(merged_df['month'], categories=month_order, ordered=True)
    
    monthly_sales = merged_df[merged_df['year'] == latest_year].groupby('month').size().reset_index(name='sales')
    monthly_sales = monthly_sales.sort_values('month')
    return monthly_sales

def top_3_products_last_month_sales(merged_df):
    merged_df['month'] = merged_df['order_date'].dt.to_period('M')
    last_month = merged_df['month'].max()
    top_products_last_month = merged_df[merged_df['month'] == last_month].groupby('product_name').size().reset_index(name='sales')
    top_products_last_month = top_products_last_month.sort_values('sales', ascending=False)
    top_3_products_last_month = top_products_last_month.head(3)
    return top_3_products_last_month

@app.route('/')
def index():
    return '''<h1>Welcome to the Sales Prediction API</h1>
              <p>Use the endpoint <code>/predict-sales</code> to get sales predictions and suggestions.</p>'''

@app.route('/monthly-sales-latest-year', methods=['GET'])
def monthly_sales_latest_year_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    monthly_sales = monthly_sales_latest_year(merged_df)
    return jsonify(monthly_sales.to_dict(orient='list'))

@app.route('/top-3-products-last-month', methods=['GET'])
def top_3_products_last_month_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    top_3_products_last_month = top_3_products_last_month_sales(merged_df)
    return jsonify(top_3_products_last_month.to_dict(orient='list'))

@app.route('/future-sales-next-month', methods=['GET'])
def future_sales_next_month_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    time_series = aggregate_sales_by_date(merged_df)

    end_date = time_series.index[-1]
    split_date = end_date - pd.DateOffset(months=3)

    train_data = time_series.loc[time_series.index < split_date]
    test_data = time_series.loc[time_series.index >= split_date]

    n_past = 30
    n_future = len(test_data)
    model, scaler = train_lstm_model(train_data, n_past, n_future)

    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model, scaler, train_data, n_past, 90)

    result = {
        'title': 'Next Month Prediction',
        'value': next_month_prediction
    }

    return jsonify(result)

@app.route('/future-sales-trimester', methods=['GET'])
def future_sales_trimester_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    time_series = aggregate_sales_by_date(merged_df)

    end_date = time_series.index[-1]
    split_date = end_date - pd.DateOffset(months=3)

    train_data = time_series.loc[time_series.index < split_date]
    test_data = time_series.loc[time_series.index >= split_date]

    n_past = 30
    n_future = len(test_data)
    model, scaler = train_lstm_model(train_data, n_past, n_future)

    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model, scaler, train_data, n_past, 90)

    result = {
        'title': 'Next Trimester Prediction',
        'value': next_trimester_prediction
    }

    return jsonify(result)

@app.route('/generate-suggestions', methods=['GET'])
def generate_suggestions_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    time_series = aggregate_sales_by_date(merged_df)

    end_date = time_series.index[-1]
    split_date = end_date - pd.DateOffset(months=3)

    train_data = time_series.loc[time_series.index < split_date]
    test_data = time_series.loc[time_series.index >= split_date]

    n_past = 30
    n_future = len(test_data)
    model, scaler = train_lstm_model(train_data, n_past, n_future)

    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model, scaler, train_data, n_past, 90)
    suggestions = generate_suggestions(future_data_monthly, next_month_prediction, next_trimester_prediction)

    return jsonify(suggestions)

@app.route('/sales-of-every-product-last-month', methods=['GET'])
def sales_of_every_product_last_month_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    sales_last_month_data = sales_of_every_product_last_month(merged_df)
    series = sales_last_month_data['units_sold'].tolist()
    labels = sales_last_month_data['product_name'].tolist()
    result = {
        'series': series,
        'labels': labels
    }
    return jsonify(result)

def sales_of_every_product_last_month(merged_df):
    merged_df['month'] = merged_df['order_date'].dt.to_period('M')
    last_month = merged_df['month'].max()
    sales_last_month = merged_df[merged_df['month'] == last_month].groupby('product_name').size().reset_index(name='units_sold')
    return sales_last_month

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == "__main__":
    app.run(debug=True)
