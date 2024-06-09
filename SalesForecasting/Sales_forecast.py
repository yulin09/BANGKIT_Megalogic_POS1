import os
from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import mysql.connector
from statsmodels.tsa.arima.model import ARIMA

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

def train_arima_model(train_data):
    model_arima = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model_arima.fit()
    return model_fit

def evaluate_model(test_data, y_pred_arima):
    mse_arima = mean_squared_error(test_data, y_pred_arima)
    mae_arima = mean_absolute_error(test_data, y_pred_arima)
    r2_arima = r2_score(test_data, y_pred_arima)
    mape_arima = np.mean(np.abs((test_data['sales'] - y_pred_arima) / test_data['sales'])) * 100
    return {
        'Mean Squared Error': mse_arima,
        'Mean Absolute Error': mae_arima,
        'R-squared': r2_arima,
        'Mean Absolute Percentage Error': mape_arima
    }

def monthly_sales_latest_year(merged_df):
    merged_df['year'] = merged_df['order_date'].dt.year
    latest_year = merged_df['year'].max()
    merged_df['month'] = merged_df['order_date'].dt.month_name()
    
    # Define the categorical type with the correct order
    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June', 
        'July', 'August', 'September', 'October', 'November', 'December'
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

def sales_last_month(merged_df):
    merged_df['month'] = merged_df['order_date'].dt.to_period('M')
    last_month = merged_df['month'].max()
    sales_last_month = merged_df[merged_df['month'] == last_month].groupby('product_name').size().reset_index(name='units_sold')
    return sales_last_month

def future_sales_predictions(model_fit, test_data):
    future_dates = pd.date_range(start=test_data.index[-1] + pd.Timedelta(days=1), periods=90, freq='D')
    future_predictions = model_fit.get_forecast(steps=90)
    future_data = future_predictions.predicted_mean
    future_data.index = future_dates
    future_data_monthly = future_data.resample('M').sum()
    future_data_monthly.index = future_data_monthly.index.strftime('%Y-%m')
    next_month_prediction = future_data_monthly.iloc[0]
    next_trimester_prediction = future_data_monthly.sum()
    return future_data_monthly, next_month_prediction, next_trimester_prediction

def generate_suggestions(future_data_monthly, next_month_prediction, next_trimester_prediction):
    suggestions = []
    for month, sales in future_data_monthly.items():
        if sales > future_data_monthly.shift(1).get(month, 0):
            suggestions.append(
                f"{month}: The sales trend is positive for the upcoming month. To capitalize on this momentum, ensure you have sufficient inventory ready to meet the demand. Consider running promotional campaigns or offering discounts to further boost sales."
            )
        else:
            suggestions.append(
                f"{month}: The sales forecast indicates a potential dip. It might be beneficial to increase your marketing efforts and enhance your product visibility. Explore new advertising channels, engage with your customers on social media, and possibly offer limited-time deals to attract more buyers."
            )
    suggestions.append(
        f"Next month: Predicted sales are {next_month_prediction}. Adjust your strategies accordingly to maximize sales."
    )
    suggestions.append(
        f"Next trimester: Predicted sales are {next_trimester_prediction}. Plan your marketing and inventory strategies to handle the expected demand."
    )
    return suggestions

def plot_sales(test_data, y_pred_arima):
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data['sales'], label='Actual Sales')
    plt.plot(test_data.index, y_pred_arima, label='Predicted Sales')
    plt.legend()
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

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

@app.route('/sales-last-month', methods=['GET']) #Key error 'month'
def sales_last_month_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    sales_last_month_data = sales_last_month(merged_df)
    return jsonify(sales_last_month_data.to_dict(orient='list'))

@app.route('/future-sales-predictions', methods=['GET']) 
def future_sales_predictions_route():
    orders_df, products_df = fetch_data()
    if orders_df is None or products_df is None:
        return jsonify({"error": "Unable to fetch data from the database"}), 500
    
    merged_df = preprocess_data(orders_df, products_df)
    time_series = aggregate_sales_by_date(merged_df)

    end_date = time_series.index[-1]
    split_date = end_date - pd.DateOffset(months=3)

    train_data = time_series.loc[time_series.index < split_date]
    test_data = time_series.loc[time_series.index >= split_date]

    model_fit = train_arima_model(train_data)
    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model_fit, test_data)

    result = {
        'future_sales': future_data_monthly.to_dict(),
        'next_month_prediction': next_month_prediction,
        'next_trimester_prediction': next_trimester_prediction
    }

    return jsonify(result)

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

    model_fit = train_arima_model(train_data)
    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model_fit, test_data)

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

    model_fit = train_arima_model(train_data)
    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model_fit, test_data)

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

    model_fit = train_arima_model(train_data)
    future_data_monthly, next_month_prediction, next_trimester_prediction = future_sales_predictions(model_fit, test_data)
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
