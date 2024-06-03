import mysql.connector
import pandas as pd

def fetch_orders_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="pos_7"
    )
    query = "SELECT ID, order_date, ship_date, customer_id, product_id, product_name FROM orders"
    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    columns = ['ID', 'order_date', 'ship_date', 'customer_id', 'product_id', 'product_name']
    return pd.DataFrame(data, columns=columns)
