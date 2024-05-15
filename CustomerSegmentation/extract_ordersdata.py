import mysql.connector
import pandas as pd

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="pos_2"
)

# SQL query to extract data from the tables
query = """
    SELECT * FROM products
"""

# Execute the query
cursor = conn.cursor()
cursor.execute(query)

# Fetch the data
data = cursor.fetchall()

# Define column names
columns = ['ID', 'order_date', 'ship_date', 'customer_id', 'product_id']

# Convert data to DataFrame
orders_df = pd.DataFrame(data, columns=columns)

# Close cursor and connection
cursor.close()
conn.close()

# Display DataFrame
print(orders_df)