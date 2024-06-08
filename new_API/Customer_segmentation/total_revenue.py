from flask import Flask, jsonify
import mysql.connector
from mysql.connector import Error

app = Flask(__name__)

# Database configuration
DB_CONFIG = {
    'host': '103.219.251.246',
    'user': 'braincor_ps01',
    'password': 'Bangkit12345.',
    'database': 'braincor_ps01'
}

# Function to fetch and calculate total revenue
def get_total_revenue():
    try:
        print("Attempting to connect to the database...")
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("Successfully connected to the database.")
            cursor = conn.cursor(dictionary=True)
            
            # Query to fetch order details and product unit prices
            query = """
                SELECT o.ID, o.order_date, o.ship_date, o.customer_id, o.product_id, o.product_name, p.unit_price
                FROM orders o
                JOIN products p ON o.product_id = p.ID
            """
            print("Executing query: ", query)
            cursor.execute(query)
            orders = cursor.fetchall()
            print("Query executed. Fetched orders: ", orders)
            
            if not orders:
                print("No orders found.")
                return None

            total_revenue = sum(order['unit_price'] for order in orders)
            print("Total revenue calculated: ", total_revenue)
            
            cursor.close()
            conn.close()
            return total_revenue
        else:
            print("Failed to connect to the database.")
            return None
    except Error as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"General error: {e}")
        return None

@app.route('/total_revenue', methods=['GET'])
def total_revenue():
    revenue = get_total_revenue()
    if revenue is not None:
        return jsonify({"total_revenue": revenue}), 200
    else:
        return jsonify({"error": "Unable to fetch total revenue"}), 500

if __name__ == "__main__":
    app.run(debug=True)
