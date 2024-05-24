import streamlit as st
import mysql.connector
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch orders data from MySQL
@st.cache_data
def fetch_orders_data():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="pos_7"
    )

    query = """
        SELECT ID, order_date, ship_date, customer_id, product_id, product_name FROM orders
    """

    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    columns = ['ID', 'order_date', 'ship_date', 'customer_id', 'product_id', 'product_name']
    orders_df = pd.DataFrame(data, columns=columns)
    
    return orders_df

def market_basket_analysis(orders_df):
    # Clone the DataFrame to avoid modifying the cached object
    orders_df = orders_df.copy()
    
    # Convert order_date to datetime
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    # Aggregate products by customer to form transactions
    transactions = orders_df.groupby(['customer_id'])['product_name'].apply(list).reset_index()

    # Prepare data for MBA
    transaction_list = transactions['product_name'].tolist()

    # Transform the transaction data
    transaction_encoder = TransactionEncoder()
    transaction_encoder_ary = transaction_encoder.fit(transaction_list).transform(transaction_list)
    transaction_df = pd.DataFrame(transaction_encoder_ary, columns=transaction_encoder.columns_)

    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(transaction_df, min_support=0.05, use_colnames=True)

    # Generate association rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

        # Filter out rules with confidence of 1 to avoid infinite conviction
        rules = rules[rules['confidence'] < 1]

        # Convert numeric values to percentages
        rules['support'] = rules['support'] * 100
        rules['confidence'] = rules['confidence'] * 100
        rules['lift'] = rules['lift'] * 100

        # Create a column for the basket pair
        rules['basket_pair'] = rules['antecedents'].apply(lambda x: ', '.join(list(x))) + " -> " + rules['consequents'].apply(lambda x: ', '.join(list(x)))

        # Add count column to show how many pairs contain the item sets
        rules['count'] = rules['antecedents'].apply(lambda x: transaction_df.loc[transaction_df.apply(lambda row: row.astype(str).str.contains(', '.join(list(x))).all(), axis=1)].shape[0])

        print("Association Rules Head:")
        print(rules.head())

        return rules
    else:
        return pd.DataFrame()


# Streamlit app
st.title("Market Basket Analysis")

orders_df = fetch_orders_data()
st.write("Orders Data", orders_df)

if st.button("Run Market Basket Analysis"):
    rules = market_basket_analysis(orders_df)
    
    if not rules.empty:
        # Select relevant columns for display
        rules_display = rules[['basket_pair', 'support', 'confidence', 'lift', 'count']]
        st.write("Association Rules", rules_display)
        
        # Create scatter plot for Lift vs Confidence
        fig, ax = plt.subplots()
        sns.scatterplot(data=rules, x='confidence', y='lift', ax=ax)
        ax.set_title('Lift vs Confidence')
        ax.set_xlabel('Confidence (%)')
        ax.set_ylabel('Lift (%)')
        st.pyplot(fig)
        st.write("Scatter plot generated successfully.")
    else:
        st.write("No association rules found. Try lowering the min_threshold value.")
