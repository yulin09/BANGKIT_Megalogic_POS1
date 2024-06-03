import mysql.connector
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Fetch orders data from MySQL
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
    frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)

    # Generate association rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)

        # Filter out rules with confidence of 1 to avoid infinite conviction
        rules = rules[rules['confidence'] < 1]

        # Remove duplicate rules based on 'lift', keeping the first occurrence
        rules = rules.drop_duplicates(subset=['lift'], keep='first')

        # Add count of occurrences
        rules['count'] = rules.apply(lambda row: sum(transaction_df[list(row['antecedents'])].all(axis=1) & transaction_df[list(row['consequents'])].all(axis=1)), axis=1)

        # Create a column for the basket pair
        rules['basket_pair'] = rules['antecedents'].apply(lambda x: ', '.join([f"{item}" for item in list(x)])) + " -> " + rules['consequents'].apply(lambda x: ', '.join(list(x)))

        return rules
    else:
        return pd.DataFrame()

def plot_top_rules(rules):
    if not rules.empty:
        # Select relevant columns for display
        rules_display = rules[['basket_pair', 'support', 'confidence', 'lift', 'count']]
        print("Association Rules")
        print(rules_display)
        
        # Extract unique pairs for top 10 rules by lift
        seen_pairs = set()
        unique_top_rules = []
        for _, rule in rules.nlargest(20, 'lift').iterrows():
            pair = rule['basket_pair']
            if pair not in seen_pairs:
                unique_top_rules.append(rule)
                seen_pairs.add(pair)
            if len(unique_top_rules) == 10:
                break

        unique_top_rules_df = pd.DataFrame(unique_top_rules)
        
        # Create horizontal bar chart for the unique top 10 rules by lift
        fig, ax = plt.subplots()
        colors = plt.cm.tab20.colors
        unique_top_rules_df.plot(kind='barh', x='basket_pair', y='lift', ax=ax, color=colors, legend=False)
        ax.set_title('Top 10 Unique Association Rules by Lift')
        ax.set_xlabel('Lift')
        ax.set_ylabel('Basket Pair')
        
        # Dynamically set x-axis limit based on maximum lift value
        max_lift = unique_top_rules_df['lift'].max()
        ax.set_xlim(0, max_lift * 1.1)  
        
        ax.set_yticklabels(unique_top_rules_df['basket_pair'], fontsize=10)  
        plt.show()
        
        # Display MBA insights
        print("MBA Insights")
        insights = []
        count = 0
        for _, rule in unique_top_rules_df.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            for ant in antecedents:
                for cons in consequents:
                    insight = f"Customers who usually buy **{ant}** are more likely to buy **{cons}**"
                    if insight not in insights:
                        insights.append(insight)
                        count += 1
                        print(f"{count}. {insight}")
                        if count == 5:
                            break
                if count == 5:
                    break
            if count == 5:
                break
    else:
        print("No association rules found. Try lowering the min_threshold value.")

# Main execution
if __name__ == "__main__":
    orders_df = fetch_orders_data()
    print("Orders Data")
    print(orders_df.head())
    
    rules = market_basket_analysis(orders_df)
    plot_top_rules(rules)
