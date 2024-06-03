from flask import Flask, jsonify, render_template
import pandas as pd
import mysql.connector
import pickle
import json
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# Use Agg backend for Matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)

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

def market_basket_analysis(orders_df):
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    transactions = orders_df.groupby(['customer_id'])['product_name'].apply(list).reset_index()
    transaction_list = transactions['product_name'].tolist()
    transaction_encoder = TransactionEncoder()
    transaction_encoder_ary = transaction_encoder.fit(transaction_list).transform(transaction_list)
    transaction_df = pd.DataFrame(transaction_encoder_ary, columns=transaction_encoder.columns_)
    frequent_itemsets = apriori(transaction_df, min_support=0.001, use_colnames=True)
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.001)
        rules = rules[rules['confidence'] < 1]
        rules = rules.drop_duplicates(subset=['lift'], keep='first')
        rules['count'] = rules.apply(lambda row: sum(transaction_df[list(row['antecedents'])].all(axis=1) & transaction_df[list(row['consequents'])].all(axis=1)), axis=1)
        rules['basket_pair'] = rules['antecedents'].apply(lambda x: ', '.join([f"{item}" for item in list(x)])) + " -> " + rules['consequents'].apply(lambda x: ', '.join(list(x)))
        return rules
    else:
        return pd.DataFrame()

def plot_top_rules(rules):
    if not rules.empty:
        rules_display = rules[['basket_pair', 'support', 'confidence', 'lift', 'count']]
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
        with open('top_rules.pkl', 'wb') as f:
            pickle.dump(unique_top_rules_df, f)
        unique_top_rules_df.to_json('top_rules.json', orient='records')
        
        # Plotting code should be in the main thread
        plt.figure(figsize=(14, 8))
        colors = plt.cm.tab20.colors
        unique_top_rules_df.plot(kind='barh', x='basket_pair', y='lift', color=colors, legend=False)
        plt.title('Top 10 Unique Association Rules by Lift')
        plt.xlabel('Lift')
        plt.ylabel('Basket Pair')
        max_lift = unique_top_rules_df['lift'].max()
        plt.xlim(0, max_lift * 1.1)
        plt.tight_layout()
        os.makedirs('static', exist_ok=True)
        plot_filename = os.path.join('static', 'top_rules.png')
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close()

        insights = []
        count = 0
        insights_text = ""
        for _, rule in unique_top_rules_df.iterrows():
            antecedents = list(rule['antecedents'])
            consequents = list(rule['consequents'])
            for ant in antecedents:
                for cons in consequents:
                    insight = f"Customers who usually buy <b>{ant}</b> are more likely to buy <b>{cons}</b>"
                    if insight not in insights:
                        insights.append(insight)
                        count += 1
                        insights_text += f"{count}. {insight}<br>"
                        if count == 5:
                            break
                if count == 5:
                    break
            if count == 5:
                break
        return plot_filename, insights_text
    else:
        return None, ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_mba', methods=['GET'])
def run_mba():
    orders_df = fetch_orders_data()
    rules = market_basket_analysis(orders_df)
    plot_filename, insights_text = plot_top_rules(rules)
    if plot_filename and insights_text:
        return jsonify({'status': 'success', 'plot_url': plot_filename.replace("\\", "/"), 'insights': insights_text})
    else:
        return jsonify({'status': 'error', 'message': 'No association rules found'})

if __name__ == '__main__':
    app.run(debug=True)
