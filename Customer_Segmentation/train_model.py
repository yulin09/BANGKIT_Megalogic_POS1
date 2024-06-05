import os
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import mysql.connector

def fetch_customer_data():
    conn = mysql.connector.connect(
        host='103.219.251.246',
        user='braincor_ps01',
        password='Bangkit12345.',
        database='braincor_ps01'
    )

    query = """
        SELECT ID, customer_name, gender, age, job, segment, total_spend, previous_purchase FROM customers
    """

    cursor = conn.cursor()
    cursor.execute(query)
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    columns = ['ID', 'customer_name', 'gender', 'age', 'job', 'segment', 'total_spend', 'previous_purchase']
    customer_df = pd.DataFrame(data, columns=columns)
    
    return customer_df

def preprocess_data(customer_df):
    df = customer_df.drop(['ID', 'customer_name'], axis=1)
    le = LabelEncoder()
    for column in ['gender', 'job', 'segment']:
        df[column] = le.fit_transform(df[column])
    return df

# Fetch and preprocess the data
customer_df = fetch_customer_data()
df = preprocess_data(customer_df)

# Apply K-Prototypes
n_clusters = 5
kproto = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=42)
clusters = kproto.fit_predict(df, categorical=[0, 2, 3])

# Ensure the 'model' directory exists
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)

# Save the model
with open(os.path.join(model_dir, 'kproto_model.pkl'), 'wb') as file:
    pickle.dump(kproto, file)

# Save model configuration
model_config = {
    'n_clusters': n_clusters,
    'init': 'Huang',
    'random_state': 42
}

with open(os.path.join(model_dir, 'model_config.json'), 'w') as file:
    json.dump(model_config, file)
