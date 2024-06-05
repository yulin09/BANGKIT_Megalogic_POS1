import pandas as pd
from kmodes.kprototypes import KPrototypes
import mysql.connector
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Connect to the database and fetch data
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

# Create membership column if it doesn't exist
def create_membership_column_if_not_exists():
    conn = mysql.connector.connect(
        host='103.219.251.246',
        user='braincor_ps01',
        password='Bangkit12345.',
        database='braincor_ps01'
    )

    cursor = conn.cursor()

    # Check if column exists
    cursor.execute("SHOW COLUMNS FROM customers LIKE 'membership'")
    result = cursor.fetchone()

    if not result:
        cursor.execute("ALTER TABLE customers ADD membership INT")
        conn.commit()

    cursor.close()
    conn.close()

# Preprocess the data
def preprocess_data(customer_df):
    # Drop the 'ID' and 'customer_name' columns for clustering
    df = customer_df.drop(['ID', 'customer_name'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    for column in ['gender', 'job', 'segment']:
        df[column] = le.fit_transform(df[column])
    
    return df, customer_df['ID']

# Apply K-Prototypes algorithm
def apply_kprototypes(df, n_clusters=5):
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=42)
    clusters = kproto.fit_predict(df, categorical=[0, 2, 3])
    
    return clusters, kproto

# Create the summary table
def create_summary_table(df):
    summary_table = df.groupby(['previous_purchase', 'Cluster_Label'])['total_spend'].sum().unstack().fillna(0)
    summary_table.reset_index(inplace=True)
    summary_table.columns.name = None
    summary_table.rename_axis(None, axis=1, inplace=True)
    
    return summary_table

# Visualize the results
def visualize_clusters(df, clusters, kproto):
    df['Cluster'] = clusters
    
    # Assign labels to clusters
    cluster_labels = {2: 'bronze', 3: 'silver', 1: 'gold', 4: 'platinum', 0: 'diamond'}
    df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
    
    # Define cluster colors
    cluster_colors = {
        'bronze': 'brown',
        'silver': 'silver',
        'gold': 'gold',
        'platinum': 'green',
        'diamond': 'blue'
    }

    # Pairplot to visualize the clusters
    sns.pairplot(df, hue='Cluster_Label', palette=cluster_colors)
    plt.show()

    # Bar plot for previous purchase vs total spend
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df, x='previous_purchase', y='total_spend', hue='Cluster_Label', palette=cluster_colors, errorbar=None)
    plt.title('Total Spend vs Previous Purchase')
    plt.show()

    # Scatter plots for numerical features
    scatter_features = [('age', 'total_spend'), ('age', 'previous_purchase'), ('total_spend', 'previous_purchase')]
    for x_feature, y_feature in scatter_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Cluster_Label', palette=cluster_colors, s=100, alpha=0.6)
        plt.show()

    # Pie chart for cluster distribution
    cluster_counts = df['Cluster_Label'].value_counts()
    plt.figure(figsize=(10, 7))
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', colors=[cluster_colors[label] for label in cluster_counts.index], startangle=140)
    plt.title('Customer Distribution by Cluster')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

# Update the database with new cluster labels
def update_database_with_clusters(ids, clusters):
    conn = mysql.connector.connect(
        host='103.219.251.246',
        user='braincor_ps01',
        password='Bangkit12345.',
        database='braincor_ps01'
    )

    cursor = conn.cursor()

    update_query = """
        UPDATE customers
        SET membership = %s
        WHERE ID = %s
    """

    # Convert clusters to Python int type
    data_to_update = [(int(clusters[i]), int(ids[i])) for i in range(len(ids))]

    cursor.executemany(update_query, data_to_update)
    conn.commit()
    cursor.close()
    conn.close()

def main():
    # Create membership column if it doesn't exist
    create_membership_column_if_not_exists()

    # Fetch data
    customer_df = fetch_customer_data()
    
    # Preprocess data
    df, ids = preprocess_data(customer_df)
    
    # Apply K-Prototypes
    clusters, kproto = apply_kprototypes(df, n_clusters=5)
    
    # Update database with new cluster labels
    update_database_with_clusters(ids, clusters)
     
    # Visualize the clusters
    visualize_clusters(df, clusters, kproto)
    
    # Create and display the summary table
    summary_table = create_summary_table(df)
    print(summary_table)
    
    print("Visualizations displayed successfully.")

if __name__ == "__main__":
    main()
