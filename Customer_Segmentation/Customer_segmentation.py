import pandas as pd
from kmodes.kprototypes import KPrototypes
import mysql.connector
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
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

#Preprocess the data
def preprocess_data(customer_df):
    # Drop the 'ID' and 'customer_name' columns for clustering
    df = customer_df.drop(['ID', 'customer_name'], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    for column in ['gender', 'job', 'segment']:
        df[column] = le.fit_transform(df[column])
    
    return df

# Apply K-Prototypes algorithm
def apply_kprototypes(df, n_clusters=5):
    kproto = KPrototypes(n_clusters=n_clusters, init='Huang', random_state=42)
    clusters = kproto.fit_predict(df, categorical=[0, 2, 3])
    
    return clusters, kproto

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

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Ensure the directory exists
    os.makedirs('images', exist_ok=True)
    
    # Pairplot to visualize the clusters
    sns.pairplot(df, hue='Cluster_Label', palette=cluster_colors)
    pairplot_filename = f'images/pairplot_{timestamp}.png'
    plt.savefig(pairplot_filename)
    plt.show()
    plt.close()

    # Bar plots for specified feature distributions
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    sns.barplot(data=df, x='previous_purchase', y='total_spend', hue='Cluster_Label', palette=cluster_colors, ax=axes[0], errorbar=None)
    axes[0].set_title('Total Spend vs Previous Purchase')
    
    sns.barplot(data=df, x='gender', y='age', hue='Cluster_Label', palette=cluster_colors, ax=axes[1], errorbar=None)
    axes[1].set_title('Age vs Gender')

    barplot_filename = f'images/barplot_{timestamp}.png'
    plt.savefig(barplot_filename)
    plt.show()
    plt.close()

    # Scatter plots for numerical features
    scatter_features = [('age', 'total_spend'), ('age', 'previous_purchase'), ('total_spend', 'previous_purchase')]
    scatterplot_filenames = []
    for x_feature, y_feature in scatter_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Cluster_Label', palette=cluster_colors, s=100, alpha=0.6)
        scatterplot_filename = f'images/scatterplot_{x_feature}_{y_feature}_{timestamp}.png'
        plt.savefig(scatterplot_filename)
        plt.show()
        plt.close()
        scatterplot_filenames.append(scatterplot_filename)

    return pairplot_filename, barplot_filename, scatterplot_filenames

def main():
    # Fetch data
    customer_df = fetch_customer_data()
    
    # Preprocess data
    df = preprocess_data(customer_df)
    
    # Apply K-Prototypes
    clusters, kproto = apply_kprototypes(df, n_clusters=5)
     
    # Visualize the clusters
    pairplot_filename, barplot_filename, scatterplot_filenames = visualize_clusters(df, clusters, kproto)
    
    print(f"Visualizations saved as {pairplot_filename}, {barplot_filename}, and {scatterplot_filenames}")

if __name__ == "__main__":
    main()
