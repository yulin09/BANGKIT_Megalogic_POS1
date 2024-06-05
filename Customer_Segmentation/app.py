from flask import Flask, render_template, send_file
import os
from segmentation import fetch_customer_data, preprocess_data, apply_kprototypes, create_summary_table, visualize_clusters, create_membership_column_if_not_exists, update_database_with_clusters

app = Flask(__name__)

@app.route('/')
def index():
    # Create membership column if it doesn't exist
    create_membership_column_if_not_exists()

    # Fetch customer data
    customer_df = fetch_customer_data()
    
    # Preprocess the data
    df, ids = preprocess_data(customer_df)
    
    # Apply K-Prototypes
    clusters, kproto = apply_kprototypes(df, n_clusters=5)
    
    # Update database with new cluster labels
    update_database_with_clusters(ids, clusters)
    
    # Visualize the clusters and get plot paths
    df, plot_paths = visualize_clusters(df, clusters, kproto)
    
    # Create summary table
    summary_table = create_summary_table(df)
    
    # Convert the summary table to HTML
    summary_table_html = summary_table.to_html(classes='table table-striped', index=False)
    
    # Generate URLs for the plots
    plot_urls = [os.path.basename(plot_path) for plot_path in plot_paths]

    return render_template('index.html', summary_table=summary_table_html, plot_urls=plot_urls)

@app.route('/plot/<filename>')
def plot(filename):
    plot_path = os.path.join(os.path.dirname(__file__), 'static', 'plots', filename)
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
