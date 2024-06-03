from flask import Flask, render_template, send_from_directory
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    # Define filenames based on your pattern
    timestamp = "20240602000401"  # Example timestamp
    scatter_timestamp = "20240602000009"  # Example timestamp for scatter plots

    pairplot_filename = f'pairplot_{timestamp}.png'
    barplot_filename = f'barplot_{timestamp}.png'
    scatterplot_age_total_spend_filename = f'scatterplot_age_total_spend_{scatter_timestamp}.png'
    scatterplot_age_previous_purchase_filename = f'scatterplot_age_previous_purchase_{scatter_timestamp}.png'
    scatterplot_total_spend_previous_purchase_filename = f'scatterplot_total_spend_previous_purchase_{scatter_timestamp}.png'

    return render_template('index.html',
                           pairplot_filename=pairplot_filename,
                           barplot_filename=barplot_filename,
                           scatterplot_age_total_spend_filename=scatterplot_age_total_spend_filename,
                           scatterplot_age_previous_purchase_filename=scatterplot_age_previous_purchase_filename,
                           scatterplot_total_spend_previous_purchase_filename=scatterplot_total_spend_previous_purchase_filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

@app.route('/results')
def results():
    return send_from_directory(app.root_path, 'cluster_results.json')

if __name__ == '__main__':
    app.run(debug=True)


