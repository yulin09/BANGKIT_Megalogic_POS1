import json
from flask import Flask, jsonify

app = Flask(__name__)

# Load the JSON file
with open(r'Customer_Segmentation/Customer_segmentation.json', 'r') as json_file:
    data = json.load(json_file)

# Extract functions from JSON data
fetch_customer_data = data['functions']['fetch_customer_data']['code']
preprocess_data = data['functions']['preprocess_data']['code']
apply_kprototypes = data['functions']['apply_kprototypes']['code']
visualize_clusters = data['functions']['visualize_clusters']['code']
main = data['functions']['main']['code']

# Define Flask routes
@app.route('/')
def index():
    return "Welcome to the Customer Segmentation!"

@app.route('/fetch_data')
def fetch_data():
    # Call the function to fetch customer data
    customer_data = eval(fetch_customer_data)()
    return jsonify(customer_data)

@app.route('/preprocess_data')
def preprocess():
    # Call the function to preprocess data
    preprocessed_data = eval(preprocess_data)(customer_data)
    return jsonify(preprocessed_data)

@app.route('/apply_kprototypes')
def apply_kprototypes_route():
    # Call the function to apply K-Prototypes algorithm
    clusters, kproto = eval(apply_kprototypes)(preprocessed_data)
    return jsonify({"clusters": clusters, "kproto": kproto})

@app.route('/visualize_clusters')
def visualize():
    # Call the function to visualize clusters
    pairplot_filename, barplot_filename, scatterplot_filenames = eval(visualize_clusters)(preprocessed_data, clusters, kproto)
    return jsonify({"pairplot_filename": pairplot_filename, "barplot_filename": barplot_filename, "scatterplot_filenames": scatterplot_filenames})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
