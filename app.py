import pandas as pd
from transformers import pipeline
import os
from flask import Flask, render_template, request

# Set up relative file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "job_placement.csv")

# Load data
college_data = pd.read_csv(data_path)

# Initialize question-answering model
nlp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

def get_college_info(question, context):
    """Get answer to a question based on the provided context"""
    result = nlp(question=question, context=context)
    return result['answer']

def generate_context():
    """Generate context from the college data for the question-answering model"""
    context = ''
    for _, row in college_data.iterrows():
        context += f"{row['name']} is a {row['age']}-year-old student pursuing a {row['degree']} in {row['stream']} at {row['college_name']}. "
        context += f"Placement status: {row['placement_status']}, GPA: {row['gpa']}, Salary: {row['salary']}, Experience: {row['years_of_experience']} years.\n"
    return context

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for handling GET and POST requests"""
    if request.method == "POST":
        user_input = request.form["user_input"]
        context = generate_context()  
        response = get_college_info(user_input, context) 
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
