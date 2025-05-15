import pandas as pd
import os
from flask import Flask, render_template, request, session
from flask_caching import Cache
import torch
from sentence_transformers import SentenceTransformer, util
import re

# Set up relative file paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "job_placement.csv")

# Flask app setup with caching and session
app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Configure app
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load data
college_data = pd.read_csv(data_path)

# Define model options - just two: one accurate, one fast
models = {
    "accurate": {
        "name": "all-mpnet-base-v2", 
        "description": "Accurate (Slower)"
    },
    "fast": {
        "name": "all-MiniLM-L6-v2", 
        "description": "Fast (Less Accurate)"
    }
}

# Dictionary to store model instances
model_instances = {}

def get_model(model_name):
    """Load model if not already loaded"""
    model_info = models.get(model_name, models["fast"])
    model_key = model_info["name"]
    
    if model_key not in model_instances:
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_instances[model_key] = SentenceTransformer(model_key, device=device)
    return model_instances[model_key]

def analyze_question(question):
    """Determine if it's a statistical/aggregation question"""
    # Common patterns for statistical questions
    stat_patterns = [
        r'highest|maximum|max|most|greatest|largest|best|top',
        r'lowest|minimum|min|least|smallest|worst|bottom',
        r'average|mean|median',
        r'how many|count|number of',
    ]
    
    for pattern in stat_patterns:
        if re.search(pattern, question.lower()):
            return True
    return False

def direct_data_query(question):
    """Handle statistical queries directly with pandas"""
    question = question.lower()
    
    # Extract relevant columns based on the question
    target_col = None
    
    # Map question terms to columns
    if any(term in question for term in ['salary', 'paid', 'earning', 'income']):
        target_col = 'salary'
    elif any(term in question for term in ['gpa', 'grade', 'score']):
        target_col = 'gpa'
    elif any(term in question for term in ['age', 'old', 'young']):
        target_col = 'age'
    elif any(term in question for term in ['experience', 'exp', 'years']):
        target_col = 'years_of_experience'
        
    if not target_col:
        return None
        
    # Handle different query types
    if any(term in question for term in ['highest', 'maximum', 'max', 'most', 'greatest', 'largest', 'best', 'top']):
        result = college_data.loc[college_data[target_col].idxmax()]
        return f"The highest {target_col} is {result[target_col]}, belonging to {result['name']}, who studied {result['degree']} in {result['stream']} at {result['college_name']}."
        
    elif any(term in question for term in ['lowest', 'minimum', 'min', 'least', 'smallest', 'worst', 'bottom']):
        result = college_data.loc[college_data[target_col].idxmin()]
        return f"The lowest {target_col} is {result[target_col]}, belonging to {result['name']}, who studied {result['degree']} in {result['stream']} at {result['college_name']}."
        
    elif any(term in question for term in ['average', 'mean']):
        avg_value = college_data[target_col].mean()
        return f"The average {target_col} is {avg_value:.2f}."
        
    elif any(term in question for term in ['how many', 'count', 'number']):
        if 'placement' in question or 'placed' in question:
            placed_count = college_data[college_data['placement_status'] == 'Placed'].shape[0]
            total = college_data.shape[0]
            return f"{placed_count} out of {total} students were placed ({placed_count/total*100:.1f}%)."
        else:
            return f"There are {college_data.shape[0]} students in the dataset."
    
    return None

def semantic_search_answer(question, model_name="fast"):
    """Get answer based on semantic search"""
    model = get_model(model_name)
    
    # Create context passages from DataFrame
    passages = []
    for _, row in college_data.iterrows():
        passage = f"{row['name']} (Age: {row['age']}, Degree: {row['degree']} - {row['stream']}, College: {row['college_name']}). "
        passage += f"Placement: {row['placement_status']}, GPA: {row['gpa']}, Salary: {row['salary']}, Experience: {row['years_of_experience']} years."
        passages.append(passage)
    
    # Encode the question and passages
    question_embedding = model.encode(question, convert_to_tensor=True)
    passage_embeddings = model.encode(passages, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_scores = util.pytorch_cos_sim(question_embedding, passage_embeddings)[0]
    
    # Get top 3 most similar passages
    top_results = torch.topk(cos_scores, k=min(3, len(passages)))
    
    # Construct answer from top results
    answer = ""
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        if i > 0:
            answer += "\n\n"
        answer += passages[idx]
        
    # If no good match found
    if not answer or max(top_results[0]) < 0.3:
        return "I couldn't find specific information about that in the college data."
        
    return answer

def get_college_info(question, model_name="fast"):
    """Main function to answer questions about college data"""
    # First check if it's a statistical/aggregation question
    if analyze_question(question):
        direct_answer = direct_data_query(question)
        if direct_answer:
            return direct_answer
    
    # Fall back to semantic search for other questions
    return semantic_search_answer(question, model_name)

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route for handling GET and POST requests"""
    # Initialize session chat history if not exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    selected_model = "fast"  # Default to fast model
    
    if request.method == "POST":
        user_input = request.form["user_input"]
        selected_model = request.form.get("model_choice", "fast")
        
        # Get response
        response = get_college_info(user_input, selected_model)
        
        # Add to chat history
        session['chat_history'].append({
            'user': user_input,
            'bot': response,
            'model': selected_model
        })
        
        # Keep history limited to last 10 conversations
        if len(session['chat_history']) > 10:
            session['chat_history'] = session['chat_history'][-10:]
        
        # Save session
        session.modified = True
        
    return render_template(
        "index.html",
        chat_history=session['chat_history'],
        models=models,
        selected_model=selected_model
    )

@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear chat history"""
    session['chat_history'] = []
    return "", 204  # No content response

if __name__ == "__main__":
    # Initialize the faster model for better initial performance
    get_model("fast")
    # Run the Flask app
    app.run(debug=True)
