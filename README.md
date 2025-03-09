# College Placement Chatbot

A web-based chatbot that answers questions about college students' placement data using natural language processing.

## Overview

This project implements a question-answering system that allows users to query information about college students, their placement status, GPA, salary, and other details. The system uses a pre-trained NLP model to interpret questions and provide relevant answers based on the available data.

## Features

- Web interface for submitting questions
- Natural language processing to understand user queries
- Data-driven responses about student placement information
- Responsive design that works on both desktop and mobile devices

## Tech Stack

- **Backend**: Flask (Python)
- **NLP**: Hugging Face Transformers (DistilBERT)
- **Data Processing**: Pandas
- **Frontend**: HTML, CSS, Jinja2 Templates

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/college-placement-chatbot.git
   cd college-placement-chatbot
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare your data:
   - Place your `job_placement.csv` file in the `data/` directory
   - Ensure it has the required columns: name, age, degree, stream, college_name, placement_status, gpa, salary, years_of_experience

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Type your question in the text area and submit to get a response

## Example Questions

- "Which students have been placed?"
- "What is the average GPA of placed students?"
- "Who has the highest salary?"
- "How many students are pursuing a Computer Science degree?"

## Data Format

The `job_placement.csv` file should contain the following columns:
- `name`: Student name
- `age`: Student age
- `degree`: Degree program (B.Tech, M.Tech, etc.)
- `stream`: Field of study (Computer Science, Electrical, etc.)
- `college_name`: Name of the college
- `placement_status`: Whether placed or not (Yes/No)
- `gpa`: Grade Point Average
- `salary`: Placement salary (if applicable)
- `years_of_experience`: Prior work experience in years

## Project Structure

```
college-placement-chatbot/
├── app.py                  # Main Flask application
├── templates/              # HTML templates
│   └── index.html          # Main web interface
├── static/                 # Static assets
│   └── css/
│       └── style.css       # Styles for the web interface
├── data/                   # Data files
│   └── job_placement.csv   # Student and placement data
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore file
└── LICENSE                 # Project license
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- PCCE IT Department

## Acknowledgments

- Thanks to Hugging Face for providing the transformer models
- Special thanks to all contributors and testers
