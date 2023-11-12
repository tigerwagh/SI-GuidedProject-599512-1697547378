from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model1 = pickle.load(open('car_.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def start():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    rd = int(request.form['rd'])
    as_val = int(request.form['as'])
    s = request.form['s']

    # Add age and salary limits
    age_limit = 18  # Minimum age limit
    salary_limit = 14000  # Minimum salary limit

    # Convert state to binary (Male=1, Female=0)
    if s == 'cal':
        s = 1
    elif s == 'flo':
        s = 0

    # Check if input values meet the specified limits
    if rd >= age_limit and as_val >= salary_limit:
        input_data = [[s, rd, as_val]]
        prediction = model1.predict(input_data)
        prediction_bool = bool(prediction[0])
    else:
        prediction_bool = False  # If input values do not meet the limits

    return render_template('index.html', y=prediction_bool)

if __name__ == '__main__':
    app.run(debug=True)