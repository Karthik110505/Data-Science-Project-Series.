# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__,template_folder='templates')

# Load the Naive Bayes model
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the form
    text = request.form['text']
    # Use the model to predict sentiment
    prediction = model.predict([text])
    if prediction:
       result = "Positive"
    else:
        result = "Negative" 
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
