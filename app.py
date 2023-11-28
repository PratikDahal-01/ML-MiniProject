from flask import Flask, render_template, request
from Models import DecisionTreeModel, RandomForestModel, NaiveBayesModel
from Models import l1

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/Disease_Prediction')
def disease_prediction():
    symptoms = l1
    return render_template('disease.html', symptoms=symptoms)
@app.route('/predict', methods=['POST'])
def predict():
    symptoms = [request.form['Symptom1'], request.form['Symptom2'], request.form['Symptom3'], request.form['Symptom4'], request.form['Symptom5']]

    # Machine learning model functions here
    dt_prediction = DecisionTreeModel(symptoms)
    rf_prediction = RandomForestModel(symptoms)
    nb_prediction = NaiveBayesModel(symptoms)

    return render_template('disease_predicted.html', symptoms=l1, dt_prediction=dt_prediction, rf_prediction=rf_prediction, nb_prediction=nb_prediction)

if __name__ == '__main__':
    app.run(debug=True)