"""
@Author: Mamunur Rahman
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
   return render_template('home.html')

@app.route('/input_wart')
def input_wart():
   return render_template('input_wart.html')

@app.route('/input_breast_cancer')
def input_breast_cancer():
   return render_template('input_breast_cancer.html')


@app.route('/ongoing_projects')
def ongoing_projects():
   return render_template('ongoing_projects.html')



@app.route('/result_wart', methods = ['POST'])
def result_wart():
    user_input = request.form
    time = float(user_input['time'])
    age = float(user_input['age'])
    type_ = float(user_input['type'])
    area = float(user_input['area'])
    
        
    ## Cryotherapy
    new_data_cryo = np.array([time, age, type_, area]).reshape(1, -1)
    # load the saved scaler and classifier models
    scaler_cryo = pickle.load(open('scaler_crytotherapy.pkl','rb'))
    model_cryo = pickle.load(open('model_crytotherapy.pkl','rb'))
    # scale the new data
    new_data_cryo = scaler_cryo.transform(new_data_cryo)
    # estimate the probability of treatment success
    probability_cryo = model_cryo.predict_proba(new_data_cryo)[:,1] # take the second column of the probability matrix
    # get the probability in % and round it up to one decimal points
    probability_cryo = round(probability_cryo[0]*100, 1)
    

    ## Immunotherapy
    new_data_immuno = np.array([time, age, type_]).reshape(1, -1)
    # load the saved scaler and classifier models
    scaler_immuno = pickle.load(open('scaler_immunotherapy.pkl','rb'))
    model_immuno = pickle.load(open('model_immunotherapy.pkl','rb'))
    # scale the new data
    new_data_immuno = scaler_immuno.transform(new_data_immuno)
    # estimate the probability of treatment success
    probability_immuno = model_immuno.predict_proba(new_data_immuno)[:,1] # take the second column of the probability matrix
    # get the probability in % and round it up to one decimal points
    probability_immuno = round(probability_immuno[0]*100, 1)    
    
    
    ## output text
    if probability_cryo <= 50 and probability_immuno <= 50:
        output_text = f"Neither Cryotherapy nor Immunotherapy treatment method will work \
                        for this patient. The probabilities of treatment success are only \
                        {probability_cryo}% for Cryotherapy and only {probability_immuno}% \
                        for Immunotherapy. Other treatment methods are recommended."
    
    elif probability_cryo > probability_immuno:
        output_text = f"Cryotherapy is preferred to Immunotherapy for this patient. \
                        The probabilities of treatment success are {probability_cryo}% for \
                        Cryotherapy and {probability_immuno}% for Immunotherapy."
                        
    elif probability_cryo < probability_immuno:
        output_text = f"Immunotherapy is preferred to Cryotherapy for this patient. \
                        The probabilities of treatment success are {probability_immuno}% for \
                        Immunotherapy and {probability_cryo}% for Cryotherapy."
    else:
        output_text = f"Both treatment methods are equally likely to work on this patient. \
                        The probability of treatment success is approximately {probability_cryo}%"
        
    return render_template("result_wart.html", user_input = user_input, result = output_text)


@app.route('/result_breast_cancer', methods = ['POST'])
def result_breast_cancer():
    user_input = request.form
    age = float(user_input['age'])
    bmi = float(user_input['bmi'])
    glucose = float(user_input['glucose'])
    insulin = float(user_input['insulin'])
    resistin = float(user_input['resistin'])
    mcp_1 = float(user_input['mcp_1'])
    
    new_data = np.array([age, bmi, glucose, insulin, resistin, mcp_1]).reshape(1, -1)
    # load the saved scaler and classifier models
    scaler = pickle.load(open('scaler_breast_cancer.pkl','rb'))
    model = pickle.load(open('model_breast_cancer.pkl','rb'))
    # scale the new data
    new_data = scaler.transform(new_data)
    # estimate the probability of treatment success
    probability = model.predict_proba(new_data)[:,1] # take the second column of the probability matrix
    # get the probability in % and round it up to one decimal points
    probability = round(probability[0]*100, 1)
    
    
    ## output text
    if probability < 40:
        output_text = "Good news!!! The probability of having breast cancer for this person is very low."
    elif probability >= 40 and probability < 60:
        output_text = f"The probability of having breast cancer is moderate, approximately {probability}%.\
                        You might want to discuss with your doctor for further investigation."
    else:
        output_text = f"The probability of having breast cancer is high, approximately {probability}%.\
                        Please discuss with your doctor for further investigation."
            
    return render_template("result_breast_cancer.html", user_input = user_input, result = output_text)
    



if __name__ == '__main__':
   app.run(debug = True)


