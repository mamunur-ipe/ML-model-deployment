"""
@Author: Mamunur Rahman
"""
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity


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

@app.route('/input_movie_recommender')
def input_movie_recommender():
   return render_template('input_movie_recommender.html')


@app.route('/more_about_apps')
def more_about_apps():
   return render_template('more_about_apps.html')

@app.route('/about_me')
def about_me():
   return render_template('about_me.html')



@app.route('/result_wart', methods = ['POST'])
def result_wart():
    try:
        user_input = request.form
        time = float(user_input['time'])
        age = float(user_input['age'])
        type_ = int(user_input['type'])
        area = float(user_input['area'])
        
        ## check the input values whether is it within the recommended range
        if age<15 or age>100 or time<0 or time>24 or area<1 or area>1000:
            output_text = "Please insert values within the recommended range, \
                            otherwise the prediction would not be reliable"
        
            return render_template("input_error_wart.html", user_input = user_input, result = output_text)
        
        else:    
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
    
    ## the user input is not numeric
    except:
        user_input = request.form
        output_text = "Please insert only numeric values"
        
        return render_template("input_error_wart.html", user_input = user_input, result = output_text)
                


@app.route('/result_breast_cancer', methods = ['POST'])
def result_breast_cancer():
    
    try:
        user_input = request.form
        age = float(user_input['age'])
        bmi = float(user_input['bmi'])
        glucose = float(user_input['glucose'])
        insulin = float(user_input['insulin'])
        resistin = float(user_input['resistin'])
        mcp_1 = float(user_input['mcp_1'])
        
        ## check the input values whether it is within the recommended range
        if age<15 or age>100 or bmi<15 or bmi>40 or glucose<60 or glucose>250 or insulin<2 or insulin>60 or resistin<3 or resistin>80 or mcp_1<40 or mcp_1>1700:
            output_text = "Please insert values within the recommended range, \
                            otherwise the prediction will not be reliable"
        
            return render_template("input_error_breast_cancer.html", user_input = user_input, result = output_text)
        
        else:
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
            if probability < 10:
                output_text = "Great!!! The probability of having breast cancer for this \
                                person is very low, less than 10%."
            elif probability < 30:
                output_text = f"The probability of having breast cancer for this person \
                                is low, approximately {probability}%."
            elif probability < 40:
                output_text = f"The probability of having breast cancer for this person \
                                is approximately {probability}%."
            elif probability < 60:
                output_text = f"The probability of having breast cancer for this person \
                                is approximately {probability}%. You might want to \
                                discuss with your doctor for further investigation."
            elif probability <90:
                output_text = f"The probability of having breast cancer for this person \
                                is approximately {probability}%. Please discuss with \
                                your doctor for further investigation."
            else:
                output_text = "The probability of having breast cancer for this person is \
                                extremely high, more than 90%. Please discuss with your \
                                doctor immediately for further investigation."
                    
            return render_template("result_breast_cancer.html", user_input = user_input, result = output_text)
    
    ## the user input is not numeric
    except:
        user_input = request.form
        output_text = "Please insert only numeric values"
        
        return render_template("input_error_breast_cancer.html", user_input = user_input, result = output_text)
    



@app.route('/result_movie_recommender', methods = ['POST'])
def result_movie_recommender():

    original_user_input = request.form['movie_name']
    
    # define a function which will recommend best 'n' movies for a specific user
    def recommend_movie(original_user_input, n=5):
        user_input = original_user_input.strip()
        user_input = user_input.lower().replace(' ', '').replace(':', '').replace('\'', '').replace('-', '')
        # get index of the movie from the movie_list
        movie_list = pickle.load(open('movie_list.pkl','rb'))
        idx = movie_list.index(user_input)
        # unpickle scaled data
        scaled_data = pickle.load(open('scaled_data_movie_recommender.pkl','rb'))
        # calculate cosine similarity for the user input movie
        similarity_matrix = cosine_similarity(scaled_data, scaled_data[idx].reshape(1, -1))
        # get the index of the top 10 movies similar to user_input
        # the index 0 contains the user input movie. So, we start the index from 1
        idx = list(np.argsort(-similarity_matrix.flatten())[1:10])
        # reload df_name_and_weighted_rank
        df_name_and_weighted_rank = pickle.load(open('df_name_and_weighted_rank_movie_recommender.pkl','rb'))  
        df_top_10_by_type = df_name_and_weighted_rank.loc[idx]
        # sort the list by weighted rank and show the first 5 movies
        result = df_top_10_by_type.sort_values(by=['weighted_rank'], ascending=False)['movie_title'][:n].tolist()
        return result
		
              
    try:
        try:
            recommendations = recommend_movie(original_user_input)
            output_text = f"Since you liked {original_user_input}, our recommendations are: \n "
			
        except:
            movie_list = pickle.load(open('movie_list.pkl','rb'))
            close_match = difflib.get_close_matches(original_user_input, movie_list, n=1)[0]
            user_input = close_match
            idx = movie_list.index(user_input)
            df_name_and_weighted_rank = pickle.load(open('df_name_and_weighted_rank_movie_recommender.pkl','rb'))
            movie_title = df_name_and_weighted_rank.loc[idx, 'movie_title']
            output_text = f"Did you mean- {movie_title.strip()}? \nIf yes, our recommendations are:\n"
            recommendations = recommend_movie(close_match)
    
    except:
        output_text = "Sorry!!! The movie is not in our database. Please try another movie or keyword."
        recommendations = []

     
    return render_template("result_movie_recommender.html", result = output_text, recommended_movies = recommendations)




if __name__ == '__main__':
   app.run()




