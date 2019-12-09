"""
@Author: Mamunur Rahman
"""
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import random
from nltk.stem import WordNetLemmatizer
import base64


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


@app.route('/input_chatbot')
def input_chatbot():
    #  clear the chat history of previous user
    global response
    response= ['Komola: Welcome!! What name should I call you??']
    return render_template('input_chatbot.html', result = response)


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
                            otherwise the prediction wouldn't be reliable"
        
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
                            otherwise the prediction wouldn't be reliable"
        
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
    def recommend_movie(original_user_input, n=7):
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
        idx = list(np.argsort(-similarity_matrix.flatten())[1:15])
        # reload df_name_and_weighted_rank
        df_name_and_weighted_rank = pickle.load(open('df_name_and_weighted_rank_movie_recommender.pkl','rb'))  
        df_top_10_by_type = df_name_and_weighted_rank.loc[idx]
        # sort the list by weighted rank and show the first 5 movies
        result = df_top_10_by_type.sort_values(by=['weighted_rank'], ascending=False)['movie_title'][:n].tolist()
        return result
		
              
    try:
        try:
            recommendations = recommend_movie(original_user_input)
            output_text = f"Since {original_user_input} is your favorite, you might also like: \n "
			
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


## ChatBot
    
# unpickle the necessary data and variables
with open("data_chatbot.pickle", "rb") as f:
    remove_punct_dict, remove_word_dict, scaled_data_chatbot, df_user_input, df_response, vectorizer = pickle.load(f)
# clean and preprocess the target column of the dataframe
def process_sentence(sentence, dictionary):
    #remove punctuations
    sentence = sentence.translate(remove_punct_dict)
    new_sentence = ''
    for word in sentence.split():
        try:
            # replace word if it is in remove_word_dict
            word = dictionary[word]
        except:
            pass
        # lemmatize the word
	try:
		lemmatizer = WordNetLemmatizer()
        	word = lemmatizer.lemmatize(word)
	except:
		pass
        new_sentence = new_sentence + word + ' '
    return new_sentence.strip()  # remove leading and trailing space of the sentence    





response= []

@app.route('/result_chatbot', methods = ['POST'])
def result_chatbot():
      
    def generate_response(original_user_input):

        processed_user_input= []
        processed_user_input.append(process_sentence(original_user_input, remove_word_dict))
        user_input_count_vectorizer = vectorizer.transform(processed_user_input).toarray()
        # calculate cosine similarity for the user input movie
        similarity_matrix = cosine_similarity(scaled_data_chatbot, user_input_count_vectorizer)
        
        # find the maximum value of the similarity matrix
        max_value = max(similarity_matrix.flatten())
        if max_value < 0.2:
            return ["I apolozize!! I don't understand.", "dont_understand"]
        else:
            # get the index of the maximum cosine_similarity
            idx = np.argsort(similarity_matrix.flatten())[-1]
            # get the 'intent' from the df_user_input dataframe
            intent = df_user_input.loc[idx, 'intent']
            # get a random response
            chatbot_response = df_response['chatbot_response'][df_response['intent']==intent].tolist()
            random_response = random.choice(chatbot_response)
            return random_response, intent
    
#    try:
    user_input = request.form['user_input']
    response.append(f"You: {user_input}")
    # by default, place_order is set as False
    place_order = False
    # if customer want to place order, take him to order page
    if generate_response(user_input)[1] == "place_order":
        place_order = True
    
    # At the beginning, in reply to customers name, bot will generate the below response
    if response[-2] == 'Komola: Welcome!! What name should I call you??':
        response.append("Komola: Okay, Thanks!!!")
    else:    
        response.append(f"Komola: {generate_response(user_input)[0]}")

    return render_template("result_chatbot.html", result = response, show_order_page = place_order)
    
#    except: # when the user submit the place order button, this section is executed
#        response.append("Our system is processing your order..................")
#        response.append(f"Komola: Great!!! Your order has been placed. Your order ID is {random.randint(100, 999)}. Is there anything else I can help you with? ")
#        
#        return render_template("result_chatbot.html", result = response, show_order_page = False)
        


if __name__ == '__main__':
   app.run(debug = True)




