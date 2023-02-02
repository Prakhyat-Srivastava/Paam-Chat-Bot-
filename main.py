#importing libraries to take help
import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

#Lemmatizer is function that works for understanding the same meaningful words 
lemmatizer = WordNetLemmatizer()

#reading our data 
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# to convert numarical data to words what we need 
# to do that we def four diffrent functions

# function that cleans up the sentences
# it will delete the same words and ignore punctuation
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words


# funtion of getting bag of words
# it will convert sentences to a bag of words which is numarical values
def bag_of_words(sentence):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


# function for predicting the class based on the sentence essentially
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # it will return the best predicted value of answer for particular quetion
    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# function for getting response 
# it will return the correct answer
def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# basic intro for paam
print("|============= Welcome to College Equiry Chatbot System! =============|")
print("|============================== Feel Free ============================|")
print("|================================== To ===============================|")
print("|=============== Ask your any query about our college ================|")

# to enter and exit from paam
while True:
    message = input("| You: ")
    if message == "bye" or message == "Goodbye":
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)
        print("|===================== The Program End here! =====================|")
        exit()

    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("| Bot:", res)