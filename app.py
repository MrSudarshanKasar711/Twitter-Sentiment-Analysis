
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pickle


model=pickle.load(open(r'D:\Sudarshan Kasar\Capstone\Twitter Sentiment Analysis\Twitter sentiment project\app\analysis_model.pkl','rb'))
vectorize=pickle.load(open(r'D:\Sudarshan Kasar\Capstone\Twitter Sentiment Analysis\Twitter sentiment project\app\vectorizer.pkl','rb'))

st.title('Twitter Sentiment Analysis Using Machine Learning')
#st.write('Enter A Message')

user_input=st.text_area('Enter A Tweet')

if st.button('predict'):
    if user_input=='':
        print('Plese Enter A Message')
    else:
        trasformed_input=vectorize.transform([user_input]).toarray()
        prediction=model.predict(trasformed_input)[0]


        if prediction==1:
            st.success('Positive Tweet')
        else:
            st.error('Negative Tweet')
            

        
    

# %%
