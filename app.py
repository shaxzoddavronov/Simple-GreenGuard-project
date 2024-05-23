import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os 
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_image_comparison import image_comparison
from streamlit_image_select import image_select
from streamlit_option_menu import option_menu
import joblib
import base64
import json
import time
st.set_page_config(
    layout="wide",)
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

def get_prediction(model_file,img,classes):
    model=load_model(model_file)
    img=preprocess_image(Image.open(img))
    img=np.array(img.resize((224,224)))
    img_arr=tf.keras.preprocessing.image.img_to_array(img)
    img_arr=tf.expand_dims(img_arr,0)
    predicted=model.predict(img_arr)
    #predicted=np.round(sorted(predicted,reverse=True)[:len(classes)],2)
    result=classes[np.argmax(predicted[0])]
    accuracy=np.round(100*np.max(predicted[0]),2)
    if len(classes)<4:
        return np.round(predicted[0][:-1],2),result
    
    return np.round(predicted[0],2),result

def get_result(model_path,classes):
    disease_prob={}
    prediction,result=get_prediction(model_path,image,classes)
    #for pred,disease in zip(prediction,classes):
    disease_prob['disease']=classes
    disease_prob['probability']=prediction
    #dframe=pd.DataFrame(disease_prob)
    return disease_prob,result

def stream_writer(assistant_response):
    message_placeholder=st.empty()
    full_response=""
                
    for chunk in assistant_response.split():
        full_response += chunk+" "
        time.sleep(0.07)
        message_placeholder.markdown(full_response+"▌")
    message_placeholder.markdown(full_response) 
    
scaler=joblib.load('min_max_scaler.jbl')
encoder=joblib.load('label_encoder.jbl')
recommender_model=joblib.load('lgb_model.h5')

crop_json='eco.json'
with open(crop_json,'r',encoding='utf-8') as file:
     crop_text_gen=json.load(file)

selected=option_menu(
    menu_title=None,
    options=['Home','Crop Disease Detection','Crop Recommend System'],
    icons=['house','flower1','flower2'],
    orientation='horizontal')

crop_images=['apple.jpeg','bell_pepper.jpeg','cherry.jpg','citrus.jpg','corn.jpg','grapes.jpg',
             'peach.jpg','potato.jpg','strawberry.png','coming_soon.png','coming_soon.png','coming_soon.png',
             'coming_soon.png','coming_soon.png','coming_soon.png','coming_soon.png']    
all_crop_classes={
    'apple':['Black rot','Healthy','Scab','Cedar rust'],
    'bell_pepper':['Bacterial spot','Healthy'],
    'cherry':['Healthy', 'Mildew'],
    'citrus':['Black spot', 'Healthy', 'Canker', 'Greening'],
    'corn':['Common rust','Gray leaf spot','Healthy','Northern Leaf Blight'],
    'grapes':['Black Measles','Black rot','Healthy','Isariopsis Leaf Spot'],
    'peach':['Bacterial spot','Healthy'],
    'potato':['Early blight', 'Healthy', 'Late blight'],
    'strawberry':['Healthy', 'Leaf scorch']
    }

if selected=='Crop Disease Detection':
    st.title('Crop Disease Detection from Leaf')

    crops=[os.path.join('tinified',image) for image in crop_images]
    img=image_select('Select plant',images=crops)
    choosen_plant=img.split("/")[1].split('.')[0]
             
    if choosen_plant and choosen_plant!='coming_soon':
        model_path=os.path.join('disease_models',"{}_model.h5".format(choosen_plant))

    #st.write(model_path)
    #placeholder = st.empty()
        image=st.file_uploader('Upload plant leaf image',type=["jpg", "jpeg", "png"])
    #placeholder.empty()
        classes=all_crop_classes[choosen_plant]
        if image:
            col1,col2=st.columns(2)
            with col1:
                st.image(image,width=350,use_column_width=False,)
                st.markdown('<style>img {border-radius: 10px;}</style>',unsafe_allow_html=True)
                clicked=st.button('Get Result')
            with col2:
                if clicked:
                    prob_disease,result=get_result(model_path,classes=classes)
                    st.bar_chart(prob_disease,x='disease')
                    st.success(result)
            if clicked:

                text_to_gen=crop_text_gen[choosen_plant][result]
                if result!='Healthy':
                     description=text_to_gen['description']
                     treatment=text_to_gen['treatment']
                     st.subheader('Description')
                     stream_writer(description)
                     st.write('')
                     st.subheader('Treatment')
                     stream_writer(treatment)
                else:
                    description=text_to_gen['description']
                    st.subheader('Description')
                    stream_writer(description)


                

 
#video_path='D:\Chatbot Web App\Weed_YOLO_detection\Weed_detection\WeedCrop\sample_video.mp4'

#from main import get_video_detected_result
#import tempfile


if selected=='Crop Recommend System':
    st.title('Crop Recommend System') 
    col1,col2,col3=st.columns(3)
    with col1:
        nitrogen=st.number_input(label='Input Nitrogen',min_value=0,max_value=140)
        phosphorous=st.number_input(label='Input Phosphorous',min_value=0,max_value=145)
        potassium=st.number_input(label='Input  Potassium',min_value=5,max_value=205)
        temperature=st.number_input(label='Input Temperature',min_value=0.0,max_value=44.0,step=0.001)
        humidity=st.number_input(label='Input humidity',min_value=10.0,max_value=100.0,step=0.001)
        ph=st.number_input(label='Input Ph',min_value=0.0,max_value=10.0,step=0.001)
        rainfall=st.number_input(label='Input Rainfall',min_value=20.0,max_value=300.0,step=0.001)
        input_X=[[nitrogen,phosphorous,potassium,temperature,humidity,ph,rainfall]]
        scaled_X=scaler.transform(input_X)
        get_result=st.button('Get Result')
        if get_result:
            predicted=recommender_model.predict(scaled_X)
            st.success(encoder.inverse_transform(predicted)[0])
    with col3:
        st.image('crop_recommend_image.png') 
 

if selected=='Home':
     #col1,col2=st.columns(2)
     st.write('')
     st.write('')
     st.write('')
     st.write('')
     st.write('')
     col1,col2=st.columns(2)
     with col1:
        title="""<h1 class="mb-2 lg:mb-4 mt-2 text-6xl text-center lg:text-left leading-10 font-bold md:text-3xl xl:text-5xl 2xl:text-6xl text-white">GreenGuard: Detecting Crop Diseases and Crop Recommend for Healthy Harvests with AI.</h1>"""
        st.markdown(title, unsafe_allow_html=True)
     some_text="""<p style='font-family: Calibri; font-size: 18px; font-weight: 300; letter-spacing: wide; margin-bottom: 6px; color: white;'> Website serves as a comprehensive tool for disease detection in plant leaves.</p>"""
     #st.markdown(some_text, unsafe_allow_html=True)
     
     with col2:
          st.image('plant_app_image.png',width=600)
     st.header('About',divider='rainbow')
     st.title('')

     col1,col2=st.columns(2,gap='large')
     with col1:

          st.write('''
                   #### Disease Detection
                   Through sophisticated algorithms, users will receive detailed reports indicating
                   the percentage of leaf area affected by each identified disease. This functionality will
                    empower farmers and agricultural professionals with timely and accurate information''')
     with col2:
          st.write('''
                   #### Crop Recommend System
                   Our Crop Recommendation System uses advanced machine learning models to analyze soil and climate data,
                   providing farmers with optimized crop suggestions tailored to their specific conditions. Maximize yields
                   and sustainability with our cutting-edge agricultural technology.''')
     st.title('')
     st.title('')
     st.write('''
                * Note! This  is just demo web app. In the future, additional work will be done to develop the Project''')
