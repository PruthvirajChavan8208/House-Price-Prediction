
  
import streamlit as st
import pandas as pd 
import numpy as np
import pickle as pkl


st.title("House Price Prediction App")
st.write("Welcome to pruthviraj chavan's Project")

model=pkl.load(open("HPP.pkl","rb+"))
ds=pd.read_csv("cleaned_data.csv")

post_by=sorted(ds["POSTED_BY"].unique())
bhk_rk=sorted(ds["BHK_OR_RK"].unique())
city=sorted(ds["CITY"].unique())

POSTED_BY=st.sidebar.selectbox("Select who posted this House:",post_by,key=post_by)
BHK_OR_RK=st.sidebar.selectbox("Select Your House Type:",bhk_rk,key=bhk_rk)
BHK_NO=st.sidebar.text_input("Enter the No of Bhk/RK:","2")
SQUARE_FT=st.sidebar.text_input("Enter the Square_Ft:")
CITY=st.sidebar.selectbox("Enter the city:",city,key=city)

if st.button("Predict"):
    columns=["POSTED_BY","BHK_NO.","BHK_OR_RK","SQUARE_FT","CITY"]
    # data=[[POSTED_BY,BHK_NO,BHK_OR_RK,SQUARE_FT,CITY]]
    # st.write("You Provide Following Info:")
    # st.write("1.POSTED_BY:",POSTED_BY)
    # st.write("2.BHK_OR_RK:",BHK_OR_RK)
    # st.write("3.NO_Of BHK/Rk:",BHK_NO)
    # st.write("4.SQUARE_FT:",SQUARE_FT)
    # st.write("5.CITY:",CITY)
    myinput=pd.DataFrame([[POSTED_BY,int(BHK_NO),BHK_OR_RK,int(SQUARE_FT),CITY]],columns=columns)
    result=model.predict(myinput)
    st.success(f'The predicted price is:{np.round(result[0],2)}')
    Price=st.write("Predicted Price of the house is:",result[0,0])
    st.write("(Above Price are in lakh )",Price)
    
