import streamlit as st
import numpy as np
import pickle

model=pickle.load(open('C:/projects/ml_projects/end_to_end_project_with_streamlit/model.pkl','rb'))

standard_scaler=pickle.load(open('ss.pkl','rb'))

    
fuel_type_values={'petrol':2,'diesel':1,'cng':0}
seller_type_values={'dealer':0,'individual':1}
transmission_type_values={'manual':1,'automatic':0}
year=st.text_input('enter car model year')
present_price=st.text_input('enter present price in lakhs :')
kms_driven=st.text_input("enter that car was how many kms driven :")
fuel_type=fuel_type_values[st.selectbox('select fuel type',['petrol','diesel','cng'])]
seller_type=seller_type_values[st.selectbox('select seller type',['dealer','individual'])]
transmission=transmission_type_values[st.selectbox('select transmission type',['manual','automatic'])]
if st.button('predict car price'):
    features=np.array([[year,present_price,kms_driven,fuel_type,seller_type,transmission,0]])
    features=standard_scaler.transform(features)
    prediction_value=model.predict(features)
    st.write('predicted car price is ',prediction_value)
    
