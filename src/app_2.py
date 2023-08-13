
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import os




# Useful functions
#def load_ml_components(fp):
#    "Load the ml components to re-use in app"
 #   with open(fp, "rb") as f:
 #       object = pickle.load(f)
 #       return object


def Churn_Predict(TotalCharges,
    MonthlyCharges,
    tenure,
    StreamingTV,
    PaperlessBilling,
    DeviceProtection,
    TechSupport,
    InternetService,
    OnlineSecurity,
    StreamingMovies,
    PaymentMethod,
    Dependents,
    Partner,
    tenure_group,
    OnlineBackup,
    gender,
    SeniorCitizen,
    MultipleLines,
    Contract,
    PhoneService
    ):
    """
    """
    df = pd.DataFrame({'TotalCharges': [TotalCharges], 'MonthlyCharges': [MonthlyCharges], 'tenure': [tenure], 'StreamingTV': [StreamingTV], 'PaperlessBilling': [PaperlessBilling], 'DeviceProtection': [DeviceProtection], 'TechSupport': [TechSupport], 'InternetService': [InternetService], 'OnlineSecurity': [OnlineSecurity] , 'StreamingMovies': [StreamingMovies], 'PaymentMethod': [PaymentMethod], 'Dependents': [Dependents], 'Partner': [Partner], 'tenure_group': [tenure_group], 'OnlineBackup': [OnlineBackup], 'gender': [gender], 'SeniorCitizen': [SeniorCitizen], 'MultipleLines': [MultipleLines], 'Contract': [Contract], 'PhoneService': [PhoneService]}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               )
    print(f"Inputs as Dataframe : {df.to_markdown()}")
    
    
 # Setup
 # Variables and constants
 
 
 
 # Execution    
        

# interface
inputs = [gr.Dropdown(elem_id=i) for i in range(20)] + [gr.Number(elem_id=i)
                                                        for i in range(5)]

demo = gr.Interface(
    Churn_Predict,
    inputs,
    "number",
    examples=[],   
    )

demo.launch()


    
## st.title ('Sales Prediction in Favorita Stores')

## st.subheader ('Using Time Series Forecasting') 

# Add some text
## st.write("""Welcome to Favorita Corporation, an Ecuadorian company which engages in the development and investment of commercial, industrial, and real estate ventures. Its subsidiaries operate in six countries within the region, namely Panama, Paraguay, Peru, Colombia, Costa Rica, and Chile. The company is committed to delivering top-notch products, services, and experiences in a manner that is both efficient and environmentally responsible, aiming to enhance the overall quality of life.


         
 #        Please ENTER the relevant data and CLICK Predict.
         
 #        """ )

     
# with st.form(key="information", clear_on_submit=True):
    
    # Inputs
 #   store_nbr = st.slider("Enter store nbr",0,54)
 #   products = st.selectbox(" Family", ['OTHERS', 'CLEANING', 'FOODS', 'STATIONERY', 'GROCERY', 'HARDWARE',
 #       'HOME', 'CLOTHING'])
 #   onpromotion = st.number_input("Discount Amt On Promotion_Expressed in Percentage %",step=1)
 #   state = st.selectbox("State", ['Choose a State', 'Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
 #       'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
 #       'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
 #       'El Oro', 'Esmeraldas', 'Manabi'])

 #   store_type = st.selectbox("Store Type",['Choose Store Type', 'D', 'C', 'B', 'E', 'A'])
 #   Cluster = st.number_input("Cluster",step=1)
 #   dcoilwtico = st.number_input("DCOILWTICO",step=1)
 #   year = st.number_input("Year to Predict",step=1)

 #   month = st.slider("Month",1,12)
 #   day = st.slider("Day",1,31)
 #   dayofweek = st.number_input("Day of Week,0=Sunday and 6=Satruday",step=1)


    # prediction executed
 #   if st.form_submit_button("Predict"):
 #       try:
            # Dataframe creation
 #           df = pd.DataFrame(
 #               {
 #               "store_nbr":[store_nbr],  "products":[products], "onpromotion":[onpromotion], "state": [state], "store_type": [store_type],
 #               "Cluster": [Cluster], "dcoilwtico":[dcoilwtico], "year":[year], "month":[month], "day":[day], "dayofweek":[dayofweek],

 #               }   
 #                   )
 #           print(f"[Info] Input data as dataframe:\n{df.to_markdown()}") 
   
    
 #           st.text(f"The Predicted Sales: '{''}'.")
 #       except:
 #           st.error(f"Something went wrong during prediction.") python -m pip install -qr requirements.txt