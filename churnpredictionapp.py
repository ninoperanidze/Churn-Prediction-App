import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go 
import plotly

# Load the trained model
with open("model.pkl", 'rb') as file:
    model = pickle.load(file)

    # Load the dataset
    df = pd.read_csv("BankChurners.csv")

def run():

    # Add a title and some text to the app
    st.title("Churn Prediction App")

    # Create a select box for the tabs
    tab1, tab2 = st.tabs(["Prediction", "Histogram"])

    with tab1:
        # Create two columns in the main area
        col1, col2 = st.columns(2)
        
        # Define a mapping from display names to internal values
        education_mapping = {
            'Less than High School': 'Uneducated',
            'High School': 'High School',
            'College': 'College',
            'Graduate': 'Graduate',
            'Post-Graduate': 'Post-Graduate',
            'Doctorate': 'Doctorate',
            'Unknown': 'Unknown'
        }

        with col1:
            Gender = st.radio('Gender', ['Male', 'Female'])
            Customer_Age = st.slider('Customer Age', 18, 100, 50)
            Education_Level = st.radio('Education Level', list(education_mapping.keys()))
            Marital_Status = st.radio('Marital Status', ['Married', 'Single', 'Divorced', 'Unknown'])
            Income_Category = st.radio('Income Category', ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'])
            Dependent_count = st.slider('Dependent count', 0, 5, 2)
            Card_Category = st.radio('Card Category', ['Blue', 'Silver', 'Gold', 'Platinum'])

        with col2:
            Months_on_book = st.slider('Months on book', 13, 56, 36)
            Total_Relationship_Count = st.slider('Total Relationship Count', 1, 6, 3)
            Months_Inactive_12_mon = st.slider('Months Inactive LTM', 0, 6, 3)
            Contacts_Count_12_mon = st.slider('Contacts Count LTM', 0, 6, 2)
            Credit_Limit = st.number_input('Credit Limit', min_value=1438.3, max_value=34516.0, value=4500.0)
            Total_Revolving_Bal = st.number_input('Total Revolving Balance', min_value=0, max_value=2517, value=1271)
            Avg_Open_To_Buy = st.number_input('Avg Open To Buy', min_value=3.0, max_value=34516.0, value=7461.5)
            Total_Trans_Amt = st.number_input('Total Transaction Value', min_value=510, max_value=18484, value=4404)
            Total_Amt_Chng_Q4_Q1 = st.number_input('Change in Transaction Value Q4/Q1', min_value=0.0, max_value=3.397, value=0.772)
            Total_Trans_Ct = st.number_input('Total Number of Transactions', min_value=10, max_value=139, value=67)
            Total_Ct_Chng_Q4_Q1 = st.number_input('Change in Number of Transactions Q4/Q1', min_value=0.0, max_value=3.714, value=0.702)
            Avg_Utilization_Ratio = st.number_input('Avg Utilization Ratio', min_value=0.0, max_value=0.999, value=0.296)

        # Add a divider
        st.write('---')

        # Create a button that, when clicked, predicts whether a client will churn
        if st.button("Predict"):
            data = {'Customer_Age': Customer_Age, 'Gender': Gender, 'Dependent_count': Dependent_count, 'Education_Level': Education_Level, 'Marital_Status': Marital_Status, 'Income_Category': Income_Category, 'Card_Category': Card_Category, 'Months_on_book': Months_on_book, 'Total_Relationship_Count': Total_Relationship_Count, 'Months_Inactive_12_mon': Months_Inactive_12_mon, 'Contacts_Count_12_mon': Contacts_Count_12_mon, 'Credit_Limit': Credit_Limit, 'Total_Revolving_Bal': Total_Revolving_Bal, 'Avg_Open_To_Buy': Avg_Open_To_Buy, 'Total_Amt_Chng_Q4_Q1': Total_Amt_Chng_Q4_Q1, 'Total_Trans_Amt': Total_Trans_Amt, 'Total_Trans_Ct': Total_Trans_Ct, 'Total_Ct_Chng_Q4_Q1': Total_Ct_Chng_Q4_Q1, 'Avg_Utilization_Ratio': Avg_Utilization_Ratio}
            data_df = pd.DataFrame(data, index=[0])
            prediction = model.predict(data_df)
            if prediction[0] == 'Attrited Customer':
                st.write("Customer likely to leave")
            elif prediction[0] == 'Existing Customer':
                st.write("Customer not likely to leave")

    with tab2:
        # Define a dictionary that maps the original feature names to the new names
        feature_names = {
            'Customer_Age': 'Customer Age',
            'Dependent_count': 'Dependent Count',
            'Months_on_book': 'Months on Book',
            'Total_Relationship_Count': 'Total Relationship Count',
            'Months_Inactive_12_mon': 'Months Inactive LTM',
            'Contacts_Count_12_mon': 'Contacts Count LTM',
            'Credit_Limit': 'Credit Limit',
            'Total_Revolving_Bal': 'Total Revolving Balance',
            'Avg_Open_To_Buy': 'Average Open to Buy',
            'Total_Amt_Chng_Q4_Q1': 'Change in Number of Transactions Q4/Q1',
            'Total_Trans_Amt': 'Total Transaction Value',
            'Total_Trans_Ct': 'Total Number of Transactions',
            'Total_Ct_Chng_Q4_Q1': 'Change in Transaction Value Q4/Q1',
            'Avg_Utilization_Ratio': 'Average Utilization Ratio'
        }
        
        # Ask the user to select a feature
        selected_feature = st.selectbox('Select a feature', list(feature_names.values()))
        
        # Get the original feature name
        feature = list(feature_names.keys())[list(feature_names.values()).index(selected_feature)]
        
        # Filter the data based on the selected feature
        churned = df[df['Attrition_Flag'] == 'Attrited Customer'][feature]
        retained = df[df['Attrition_Flag'] == 'Existing Customer'][feature]

        # Ask the user to select a distribution
        churned_checkbox = st.checkbox('Show Churned')
        retained_checkbox = st.checkbox('Show Retained')

        # Create a histogram
        fig = go.Figure()

        # Add the selected distributions to the histogram
        if retained_checkbox:
            fig.add_trace(go.Histogram(x=retained, name='Retained', marker_color='grey'))
        if churned_checkbox:
            fig.add_trace(go.Histogram(x=churned, name='Churned', marker_color='red'))


        # Overlay both histograms
        fig.update_layout(barmode='overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)

        # Display the histogram
        with st.spinner(text='In progress...'):
            st.plotly_chart(fig)

if __name__ == '__main__':
    run()