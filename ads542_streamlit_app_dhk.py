import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

# Custom transformer for label encoding categorical columns
class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # Create 'previous_contact'
        X_new = X.copy()
        X_new['previous_contact'] = (X_new['pdays'] != 999).astype(int)
        X_new.loc[X_new["previous_contact"] == 0, "pdays"] = -1
        
        # Create 'unemployed'
        X_new["unemployed"] = X_new["job"].isin(["student", "retired", "unemployed"]).astype(int)
        
        return X_new

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.encoders = None
    
    def fit(self, X, y=None):
        self.encoders = {
            col: LabelEncoder().fit(X[col].astype(str))
            for col in self.categorical_columns
        }
        return self
    
    def transform(self, X, y=None):
        X_new = X.copy()
        for col, le in self.encoders.items():
            # Handle unknown categories by converting them to string 'unknown'
            X_new[col] = X_new[col].map(lambda s: 'unknown' if s not in le.classes_ else s).astype(str)
            le.classes_ = np.append(le.classes_, 'unknown')
            X_new[col] = le.transform(X_new[col])
        return X_new


# loading the saved model
loaded_model = pickle.load(open('bank_marketing_prediction.pkl', 'rb'))


def bank_marketing_prediction(input_data):
    # Correct the feature names to match the training data
    column_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 
                    'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                    'euribor3m', 'nr.employed', 'risky']  # Use dot notation instead of underscores

    input_df = pd.DataFrame([input_data], columns=column_names)
    prediction = loaded_model.predict(input_df)
    return prediction[0]


def main():
    
    # Giving a title
    st.title('Bank Marketing Prediction')
    st.header('developed by Duygu Halim KIRLI')
    st.subheader('12 May 2025')
    
    # Sidebar title
    st.sidebar.title("Brief Summary")
    st.sidebar.write("This model is trained by a random forest model. Model contains stage of encoding, spliting, resampling and scaling in a pipeline and  turning hyperparameters.")
    
    # Getting input from the user
    st.subheader('Age')
    age = st.slider('How old are you?', 1, 110, 25)
    st.write("Your age is ", age)


    st.subheader('Select Job')
    options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                               'retired', 'self-employed', 'services', 'student', 'technician', 
                               'unemployed', 'unknown']
    job = st.selectbox("Job", options, label_visibility='hidden')
    st.write("Your job is ", job)

    st.subheader('Marital Status')    
    options = ['divorced', 'married', 'single', 'unknown']
    marital = st.selectbox("Marital Status", options, label_visibility='hidden')
    st.write("Your marital status is  ", marital)
    
    st.subheader('Education')
    options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                                 'illiterate', 'professional.course', 'university.degree', 'unknown']
    education = st.selectbox("Education Level", options, label_visibility='hidden')
    st.write("Your education level is  ", education)
    
    st.subheader('Credit In Default')
    options = ['no', 'yes', 'unknown']
    default = st.selectbox('Credit in Default', options, label_visibility='hidden')
    st.write("Your credit in default is  ", default)
    
    st.subheader('Housing Loan')
    options = ['no', 'yes', 'unknown']
    housing = st.selectbox('Housing Loan', options, label_visibility='hidden')
    st.write("Your housing loan is  ", housing)
    
    st.subheader('Personal Loan')
    options = ['no', 'yes', 'unknown']
    loan = st.selectbox('Personal Loan', options, label_visibility='hidden')
    st.write("Your personal loan is  ", loan)
    
    st.subheader('Contact Preference')
    options = ['cellular', 'telephone']
    contact = st.selectbox('Contact Preference', options, label_visibility='hidden')
    st.write("Your Contact Preference is  ", contact)
    
    st.subheader('Last Contact Month')
    options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month = st.selectbox('Last Contact Month', options, label_visibility='hidden')
    st.write("Your Last Contact Month is  ", month)
    
    st.subheader('Last Contact Day of Week')
    options = ['mon', 'tue', 'wed', 'thu', 'fri']
    day_of_week = st.selectbox('Last Contact Day of Week', options, label_visibility='hidden')
    st.write("Your Last Contact Day of Week is", day_of_week)
    
    st.subheader('Outcome of the Previous Marketing Campaign')
    options = ['failure', 'nonexistent', 'success']
    poutcome = st.selectbox('Outcome of the Previous Marketing Campaign', options, label_visibility='hidden')
    st.write("Your Outcome of the Previous Marketing Campaign is", poutcome)
    
    st.subheader('Number of Contacts Performed During this Campaign')
    campaign = st.number_input('Number of Contacts Performed During this Campaign')
    st.write("Your Number of Contacts Performed During this Campaign is", campaign)
    
    st.subheader('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
    pdays = st.number_input('Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign')
    st.write("Your Number of Days that Passed by After the Client was Last Contacted from a Previous Campaign is", pdays)
    
    st.subheader('Number of Contacts Performed Before this Campaign')
    previous = st.number_input('Number of Contacts Performed Before this Campaign')
    st.write("Your Number of Contacts Performed Before this Campaign is", previous)
    
    st.subheader('Employment Variation Rate')
    emp_var_rate = st.number_input('Employment Variation Rate')
    st.write("Your Employment Variation Rate is", emp_var_rate)
    
    st.subheader('Consumer Price Index')
    cons_price_idx = st.number_input('Consumer Price Index')
    st.write("Your Consumer Price Index is", cons_price_idx)
    
    st.subheader('Consumer Confidence Index')
    cons_conf_idx = st.number_input('Consumer Confidence Index')
    st.write("Your Consumer Confidence Index is", cons_conf_idx)
    
    st.subheader('Euribor 3 Month Rate')
    euribor3m = st.number_input('Euribor 3 Month Rate')
    st.write("Your Euribor 3 Month Rate is", euribor3m)
    
    st.subheader('Number of Employees')
    nr_employed = st.number_input('Number of Employees')
    st.write("Your Number of Employees is", nr_employed)
    
    st.subheader('Risky')
    choice = st.selectbox("Risky", ["Yes", "No"], label_visibility='hidden')
    risky = 1 if choice == "Yes" else 0
    st.write("Selected value:", risky)
   

    # Code for prediction
    prediction = ''
    
    # Getting the input data from the user
    if st.button('Bank Marketing Prediction'):
        prediction = bank_marketing_prediction([age, job, marital, education, default, housing, loan,
                                                contact, month, day_of_week, campaign, pdays, previous, 
                                                poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, 
                                                euribor3m, nr_employed, risky])
        st.success(prediction)
        st.write('prediction result: ', prediction)
        if prediction == 1:
            st.write("You subscribed a term deposit")
        else:
            st.write("Your subscribtion failed.")
        
    
if __name__ == '__main__':
    main()