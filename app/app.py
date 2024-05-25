import pickle
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Prediction", page_icon="img/dnc.webp")
st.title("Telco Churn prediction")

st.image('img/customer_churn.jpeg')

st.markdown("""
Churn prediction is a crucial aspect of any business that aims to retain its customers.
In the context of a machine learning prediction app, churn prediction refers to the process of identifying customers who are most likely to 
stop using a company's products or services. The app uses historical data and machine learning algorithms to analyze patterns and 
behaviors of past customers who have churned, and then applies this knowledge to identify the customers
who are most likely to leave in the future. By accurately predicting customer churn, a company can take proactive steps 
to retain its valuable customers and minimize the negative impact of churn on its bottom line.

By using this machine learning app for churn prediction, companies can save time and resources, 
and make more informed business decisions.
""")

# -- Model -- #
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)

data = st.file_uploader('Upload your file')
if data:
    df_input = pd.read_csv(data)
    df_output = df_input.assign(
        churn=model.predict(df_input),
        churn_probability=model.predict_proba(df_input)[:,1]
        )

    st.markdown('Churn prediction:')
    st.write(df_output)
    st.download_button(
        label='Download CSV', data=df_output.to_csv(index=False).encode('utf-8'),
        mime='text/csv', file_name='churn_prediction.csv'
        )
