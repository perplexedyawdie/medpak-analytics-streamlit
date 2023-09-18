import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from mitosheet.streamlit.v1 import spreadsheet
from statsmodels.tsa.arima_model import ARIMA
import altair as alt
from medpak_analytics import bg_pred
import plotly.express as px
from pangea_sec import scan_excel
st.set_page_config(
    page_title="Dietitian"
)

try:
    if st.session_state["selected_professional"] != "dietitian":
        switch_page(st.session_state["selected_professional"])
    
    st.subheader(f"Welcome to your session on {st.session_state['selected_health_condition']}")
    st.header('Overview')
    st.write('''
    This application is designed to assist healthcare providers in leveraging data to make informed decisions and predictions. Here are the types of analytics you can perform and how they can be beneficial:

    1. **Blood Glucose Level Prediction:**
    - **Benefit:** Enables the prediction of future blood glucose levels based on historical data, helping in better diabetes management.
    - **How to use:** Upload the historical data, review and edit it if necessary, then click on "Predict Blood Glucose Levels" to get the predictions.
    ''')

    st.subheader('Blood Glucose Level Prediction')

    st.subheader('Step 1: Upload Your Data')
    uploaded_file = st.file_uploader("Choose a CSV file", type="xlsx")

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        if scan_excel(uploaded_file):
            st.write("Data uploaded successfully!")
            # Section to edit data in a spreadsheet
            st.subheader('Step 2: Edit Your Data')
            if uploaded_file is not None:
                edited_df = st.data_editor(data, num_rows="dynamic")
                
                with st.container():
                    bg_data_cols = edited_df.columns.tolist()
                    bg_readings = st.selectbox('Please select the column with blood glucose readings: ', bg_data_cols)
                    bg_reading_time = st.selectbox('Please select the column with time of the readings: ', bg_data_cols)
                    bg_lim = st.number_input('Please enter the Blood Glucose threshold.')
                if bg_readings is not None and bg_reading_time is not None and bg_readings != bg_reading_time:
                    st.subheader('Step 3: Predictive Analytics')
                steps = st.slider("Select the predection range (each represents 5 minutes)", min_value=1, max_value=10)
                pred_24h_glucose = st.button("Predict Glucose Level")
                if pred_24h_glucose:
                    try:
                        if uploaded_file is not None and bg_readings is not None and bg_reading_time is not None and bg_readings != bg_reading_time:
                            prediction, y_test, orig = bg_pred(edited_df, bg_readings, bg_reading_time, bg_lim, steps)
                            array_0 = prediction.ravel()
                            array_1 = y_test.ravel()
                            array_2 = orig.ravel()
                            series_0 = pd.Series(array_0)
                            series_1 = pd.Series(array_1)
                            series_2 = pd.Series(array_2)
                            my_df = pd.concat([series_2, series_1], axis=1)
                            my_df.columns = ['Actual Values', 'Predicted Values']
                            model_accuracy = px.line(my_df, title='Accuracy')
                            model_accuracy.update_layout(xaxis_title="Time",
                                                    yaxis_title="glucose",)
                            st.plotly_chart(model_accuracy)

                            
                            # plotly_chart = px.line(series_0, title='Predictions')
                            # plotly_chart.update_layout(xaxis_title="Time",
                            #                         yaxis_title="glucose",)
                            # Convert series to data frame and reset index to get a column for time
                            df = series_0.reset_index()
                            df.columns = ['Time', 'Glucose']

                            # Create the plot
                            plotly_chart = px.line(df, x='Time', y='Glucose', title='Predictions')
                            plotly_chart.update_layout(xaxis_title="Time", yaxis_title="Glucose")
                            st.plotly_chart(plotly_chart)
                        else:
                            st.warning("Please complete all the steps!")
                    except Exception as e:
                        
                        print("in dietitian page")
                        print(e)
        else:
            st.error("Please use a next file")
except Exception as e:
    print(e)
    switch_page("main")


