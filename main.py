import streamlit as st
from healthcare_data import healthcare_professionals, health_conditions
import shortuuid
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Initialize Session",
    initial_sidebar_state="collapsed"
)

with st.form("create_session"):
    selected_professional = st.selectbox('Please select your role:', healthcare_professionals, index=7)
    selected_health_condition = st.selectbox('Please select the focus of this session: ', health_conditions)
    submit_create_session = st.form_submit_button("Start Session", use_container_width=True)

if submit_create_session:
    st.session_state["collab_code"] = shortuuid.ShortUUID().random(length=7)
    st.session_state["selected_professional"] = selected_professional
    st.session_state["selected_health_condition"] = selected_health_condition

    if selected_professional in ["dietitian"] and selected_health_condition in ["diabetes type 1"]:

        # print(response.content)
        switch_page(selected_professional)
    else:
        st.warning("We will make that available soon!")