import os
from dotenv import load_dotenv
import pangea.exceptions as pe
from pangea.config import PangeaConfig
from pangea.services import FileScan
from pangea.tools import logger_set_pangea_config
import streamlit as st
load_dotenv()
token = os.getenv("PANGEA_FILE_SCAN_TOKEN")
domain = os.getenv("PANGEA_DOMAIN")

def scan_excel(excel_data):
    # To work in sync it's need to set up queue_retry_enable to true and set up a proper timeout
    # If timeout it's so little service won't end up and will return an AcceptedRequestException anyway
    with st.spinner('Scanning your file...'):
        config = PangeaConfig(domain=domain, queued_retry_enabled=True, poll_result_timeout=120)
        client = FileScan(token, config=config, logger_name="pangea")
        logger_set_pangea_config(logger_name=client.logger.name)
        try:
            
            response = client.file_scan(file=excel_data, verbose=True, provider="crowdstrike")
            print(f"Response: {response.result}")
            if response.result.data.verdict == "benign":
                st.success("File safe!")
                return True
            else:
                st.error("File was detected to be malicious")
                return False
        except pe.PangeaAPIException as e:
            print(f"Request Error: {e.response.summary}")
            for err in e.errors:
                print(f"\t{err.detail} \n")
            st.warning("Please try another file.")
            return False
        




