import time
from os.path import basename
import yaml

import streamlit as st
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate

from htmlTemplates import css, bot_template, user_template
from sources import sources_ref
import constants
import backend


import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Tesla T4


def on_query():
    constants.question = st.session_state.InputText
    st.session_state.InputText = ""


def handle_userinput(user_question):
    res = st.session_state.model({'query': user_question})
    answer, docs = res["result"], res["source_documents"]
    source = sources_ref[basename(docs[0].metadata["source"])]

    st.session_state.chat_history.insert(0, user_question)
    st.session_state.chat_history.insert(1, answer + "<br><br><em>Source: " + source)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def login_page():
    st.set_page_config(page_title="logotherapyGPT",
                       page_icon=":books:")

    with open('auth_config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:

        if 'db_clicked' not in st.session_state:
            st.session_state.db_clicked = False

        placeholder_box = st.empty()
        placeholder_button = st.empty()
        placeholder_box.selectbox(
            'Choose a Database',
            ("All", "Frankl's Works", "Journal of Search for Meaning"),
            key="database"
        )
        placeholder_button.button(
            "Enter",
            on_click=db_button_clicked
        )

        if st.session_state.db_clicked:
            placeholder_box.empty()
            placeholder_button.empty()
            main_page()

    elif not authentication_status:
        st.error('Username/password is incorrect')

    elif authentication_status is None:
        st.warning('Please enter your username and password')


def db_button_clicked():
    st.session_state.db_clicked = True


def main_page():
    st.write(css, unsafe_allow_html=True)

    st.markdown("# logotherapyGPT")
    st.markdown("Chosen Database: " + st.session_state.database)
    st.markdown("An AI chatbot to quickly access logotherapy information. Trained on the works of Viktor "
                "Frankl and The International Forum for Logotherapy.")

    if "model" not in st.session_state:
        with st.spinner("Loading Database..."):
            st.session_state.model = backend.load_qa(st.session_state.database)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.text_input(
        "Input",
        label_visibility="collapsed",
        placeholder="Ask a Question...",
        on_change=on_query,
        key="InputText"
    )

    print(constants.question)

    if st.session_state.model is not None and constants.question is not None:
        start = time.time()

        with st.spinner("Processing..."):
            handle_userinput(constants.question)

        end = time.time()
        print(f"\n> Answer (took {round(end - start, 2)} s.):")


if __name__ == '__main__':
    login_page()
