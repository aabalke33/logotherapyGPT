import time
import re
import yaml
import streamlit as st
import backend
from os.path import basename
from yaml.loader import SafeLoader
from streamlit_authenticator import Authenticate
import constants
from sources import sources_ref
from htmlTemplates import css, bot_template, user_template, source_template


def on_query():
    constants.question = st.session_state.InputText
    st.session_state.InputText = ""


def handle_userinput(user_question):

    res = st.session_state.model({'query': user_question})
    answer, docs = res["result"], res["source_documents"]

    st.session_state.chat_history.insert(0, user_question)
    st.session_state.chat_history.insert(1, answer)
    st.session_state.chat_history.insert(2, docs)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 3 == 0:
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        elif i % 3 == 1:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            source = sources_ref[basename(message[0].metadata["source"])]
            output = ("Source: <em>\"" + message[0].page_content +
                      "\"<br><br></em>" + source)

            st.write(source_template.replace("{{MSG}}", str(output)), unsafe_allow_html=True)


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
        main_page()

    elif not authentication_status:
        st.error('Username/password is incorrect')

    elif authentication_status is None:
        st.warning('Please enter your username and password')


def main_page():
    st.write(css, unsafe_allow_html=True)

    st.markdown("# logotherapyGPT")
    st.markdown("An AI chatbot to quickly access logotherapy information. Trained on the works of Viktor "
                "Frankl and The International Forum for Logotherapy.")

    if "model" not in st.session_state:
        with st.spinner("Loading Database..."):
            st.session_state.model = backend.load_qa()

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
    # login_page()
    main_page()
