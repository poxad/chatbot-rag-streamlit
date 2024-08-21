import time
import os
import joblib
import streamlit as st
import pandas as pd
from pathlib import Path
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a data/ folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)


try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}



def user_input(user_question):
    llm = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY)
    response = llm.invoke(user_question)   
    return response.content


@st.dialog("Clear chat history?")
def modal():
    button_cols = st.columns([1, 1])  # Equal column width for compact fit
    
    # Add custom CSS for button styling
    st.markdown(
        """
        <style>
        .stButton button {
            width: 100%;
            padding: 10px;
            font-size: 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if button_cols[0].button("Yes"):
        clear_chat_history()
        st.rerun()
    elif button_cols[1].button("No"):
        st.rerun()
        
def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('gemini_history', None)
    
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar allows a list of past chats
with st.sidebar:
    st.write('# Sidebar Menu')

    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    if st.button("Clear Chat History", key="clear_chat_button"):
        # st.write(st.session_state)
        modal()

    
    st.session_state.chat_title = f'Narrative-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.write('# Chat with Gemini')

# Chat history (allows to ask multiple questions)
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except:
    st.session_state.messages = []
    st.session_state.gemini_history = []

st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)



# Check if 'messages' is not in st.session_state and initialize with a default message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "✨",  # or any valid emoji
        "content": "Hey there, I'm your Narrative Chatbot. Don't hesitate to ask any questions to me 😊"
    })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.markdown(message['content'])

# React to user input
if prompt := st.chat_input('Your message here...'):
    
    # Save this as a chat for later
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append(
        dict(
            role='user',
            content=prompt,
        )
    )

    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt)
        
    # Display assistant response in chat message container
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=response,
            avatar=AI_AVATAR_ICON,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history
    # Save to file
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')


# print(st.session_state)