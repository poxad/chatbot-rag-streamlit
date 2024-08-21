import time
import os
import joblib
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
import datetime

load_dotenv()

# Constants
CHAT_HISTORY_FILE = 'data/past_chats_list'
CHAT_MESSAGES_DIR = 'data/'

# Load or initialize past chats
if os.path.exists(CHAT_HISTORY_FILE):
    past_chats = joblib.load(CHAT_HISTORY_FILE)
else:
    past_chats = {}

# Initialize new chat ID
new_chat_id = None

# Create a data/ folder if it doesn't already exist
os.makedirs(CHAT_MESSAGES_DIR, exist_ok=True)

# Function to clear chat history
def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('chat_title', None)
    for file in Path(CHAT_MESSAGES_DIR).glob('*'):
        file.unlink()

# Function to start a new chat
def start_new_chat():
    global new_chat_id
    new_chat_id = f'{time.time()}'
    st.session_state.chat_id = new_chat_id
    st.session_state.chat_title = f'Chat-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'
    past_chats[new_chat_id] = st.session_state.chat_title
    joblib.dump(past_chats, CHAT_HISTORY_FILE)
    st.session_state.messages = []
    st.session_state.gemini_history = []

# Automatically start a new chat if no chat ID is set
if 'chat_id' not in st.session_state or not st.session_state.chat_id:
    start_new_chat()
    st.experimental_rerun()

# Sidebar UI
with st.sidebar:
    st.write('# Sidebar Menu')
    
    # Display existing chats or disable selectbox if no history
    if past_chats:
        selected_chat = st.selectbox(
            label='Select a chat history',
            options=list(past_chats.keys()) + ['New Chat'],
            format_func=lambda x: past_chats.get(x, 'New Chat') if x != 'New Chat' else 'New Chat',
            key='chat_selector'
        )
        
        if selected_chat == 'New Chat':
            start_new_chat()
            st.experimental_rerun()
        else:
            st.session_state.chat_id = selected_chat
    else:
        st.selectbox(
            label='Select a chat history',
            options=[''],
            disabled=True,
            key='chat_selector'
        )
    
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.experimental_rerun()

# Main UI
st.write('# Chat with Gemini')

# Load or initialize chat history
if 'chat_id' in st.session_state:
    chat_id = st.session_state.chat_id
    try:
        st.session_state.messages = joblib.load(f'{CHAT_MESSAGES_DIR}/{chat_id}-st_messages')
        st.session_state.gemini_history = joblib.load(f'{CHAT_MESSAGES_DIR}/{chat_id}-gemini_messages')
    except:
        st.session_state.messages = []
        st.session_state.gemini_history = []

    # Initialize your chat model here
    st.session_state.model = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "✨",  # or any valid emoji
            "content": "Hey there, I'm your Text Extraction chatbot. Please upload the necessary files in the sidebar to add more context to this conversation."
        })

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(
            name=message.get('role', 'user'),
            avatar=message.get('avatar', None),
        ):
            st.markdown(message['content'])

    # React to user input
    if prompt := st.chat_input('Your message here...'):
        with st.chat_message('user'):
            st.markdown(prompt)
        
        # Handle the AI response
        with st.spinner("Waiting for AI response..."):
            response = "This is a dummy response"  # Replace with actual AI response code

        with st.chat_message(name='ai', avatar='✨'):
            st.markdown(response)

        # Save the messages and chat history
        st.session_state.messages.append(
            dict(
                role='user',
                content=prompt,
            )
        )
        st.session_state.messages.append(
            dict(
                role='ai',
                content=response,
                avatar='✨',
            )
        )
        st.session_state.gemini_history = []  # Update with actual chat history if needed
        joblib.dump(st.session_state.messages, f'{CHAT_MESSAGES_DIR}/{chat_id}-st_messages')
        joblib.dump(st.session_state.gemini_history, f'{CHAT_MESSAGES_DIR}/{chat_id}-gemini_messages')
