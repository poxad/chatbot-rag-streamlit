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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import datetime
from PIL import Image
import easyocr
import numpy as np
import pdf2image
from io import BytesIO

# Load environment variables from a .env file
load_dotenv()

# Get the Google API key from environment variables
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configure the Google Generative AI library with the API key
genai.configure(api_key=GOOGLE_API_KEY)
api_key = GOOGLE_API_KEY

# Create a unique ID for a new chat using the current timestamp
new_chat_id = f'{time.time()}'

# Constants for AI model
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'

# Create a 'data/' folder if it doesn't already exist
os.makedirs('data/', exist_ok=True)

# Load past chats from the 'data/' folder, if available
try:
    past_chats = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}

# Function to extract text from PDF and image files
def get_file_text(files):
    text = ""
    # Initialize OCR (Optical Character Recognition) reader for English
    reader = easyocr.Reader(['en'])
    
    # Process each uploaded file
    for file in files:
        file_name = file.name.lower()
        
        # If the file is a PDF
        if file_name.endswith(".pdf"):
            text += f"\n--- The text below is from {file.name} ---\n"
            check = False
            
            try:
                # Try to extract selectable text from the PDF
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
                        check = True
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
            
            if not check:
                # If no text can be selected, use OCR to extract text from PDF images
                file.seek(0)
                images = pdf2image.convert_from_bytes(file.read())
                for page in images:
                    results = reader.readtext(np.array(page))
                    for i in results:
                        text += i[1] + " "
        
        # If the file is an image (PNG, JPG, JPEG)
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file)
            text += f"\n--- The text below is from {file.name} ---\n"
            results = reader.readtext(np.array(image))
            for i in results:
                text += i[1] + " "

    return text

# Function to split extracted text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to convert text chunks into vectors and save them
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational AI chain for answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question in a detailed and expressive manner, providing as much relevant information from the provided context as possible. Make sure the response is storytelling and engaging.

    If the context is not provided or the context is empty:
    Answer like a normal AI chatbot with a storytelling effect.

    If the answer is not in the provided context:
    Answer like a normal chatbot if there is no context provided and specify that the answer is not in the provided context.

    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """
    # Set up the AI model and the prompt template
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate AI responses
def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to clear the chat history and remove stored data
def clear_chat_history():
    st.session_state.pop('chat_id', None)
    st.session_state.pop('messages', None)
    st.session_state.pop('gemini_history', None)
    
    # Delete all files in the 'data/' folder
    for file in Path('data/').glob('*'):
        file.unlink()

# Sidebar for navigating past chats and uploading files
with st.sidebar:
    st.write('# Sidebar Menu')

    # Select an existing chat or start a new one
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    
    # Button to clear chat history with a confirmation modal
    if st.button("Clear Chat History", key="clear_chat_button"):
        with st.dialog("Clear chat history?"):
            button_cols = st.columns([1, 1])  # Equal column width for compact fit
            if button_cols[0].button("Yes"):
                clear_chat_history()
                st.rerun()
            elif button_cols[1].button("No"):
                st.rerun()

    # Upload files (PDFs or images) for processing
    all_files = st.file_uploader("Upload PDF or Image Files and Click on the Submit & Process Button", accept_multiple_files=True, type=['pdf','csv','png','jpg','jpeg'])

    # Process the uploaded files when the button is clicked
    if st.button("Submit & Process", key="process_button", disabled=not all_files):
        with st.spinner("Processing..."):
            raw_text = get_file_text(all_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, api_key)
            st.success("Done")

    # Set the title of the new chat based on the current date and time
    st.session_state.chat_title = f'All-{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}'

st.write('# Chat with Gemini')

# Load chat history for the current chat ID
try:
    st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
    st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
except FileNotFoundError:
    st.session_state.messages = []
    st.session_state.gemini_history = []

# Initialize the AI chat model
st.session_state.model = genai.GenerativeModel('gemini-pro')
st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)

# Initialize chat history with a welcome message if empty
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "✨",  # AI avatar icon
        "content": "Hey there, I'm your Text Extraction chatbot. Please upload the necessary files in the sidebar to add more context to this conversation."
    })

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(
        name=message.get('role', 'user'),
        avatar=message.get('avatar', None),
    ):
        st.markdown(message['content'])

# Handle user input and AI response
if prompt := st.chat_input('Your message here...'):
    # Save the chat title if it's a new chat
    if st.session_state.chat_id not in past_chats.keys():
        past_chats[st.session_state.chat_id] = st.session_state.chat_title
        joblib.dump(past_chats, 'data/past_chats_list')
    
    # Display the user's message
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Add the user's message to chat history
    st.session_state.messages.append({
        "role": 'user',
        "content": prompt,
    })

    # Generate and display the AI response
    with st.spinner("Waiting for AI response..."):
        response = user_input(prompt, api_key)
    
    # Display the AI's response
    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(response)
    
    # Add the AI's response to chat history
    st.session_state.messages.append({
        "role": MODEL_ROLE,
        "content": response,
        "avatar": AI_AVATAR_ICON,
    })
    
    # Update and save the chat history
    st.session_state.gemini_history = st.session_state.chat.history
    joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
    joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')
