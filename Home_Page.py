import streamlit as st
from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 100px;
        font-size: 50px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("### 🤖 Chatbot Application with Google Gemini and Streamlit")
st.info("Repository on [Github](https://github.com/poxad/all-in-one-chatbot.git)")

st.markdown("---")

# Example prompts
example_prompts = [
    "📄 PDF Chatbot",
    "🖼️ Image Chatbot",
    "📚 Text Narrative Chatbot"
]

button_cols = st.columns(2)
button_cols_2 = st.columns(1)

if button_cols[0].button(example_prompts[0]):
    st.switch_page("pages/📄_PDF_Chatbot.py")
if button_cols[1].button(example_prompts[1]):
    st.switch_page("pages/🖼️_Image_Chatbot.py")

elif button_cols_2[0].button(example_prompts[2]):
    st.switch_page("pages/💬_Narrative_Chatbot.py")

# Add created by text
st.markdown('''
    <p style="font-size: 20px;">
    Created by <a href="https://www.linkedin.com/in/jasonjonarto" style="text-decoration: underline; color: gray;">Jason Jonarto</a>
    </p>
''', unsafe_allow_html=True)
