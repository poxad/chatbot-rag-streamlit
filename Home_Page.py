import streamlit as st
# from streamlit_extras.switch_page_button import switch_page
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
# st.info("Repository on [Github](https://github.com/your-repo-link)")

st.markdown("---")


# Example prompts
example_prompts = [
    "📄 PDF Chatbot",
    "💬 All in One Chatbot",
    "🖼️ Image Chatbot",
    "💾 Database Chatbot",
    "📚 Text Narrative Chatbot"
]


button_cols = st.columns(3)
button_cols_2 = st.columns(3)

if button_cols[0].button(example_prompts[0]):
    st.switch_page("pages/📄_PDF_Chatbot.py")
elif button_cols[1].button(example_prompts[1]):
    st.switch_page("pages/📄_All_Chatbot.py")
if button_cols[2].button(example_prompts[2]):
    st.switch_page("pages/🖼️_Image_Chatbot.py")


elif button_cols_2[0].button(example_prompts[3]):
    st.switch_page("pages/💬_Chatbot.py")
elif button_cols_2[1].button(example_prompts[4]):
    st.switch_page("pages/💬_Narrative_Chatbot.py")

