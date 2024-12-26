import streamlit as st
st.title('Sentiment Analysis')
url = st.text_input('Enter the URL')
if st.button('Analyze'):
    st.write('Analyzing the sentiment of the URL:', url)