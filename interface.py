import streamlit as st
from main import SentimentAnalyzer, Dataset
st.set_page_config(
    page_title='Sentiment Analysis Project',
    page_icon=':fire:',
    layout='centered',
    initial_sidebar_state='auto'
)

@st.cache_resource
def execute() -> SentimentAnalyzer:
    """
    Executes the whole process.
    """
    dataset = Dataset('dataset.csv', 80)
    splittedData = dataset.get_reviews_and_labels()
    analyzer = SentimentAnalyzer(splittedData)
    analyzer.tokenize()
    analyzer.build_layers()
    analyzer.save()
    return analyzer

ai = execute()


st.title('Sentiment Analysis Project')
review = st.text_input('Enter the review:')

if st.button('Analyze') and review:
    st.write('Analyzing process has been started.')
    result = ai.predict_review(review)
    st.write(f'Prediction: {"Positive" if result > 0.5 else "Negative"} with {result} confidence.')
    
