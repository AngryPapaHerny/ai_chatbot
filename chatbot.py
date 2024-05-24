import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache_resource
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_resource
def get_dataset():
    try:
        df = pd.read_csv('wellness_dataset.csv')
        df['embedding'] = df['embedding'].apply(json.loads)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=['embedding', '챗봇'])

model = cached_model()
df = get_dataset()

st.header('엉뚱한 AI 심리상당 챗봇')
st.markdown("[❤️학생 만족도 조사 링크](https://docs.google.com/forms/d/e/1FAIpQLScmBZl3Iy-otn5GocwWD4fEBRyNrUYftJnjt8PGGI5LBUMVyg/viewform?usp=sf_link)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('당신: ', '')
    submitted = st.form_submit_button('전송')

if st.button("Reset"):
    st.session_state['generated'] = []
    st.session_state['past'] = []

if submitted and user_input:
    try:
        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]

        st.session_state.past.append(user_input)
        st.session_state.generated.append(answer['챗봇'])
    except Exception as e:
        st.error(f"Error processing your input: {e}")

for i in range(len(st.session_state['past'])):
    with st.chat_message('user'):
        st.write(st.session_state['past'][i])
    if len(st.session_state['generated']) > i:
        with st.chat_message('bot'):
            st.write(st.session_state['generated'][i])
