'''
Streamlit app for displaying already generated tweets and generating new ones
'''

import streamlit as st
import ndjson, os

from generate_tweets import generate_tweets

def tweets_gen(path: str):
    with open(path) as f:
            reader = ndjson.reader(f)

            for post in reader:
                yield post

with open('description.txt') as f:
        description_text = f.read()

## app starts ##

# Initialization session_sate
if 'n_tweet' not in st.session_state:
    st.session_state['n_tweet'] = 0
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None

# title
st.title('Generate Tweets')

# choose model
label = st.sidebar.selectbox('Select model', ['-- select model --', 'Negative', 'Positive', 'Neutral'])
if label == '-- select model --':
    st.markdown(description_text)
else:
    label = label.upper()[:3]

    tweets_path = os.path.join('generated_tweets', f'{label}_tweets.ndjson')

    tweets_dict = {}
    for post in tweets_gen(tweets_path):
        prompt = post['prompt']
        tweets = post['generated_text']
        tweets_dict[prompt+'...'] = tweets

    selected_prompt = st.sidebar.radio('Select prompt', list(tweets_dict.keys())+['Custom prompt'])
    
    if selected_prompt in tweets_dict.keys():
        if selected_prompt != st.session_state['prompt']: # new prompt
            st.session_state['n_tweet'] = 0
            st.session_state['prompt'] = selected_prompt

        st.write('**Generated Tweets:**')

        col1, col2, col3, col4, col5 = st.columns(5)
        if col1.button('Previous tweet', disabled = (st.session_state['n_tweet'] == 0)):
            st.session_state['n_tweet'] -= 1
        if col5.button('Next tweet', disabled = (st.session_state['n_tweet'] == 19)):
            st.session_state['n_tweet'] += 1
        
        i = st.session_state['n_tweet']
        col3.write(f'{i+1} out of 20')

        st.markdown(tweets_dict[selected_prompt][i])

    if selected_prompt == 'Custom prompt':
        prompt = st.text_input('Input text')
        if prompt:
            tweet = generate_tweets(label, num_seq=1, prompt=prompt)
            st.write('**Generated Tweet:**')
            st.write(tweet[0])

