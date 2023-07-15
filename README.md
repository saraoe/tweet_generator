# Tweet Generation
Using gpt-neo to generate tweets based on corpus of Danish tweets during COVID-19. A streamlit interface for displaying pre-generated tweets and generating new tweets from custom prompts can be found in ``streamlit_generate_tweets.py``. Part of the Danish HOPE project.

## Project Organization
```
├── README.md                       <- The top-level README for this project.                       
├── genrated tweets                 <- pre-generated tweets              
├── logs
├── get_tweets.py                   <- getting the relevant tweets from classification of emotions
├── train.py                        <- finetuning gpt-neo¨
├── generate_tweets.py              <- generating tweets
├── streamlit_generate_tweets.py    <- streamlit app                       
├── description.txt                 <- description text for streamlit app
└── requirement.txt
```
