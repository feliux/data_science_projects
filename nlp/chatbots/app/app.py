import streamlit as st
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer 
import json

data = json.loads(open(r"data/data_tolokers.json", "r").read())

tra = []
for k, row in enumerate(data):
    #print(k)
    tra.append(row["dialog"][0]["text"])

st.sidebar.title("NLP Bot")
st.title(
    """
    NLP Bot  
    NLP Bot is an NLP conversational chatterbot. Initialize the bot by clicking the "Initialize bot" button. 
    """
)

bot = ChatBot(
    name="Bot", 
    read_only=False, 
    preprocessors=[
        "chatterbot.preprocessors.clean_whitespace",
        "chatterbot.preprocessors.convert_to_ascii",
        "chatterbot.preprocessors.unescape_html"], 
    logic_adapters = [
        "chatterbot.logic.MathematicalEvaluation", 
        "chatterbot.logic.BestMatch"
        ]
)

ind = 1
if st.sidebar.button("Initialize bot"):
    trainer2 = ListTrainer(bot) 
    trainer2.train(tra)
    st.title("Your bot is ready to talk to you")
    ind = ind +1

        
user_input = st.text_input("You: ", "So, what is in your mind")

if True:
    st.text_area(
        "Bot:", 
        value=bot.get_response(user_input), 
        height=200, 
        max_chars=None, 
        key=None
    )
else:
    st.text_area(
        "Bot:", 
        value="Please start the bot by clicking sidebar button", 
        height=200, 
        max_chars=None, 
        key=None
    )