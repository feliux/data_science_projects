# Chatbots

Chatbots examples using `nltk` and `chatterbot`.

## Usage

Chatbots on `app` folder. You can use `streamlit` for interacting with the bot.

Docker container provided to test your chatbot `$ docker-compose up -d`. To use `streamlit` just add the following entrypoint on `docker-compose.yml`: `$ streamlit run app.py`

**Extra**

Consider to download a cosrpus: `$ python -m spacy download en`.

## References

[Chatterbot tutorial](https://www.datacamp.com/community/tutorials/building-a-chatbot-using-chatterbot)

[DATA: dialogues](http://convai.io/data/)

- `data_tolokers.json`: data collected during DeepHack.Chat hackathon in July 2-8 2018 via Yandex.Toloka service (paid workers). 3127 dialogues.

- `data_intermediate.json`: dialogues by the bots from DeepHack.Chat and volunteers collected from July 9 to October 29, 2018. 291 dialogues

- `data_volunteers.json`: wild evaluation for ConvAI final round from October 29 to December 17, 2018. Dialogues by bots from ConvAI finals and volunteers. 1111 dialogues.

[Corpus](https://code.ihub.org.cn/projects/527/repository/revisions/master/show/chatterbot_corpus/data/spanish)
