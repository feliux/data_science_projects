from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

CONVERSATION = open("data/conversation.txt").readlines()
CONVERSATION = [row.replace("\n", "") for row in CONVERSATION]

# Create a new chat bot named Charlie
chatbot = ChatBot(
    name="Charlie", 
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

trainer = ListTrainer(chatbot)
trainer.train(CONVERSATION)

# Get a response to the input text
response = chatbot.get_response("¿Cómo estás?")

print(response)