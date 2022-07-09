import yaml
from chatterbot import ChatBot
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response, get_random_response
from chatterbot.trainers import ListTrainer

import logging

logging.basicConfig(level=logging.INFO)

# Read YAML file
with open("data/emotions.yaml", "r") as stream:
    data_loaded = yaml.safe_load(stream)

data = []

for key, value in data_loaded.items():
    for i in value["pregunta"]:
        for j in value["respuesta"]:
            data.append(i)
            data.append(j)

chatbot = ChatBot(
    name="Charlie", 
    read_only=False,
    preprocessors=[
        "chatterbot.preprocessors.clean_whitespace",
        "chatterbot.preprocessors.convert_to_ascii",
        "chatterbot.preprocessors.unescape_html"
        ], 
    logic_adapters = [
            {
                "import_path": "chatterbot.logic.MathematicalEvaluation"
                }, 
            {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": LevenshteinDistance(language="spa"),
            "response_selection_method": get_random_response,
            "default_response": "No entiendo a qué te refieres",
            "maximum_similarity_threshold": 0.90
            }
            ]
)

trainer = ListTrainer(chatbot)
trainer.train(data, "chatterbot.corpus.spanish")

# Get a response to the input text
response = chatbot.get_response("Soy feliz")

print(response)

#print(data)
#print(data_loaded)
