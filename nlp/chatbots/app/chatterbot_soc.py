from chatterbot import ChatBot
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response, get_random_response
from chatterbot.trainers import ListTrainer

import logging

logging.basicConfig(level=logging.INFO)

_KB = [
    "FW-025 ip blacklist y recursiva",
    "notificar a cliente",
    "FW-025 ip blacklist y no recursiva",
    "asignar a L2",
    "FW-025 ip no blacklist",
    "cerrar sin notificar"
]

chatbot = ChatBot(
    name="SOChat", 
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
            "default_response": "No entiendo a qué te refieres. Prueba con código_alera, blacklist, recursividad",
            "maximum_similarity_threshold": 0.90
            }
            ]
)

trainer = ListTrainer(chatbot)
trainer.train(_KB)

response = chatbot.get_response("FW-025 ip blacklist y no recursiva")
print(response)
