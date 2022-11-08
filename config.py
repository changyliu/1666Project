import json

def config():
    with open('./default_config.json', mode='rt', encoding='utf-8') as file:
        config = json.load(file)
    return config