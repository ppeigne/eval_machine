import yaml
import json
import os

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

with open("prompts.yml", "r") as f:
    prompts_config = yaml.safe_load(f)

with open("openai_api_call_funcs.json", "r") as f:
    api_call_funcs = json.loads(f.read())

config = config | prompts_config | api_call_funcs