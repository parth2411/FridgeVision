# Example: reuse your existing OpenAI setup

from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="LM Studio Community/Meta-Llama-3-8B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "Provide answer in one paragraph"},
    {"role": "user", "content": "generate recipe using ingredients like butter, milk, pasta, cheese"}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)


""""
import requests
import json
import random

countries = [
    "cheese",
    "milk",
    "pasta",
    "pepper"
]
country = countries
model = "llama3"

prompt = f"recommend one delicious recipe using ingredients present in {country}, and give the step to make it . Do not use common names. Respond using JSON."

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"Generating a sample user in {country}")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)

print(json.dumps(json.loads(json_data["response"]), indent=2))
"""
""""
import json
import requests

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`

model = 'llama3'  # TODO: update this for whatever model you wish to use


def generate(system_prompt, user_input, context):
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'system_prompt': system_prompt,
                          'prompt': user_input,
                          'context': context,
                      },
                      stream=True)

    r.raise_for_status()

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # the response streams one token at a time, print that as we receive it
        print(response_part, end='', flush=True)
        if 'error' in body:
            raise Exception(body['error'])
        if body.get('done', False):
            return body['context']


def main():
    context = []  # the context stores a conversation history, you can use this to make the model more context aware

    system_prompt = "You are a helpful AI assistant devoted to providing accurate and delightful recipes. Below are some guidelines for you to follow when delivering a recipe in response to a request: Pay attention to all details of the request to ensure the recipe meets the user's specifications.Ask the user for their preferred measurement system (metric or imperial) if not specified.List out the ingredients first, including quantities. Provide detailed cooking times, temperatures, and any special kitchen equipment needed.Provide step-by-step instructions for prepping, mixing, cooking, plating, and any other necessary steps, detailed enough for an inexperienced cook to follow. Include safety tips and special techniques as applicable.Offer substitutions for allergens and suggest variations for dietary preferences. If potential allergens are present, notify the user for safety.Include nutritional information and serving suggestions to enhance the dining experience.Encourage feedback on the recipe to improve future recommendations."

    while True:
        user_input = input(
            "Enter the ingredients you have in your fridge (separated by commas), or type 'quit' to exit: ")

        if user_input.lower() == 'quit':
            break

        print()
        context = generate(system_prompt, user_input, context)
        print()


if __name__ == "__main__":
   # main()
"""