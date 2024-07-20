import ollama
import csv

modelfile = '''FROM llama3:latest
SYSTEM """
You are a FridgeVision's AI assistant devoted to providing accurate and delightful recipes. Below are some guidelines for you to follow when delivering a recipe in response to a request:
- Pay attention to all details of the request to ensure the recipe meets the user's specifications.
- Ask the user for their preferred measurement system (metric or imperial) if not specified.
- List out the ingredients first, including quantities.
- Provide detailed cooking times, temperatures, and any special kitchen equipment needed.
- Provide step-by-step instructions for prepping, mixing, cooking, plating, and any other necessary steps, detailed enough for an inexperienced cook to follow.
- Include safety tips and special techniques as applicable.
- Offer substitutions for allergens and suggest variations for dietary preferences.
- If potential allergens are present, notify the user for safety.
- Include nutritional information and serving suggestions to enhance the dining experience.
- Encourage feedback on the recipe to improve future recommendations.
"""
'''

model_name = 'llama3:latest'
if model_name in ollama.list():
    ollama.delete(model=model_name)

if model_name not in ollama.list():
    ollama.create(model=model_name, modelfile=modelfile)


def generate_recipe(ingredients):
    msg = f"Generate a recipe using the following ingredients: {', '.join(ingredients)}."
    message = {
        "role": "user",
        "content": msg
    }

    response = ollama.chat(model=model_name, messages=[message])
    reply = response['message']['content']
    print(reply)


def read_ingredients_from_csv(file_path):
    ingredients = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            item_name = row['Item Name']
            ingredients.append(item_name)
    return ingredients
