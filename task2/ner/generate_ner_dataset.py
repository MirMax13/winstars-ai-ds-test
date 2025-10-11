import random
import json

animals = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]

positive_templates = [
    "I see a {animal}.",
    "There is a {animal} in the picture.",
    "The {animal} is running.",
    "It looks like a {animal}.",
    "A {animal} appears in this image.",
    "Do you see the {animal}?",
    "This photo shows a {animal}.",
    "The {animal} is standing on the ground.",
    "Look, a {animal}!",
    "Here we can find a {animal}.",
    "A {animal} is jumping.",
    "The {animal} is playing.",
    "It is a {animal}.",
    "I spotted a {animal}.",
    "The {animal} is very cute.",
    "There is a big {animal}.",
    "A small {animal} is here.",
    "The {animal} is eating.",
    "I love the {animal}.",
    "The {animal} is sleeping.",
    "I can't find the {animal}.",
    "The {animal} is hiding.",
    "A wild {animal} appears.",
    "The {animal} is climbing a tree.",
    "I saw a {animal} at the park.",
    "The {animal} is swimming.",
    "A {animal} is flying.",
    "The {animal} is barking.",
    "The {animal} is meowing.",
    "The {animal} is grazing.",
    "The {animal} is crawling.",
    "The {animal} is buzzing.",
    "The {animal} is chirping.",
    "The {animal} is hopping.",
    "The {animal} is galloping.",
    "The {animal} is fluttering.",
    "The {animal} is pecking.",
    "The {animal} is scurrying.",
    "The {animal} is spinning a web.",
    "The {animal} is digging.",
    "The {animal} is foraging.",
    "The {animal} is nesting.",
    "The {animal} is migrating.",
    "The {animal} is basking in the sun.",
    "The {animal} is howling.",
    "The {animal} is mooing.",
    "The {animal} is baaing.",
    "The {animal} is chirping.",
]

negative_templates = [
    "It's not a {animal}.",
    "I don't see a {animal} here.",
    "This is definitely not a {animal}.",
    "There is no {animal} in this image.",
    "I think this is something else, not a {animal}.",
    "This picture doesn't contain a {animal}.",
    "I can't find any {animal} here.",
    "This is not what I would call a {animal}.",
    "This is clearly not a {animal}.",
    "I am sure this is not a {animal}.",
    "This is something different, not a {animal}.",
    "This image lacks a {animal}.",
    "I don't believe there is a {animal} here."
]
examples = []

templates = positive_templates + negative_templates
for animal in animals:
    for template in templates:
        sentence = template.format(animal=animal)
        tokens = sentence.replace(".", "").replace(",", "").replace("'", "").replace("!", "").replace("?", "").split()
        tags = ["O"] * len(tokens)
        for i, token in enumerate(tokens):
            if token.lower() == animal.lower():
                tags[i] = "B-ANIMAL"
        examples.append({"tokens": tokens, "ner_tags": tags})

random.shuffle(examples)

with open("ner/data/ner_dataset.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print(f"✅ Generated {len(examples)} examples for NER dataset.")
