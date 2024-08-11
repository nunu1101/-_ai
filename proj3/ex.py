# STEP 1
from transformers import pipeline

# STEP 2
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")

# STEP 3
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# STEP 4
result = classifier(text)

# STEP 5
print(result)