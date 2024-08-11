# STEP 1
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from fastapi import FastAPI, Form

# STEP 2
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_opus_books_model")
model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_opus_books_model")
translator = pipeline("translation_xx_to_yy", model="stevhliu/my_awesome_opus_books_model")

app = FastAPI()

@app.post("/translate/")
async def translate(text: str = Form()):

    # STEP 3
    # TEXT

    # STEP 4
    inputs = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
    tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = translator(text)

    return {"result": result}