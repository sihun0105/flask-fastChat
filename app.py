from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

MODEL_NAME = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_text():
    user_input = request.form["user_input"]
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)[0]
    generated_text = tokenizer.decode(output, skip_special_tokens=True)

    return render_template("index.html", user_input=user_input, generated_text=generated_text)

if __name__ == "__main__":
    app.run(debug=True)
