from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from diffusers import StableDiffusionPipeline
import spacy
import os
import torch

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

llama321Binstruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", pad_token="[PAD]")
llama321Binstruct_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")



def llama321Binstruct_generate_response(prompt):

    inputs = llama321Binstruct_tokenizer(prompt, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = llama321Binstruct_model.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=llama321Binstruct_tokenizer.eos_token_id,
        )
    return llama321Binstruct_tokenizer.decode(outputs[0], skip_special_tokens=True)

def llama323b_generate_response(prompt):

    llama323Btokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", pad_token="[PAD]")
    llama323Bmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B",torch_dtype=torch.float16)
    llama323Bmodel.to('cuda')
    inputs = llama323Btokenizer(prompt, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs = llama323Bmodel.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            pad_token_id=llama323Btokenizer.eos_token_id,
        )
    return llama323Btokenizer.decode(outputs[0], skip_special_tokens=True)


def llama323binstruct_generate_response(prompt):
    llama323Binstructtokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", pad_token="[PAD]")
    llama323Binstructmodel = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct",
                                                                  torch_dtype=torch.float16)

    llama323Binstructmodel.to('cuda')
    inputs = llama323Binstructtokenizer(prompt, return_tensors="pt", padding=True).to('cuda')
    with torch.no_grad():
        outputs = llama323Binstructmodel.generate(
            inputs["input_ids"],
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            pad_token_id=llama323Binstructtokenizer.eos_token_id,
        )
    return llama323Binstructtokenizer.decode(outputs[0], skip_special_tokens=True)


def gpt2finetuned_generate_response(prompt):
    gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2model = GPT2LMHeadModel.from_pretrained('gpt2')
    inputs = gpt2tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gpt2model.generate(
            inputs.input_ids,
            attention_mask=inputs['attention_mask'],
            max_length=500,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=gpt2tokenizer.eos_token_id,
        )
    return gpt2tokenizer.decode(outputs[0], skip_special_tokens=True)


def stablediffusion_generate_image(prompt):
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    model.to('cuda')
    image = model(prompt).images[0]
    image_path = os.path.join(app.root_path, 'static/images', 'output.png')
    image.save(image_path)
    return image_path


def detect_intent(prompt):
    doc = nlp(prompt.lower())
    if any(token.lemma_ in ["image", "visualize", "art", "picture"] for token in doc):
        return "image_generation"
    elif any(token.lemma_ in ["detail", "explain", "complex"] for token in doc):
        return "complex_text_generation"
    elif "simple" in prompt:
        return "simple_text_generation"
    else:
        return "general_text_generation"

def generate_response(prompt):
    return llama321Binstruct_tokenizer(prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')
    response = generate_response(prompt)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
