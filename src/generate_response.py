from transformers import LlamaForCausalLM, LlamaTokenizer
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

def generate_response(question, nodes_image1=None, nodes_image2=None, similarity_score=None):
    # Load Llama model and tokenizer
    model_name ="meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=True)

    # Dynamic prompt generation
    prompt = f"Question: {question}\n"
    if nodes_image1:
        prompt += f"Nodes in Image 1: {nodes_image1}\n"
    if nodes_image2:
        prompt += f"Nodes in Image 2: {nodes_image2}\n"
    if similarity_score is not None:
        prompt += f"Similarity Score: {similarity_score:.2f}\n"
    prompt += "Provide a human-readable response."

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
