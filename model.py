from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def generate_text(prompt):
    messages = [
        {"role": "system", "content": '''
                    You will write a function that transforms an image such that it will
                    be easier to detect spots in it. You will be prompted with example functions, write
                    a different function that is similar. Only return the function itself. The function must have
                    two arguments: image (numpy.array) and clip (boolean).
         '''},
        {"role": "user", "content": prompt},
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    
    outputs = model.generate(tokenized_chat, max_new_tokens=512) 

    return tokenizer.decode(outputs[0])

def read_file_to_string(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


function_bank = read_file_to_string('function_bank.py')
print(generate_text(""))
