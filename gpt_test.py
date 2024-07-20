from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo preentrenado y el tokenizador
model_name = 'gpt2'  # Puedes cambiar a otro modelo como 'gpt2-medium', 'gpt2-large', etc.
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def ask_question(question, max_length=50):
    input_ids = tokenizer.encode(question, return_tensors='pt')

    # Generar una respuesta
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    # Decodificar y mostrar la respuesta
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

while True:
    user_input = input("Pregunta: ")
    if user_input.lower() in ['exit', 'quit']:
        break
        
    # Obtener respuesta del modelo
    answer = ask_question(user_input)
    print(f"Respuesta: {answer}")
