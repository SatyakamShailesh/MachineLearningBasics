from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode a text input
input_prompt = "Lord Shri Ram is the "
input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
print(input_ids)

# Generate text based on the input
output = model.generate(input_ids, max_length=50, num_return_sequences=2, num_beams=5)

# Decode the generated text
generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(len(output))]

# Print the generated text
print(generated_texts)

