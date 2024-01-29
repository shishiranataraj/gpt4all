from langchain_community.llms import GPT4All

# Instantiate the model. Callbacks support token-wise streaming
model = GPT4All(model="mistral-7b-openorca.Q4_0.gguf", n_threads=8)

# Generate text
response = model("Once upon a time, ")