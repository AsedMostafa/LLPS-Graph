from transformers import EsmModel, EsmTokenizer

model_name = "facebook/esm2_t33_650M_UR50D"
model = EsmModel.from_pretrained(model_name)
tokenizer = EsmTokenizer.from_pretrained(model_name)
seq2 = "SKQPTSAENSVAKKEDKVPVKKQKTRTVFSSTQLCVLNDRFQRQKYLSLQQMQELSNILNLSYKQVKTWFQNQRMKSKRWQKNN"
sequence = "MVLSPADKTNVKAAW"
print(len(sequence))
print(len(seq2))
inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
inputs2 = tokenizer(seq2, return_tensors="pt", add_special_tokens=False)

print(inputs)
print(inputs2)

embeddings = model(**inputs).last_hidden_state
embeddings2 = model(**inputs2).last_hidden_state

print(embeddings.shape)
print(embeddings2.shape)
