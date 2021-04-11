import pickle

#with open("iwslt14.tokenized.de-en/tmp/train.tags.de-en.tok.en", "r") as file:
with open("vocab.en.pkl", "rb") as file, open("iwslt14.tokenized.de-en/dict.en.txt", "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}")
		print(f"{key} {data[key]}")

print(str(data))
print(data)

with open("vocab.de.pkl", "rb") as file, open("iwslt14.tokenized.de-en/dict.de.txt", "w") as dictfile:
	data = pickle.load(file)
	for key in data:
		dictfile.write(f"{key} {data[key]}")
		print(f"{key} {data[key]}")

print(str(data))
print(data)