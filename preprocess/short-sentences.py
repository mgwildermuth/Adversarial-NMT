#!/usr/bin/env python

def shrink(path1, path2):
	f1_lines = []
	f2_lines = []

	with open(f"../data/iwslt14.tokenized.de-en/{path1}", "r") as file1, open(f"../data/iwslt14.tokenized.de-en/{path2}", "r") as file2, open(f"../data/old_{path1}", "w") as f1_write, open(f"../data/old_{path2}", "w") as f2_write:
		f1_lines = file1.readlines()
		f2_lines = file2.readlines()
		all_f1 = file1.read()
		all_f2 = file2.read()

		f1_write.write(all_f1)
		f2_write.write(all_f2)

	if len(f1_lines) != len(f2_lines):
		print("~Internally screaming~")

	print(f"Eliminating lines over 50 words in {path1}, {path2}")
	with open(f"../data/iwslt14.tokenized.de-en/{path1}", "w") as file1, open(f"../data/iwslt14.tokenized.de-en/{path2}", "w") as file2:
		for x in range(len(f1_lines)):
			if len(f1_lines[x].split(' ')) <= 50 and len(f2_lines[x].split(' ')) <= 50:
				file1.write(f1_lines[x])
				file2.write(f2_lines[x])

if __name__ == "__main__":
	lang1 = "en"
	lang2 = "de"

	shrink(f"train.{lang1}", f"train.{lang2}")
	shrink(f"valid.{lang1}", f"valid.{lang2}")
	shrink(f"test.{lang1}", f"test.{lang2}")