#!/usr/bin/env bash

#inspried from:
	#https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Translation/Transformer/examples/translation/prepare-iwslt14.sh
	#https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt14.sh 

# run the script that downloads the dataset and tokenizes it
bash prepare-iwslt14.sh


#taken from: #https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt/preprocess

# run the preprocessing scripts (creating lang dictionaries (-v 30000 is the max length of the list of vocab)) on the data
python preprocess/preprocess.py -d vocab.en.pkl -v 30000 -b binarized_text.en.pkl -p iwslt14.tokenized.de-en/*.en
python preprocess/preprocess.py -d vocab.de.pkl -v 30000 -b binarized_text.de.pkl -p iwslt14.tokenized.de-en/*.de

# convert the pickeled vocab.*.pkl to dict.*.txt in the iwslt data folder
python pickle_to_dict.py