# Manipulating GAN model for Later Full Implementation

## Part of CS1678 Final Project

---

## Use the following instructions for execution

### Data presentation

First, clone the repository and change directory into it

```
git clone https://github.com/mgwildermuth/Adversarial-NMT.git
cd Adversarial-NMT/
ls
```

Then, we need to process the data. The data we'll be downloading is the iwslt14 parallel dataset for de-en translation and preparing it for use by the model. All of the data preprocessing is done in the `preprocess/` folder.

The downloading and tokenizing of this iwslt data is done by the follwing script

```
cd preprocess
bash prepare-iwslt14.sh
```

Now that the tokenizing has finished, we then need to prepare a dictionary of words for our model. The model understands a dictionary of the following format

```
<symbol0> <count0>
<symbol1> <count1>
...
```

To get this dictionary of the x-most common words from our dataset, we'll use the following script obtained from [this repository](https://github.com/lisa-groundhog/GroundHog/tree/master/experiments/nmt). More information on data preparation can be found there as well. The following python script uses that tokenized data to turn it into a dictionary of the form we want, just pickled instead. This is the outupt of the `-d` flag. The `-v` flag tells the script how much entries to include in the dictionary (the x-most common words). The `-p` flag holds the path where the data can be found. 

```
python preprocess/preprocess.py -d vocab.en.pkl -v 30000 -b binarized_text.en.pkl -p iwslt14.tokenized.de-en/*.en
python preprocess/preprocess.py -d vocab.de.pkl -v 30000 -b binarized_text.de.pkl -p iwslt14.tokenized.de-en/*.de
```

After the pickled dictionary is made, then we must use the following script to convert the pickled dictionary into the text version that the dataloader can understand. s

```
python pickle-to-dict.py
```

All of the following steps can run by the following script also included in `preprocess/`

```
cd preprocess/
bash make-default-data.sh
cd ..
```

### Running the model

The model code can then be run with the following line. The gpuid line is necessary for the execution of the program, or else it will crash or just run on the cpu.

```
python joint_train.py --data data/iwslt14.tokenized.de-en/  --learning_rate 1e-3 --joint-batch-size 32 --clip-norm 1.0 --max-epoch 1 --epochs 10 --gpuid 0
```