#############################################################
# This file contains the code for preprocessing the data.   #
# The preprocessing steps are:                              #
# 1. Converting all the text to lowercase                   #
# 2. Removing all the special characters                    #
# 3. Removing all the stopwords                             #
# 4. Removing all the single characters                     #
# 5. Removing all the single characters from the start      #
# 6. Substituting multiple spaces with single space         #
# 7. Lemmatization                                          #
#############################################################

import re
import nltk
from nltk.corpus import wordnet, stopwords
import pandas as pd
from tqdm import tqdm

from utils import read_config, read_data, save_data

def update_nltk_pkgs() -> None:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(word):
  tag = nltk.pos_tag([word])[0][1][0].upper()
  tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

  return tag_dict.get(tag, wordnet.NOUN)

lemmatizer = nltk.stem.WordNetLemmatizer()

def get_lemmatize(sent):
  return " ".join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)])

def clean_text(text: str, num: int, contraction_mapping: dict):
    stop_words = set(stopwords.words('english'))
    
    newString = text.lower()
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    
    for i in tokens:
        if len(i)>1:
            long_words.append(i)   
    return (" ".join(long_words)).strip()

def preprocess_data(cfg, lemma: bool = True, save: bool = True):
    df = read_data(cfg.data.path)
    cleaned_text = []

    print("Preprocessing data...")
    for i,t in enumerate(tqdm(df[cfg.data.subset])):
        cleaned_text.append(clean_text(t, 0, cfg.preprocess.contraction_mapping))
        if lemma:
            get_lemmatize(cleaned_text[i])

    text_word_count = []

    print("Calculating word count...")
    for _,i in enumerate(tqdm(cleaned_text)):
        text_word_count.append(len(str(i).split()))
    
    new_data = pd.DataFrame({"Rating": df["Rating"], "Text": cleaned_text, "Word Count": text_word_count})
    
    if save:
        save_data(new_data, f"{cfg.data.path[:-4]}-preprocessed.csv")

if __name__=="__main__":
    cfg = read_config()
    update_nltk_pkgs()
    preprocess_data(cfg, cfg.preprocess.lemma, cfg.preprocess.save)