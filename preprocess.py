import re
from nltk.corpus import stopwords
import pandas as pd

from utils import read_data, save_data

def preprocess_text(text: str, num: int, contraction_mapping: dict):
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

def preprocess_data(cfg, save: bool = True):
    df = read_data(cfg.data.path)
    cleaned_text = []

    for t in df[cfg.data.subset]:
        cleaned_text.append(preprocess_text(t, 0, cfg.preprocess.contraction_mapping))

    text_word_count = []

    for i in cleaned_text:
        text_word_count.append(len(str(i).split()))
    
    new_data = pd.DataFrame({"Rating": df["Rating"], "Text": cleaned_text, "Word Count": text_word_count})
    
    if save:
        save_data(new_data, f"{cfg.data.path}-preprocessed.csv")