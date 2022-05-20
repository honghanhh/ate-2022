import os
import glob
import argparse
import pandas as pd
import pickle as pkl
from nltk import ngrams                                 
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.lang.nl import Dutch
from sacremoses import MosesTokenizer, MosesDetokenizer

class Preprocessor():
    def __init__(self):
        pass

    def load_text_corpus(self, path):
        """
        Load all text files from folder into a string

        Parameters:
        ------------
        path: directory path that the .txts are located

        Returns:
        ------------
        text_data: the concatenated text from all .txt in directory path
        """
        text_data = ""
        for file in glob.glob(path + "*.txt"):
            with open(file) as f:
                temp_data = f.read()
                text_data = text_data + " " + temp_data
        return text_data

    def preprocess(self, text, lang='en'):
        """
        Split in sentence and tokenize the text

        Parameters:
        ------------
        text: data concatenated from load_text_corpus

        Returns:
        ------------
        sentence_list: list of tuple containing tokens and original sentence
        """

        # sentenize (from spacy)
        sentencizer = Sentencizer()
        if lang == 'fr':
            nlp = French()
        elif lang == 'nl':
            nlp = Dutch()
        else:
            nlp = English()
        nlp.add_pipe(sentencizer)
        doc = nlp(text)

        #tokenize
        sentence_list=[]
        mt = MosesTokenizer(lang=lang)
        for s in doc.sents:
            tokenized_text = mt.tokenize(s, return_str = True)
            sentence_list.append((tokenized_text.split(), s))     #Append tuple of tokens and original sentences
        return sentence_list

    def find_sub_list(self, subl, l):
        """
        Find indices of a sublist in a list 

        Parameters:
        ------------
        sub: sublist
        l: list

        Returns:
        ------------
        results: indices

        """
        results = []
        subllen = len(subl)
        for ind in (i for i,e in enumerate(l) if e == subl[0]):
            if l[ind : ind + subllen] == subl:
                results.append((ind,ind + subllen - 1))
        return results


    def create_training_data(self, sentence_list, df_terms, n = 6, lang='en'):
        """
        Create training data with mapped label as sequence in form of 'B-T', 'T', and 'N'.
        
        Parameters:
        ------------
        sentence_list: sentence_list extracted from preprocess
        df_term: DataFrame containing 2 columns: Term, Labels
        n: n-grams

        Returns:
        ------------
        training_data: A list of tuple containing original text and its corresponding labels
        """
        count=0
        training_data = []
        md = MosesDetokenizer(lang=lang)

        for sen in sentence_list:
            count+=1
            if count%100==0:print(count)

            s = sen[0]  # Take first part of tuple, i.e. the tokens

            tags=["n"]*len(s) # Create label list, with "n" for non-terms, "B-T" for beginning of a term and "T" for the continuation of a term

            for i in range(1,n+1): # 1-gram up to n-gram
                
                n_grams = ngrams(s, i) # Create n-grams of this sentence
                
                if i < len(s):
                    for n_gram in n_grams: # Look if n-grams are in the annotation dataset
                        n_gram_aslist = list(n_gram)
                        n_gram = md.detokenize(n_gram)
                        # context=str(sen[1]).strip()
                        
                        # If yes add an entry to the training data
                        if n_gram.lower() in df_terms.values:
                            # Check where n_gram is in sentence and annotate it 
                            sublist_indices= self.find_sub_list(n_gram_aslist, s)
                            for indices in sublist_indices:
                                for ind in range(indices[0],indices[1]+1):
                                    # If term start
                                    if ind==indices[0]:
                                        tags[ind]="B-T"
                                    # If continuation of a Term
                                    else: 
                                        tags[ind]="T"
            training_data.append((s,tags))          
        return training_data

    def map_data_label(self, gt_path, df_term_path, lang):
        if lang == 'sl':
            df_term = pd.read_csv(df_term_path, header = None,  delimiter="\t", names=["Term", "Label"])
            # df_term = df_term[df_term.Label == 1]
        else:
            df_term = pd.read_csv(df_term_path, delimiter="\t", names=["Term", "Label"]) 
        domain_text = self.load_text_corpus(gt_path)
        domain_s_list = self.preprocess(domain_text, lang) 
        train_data = self.create_training_data(domain_s_list, df_term, 6, lang)  
        return train_data 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Map label from gold standard term list to corpus.')
    parser.add_argument('--corpus_path', type=str, dest='corpus', default="./ACTER/en/corp/texts/annotated/")
    parser.add_argument('--term_path', type=str, dest='ann', default="..ACTER/en/corp/annotations/corp_en_terms_nes.ann")
    parser.add_argument('--output_csv_path', type=str, dest='output', default="./processed_data/ann/corp.pkl")
    parser.add_argument('--language', type=str, dest='lang', default="en")
    args = parser.parse_args()

    preprocessor = Preprocessor()
    data = preprocessor.map_data_label(args.corpus, args.ann, args.lang)
    # data.to_csv()
    with open(args.output, 'wb') as f:
        pkl.dump(data, f)
            