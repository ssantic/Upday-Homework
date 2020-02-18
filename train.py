#!/usr/bin/env python

"""
Author: Srdjan Santic
Date: 18-Feb-2020

This script is used to preprocess text data provided by upday, and to train
a Random Forest classifier. It does a series of preprocessing and feature
engineering steps, and trains a model.

The script also outputs the model as a .pickle file, and also exports a .tsv
file with the test data (features and labels) for evaluating the accuracy of
the model.

The correct usage of the script is:

$ python train.py data_file_name.tsv

"""


import argparse
import re
import pickle
import spacy
import unicodedata
import pandas as pd
import numpy as np
from contractions import CONTRACTION_MAP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

# Prepare the argument parser
parser = argparse.ArgumentParser(description='Script for training a Random Forest classifier.')
parser.add_argument('data_file', type=str, help='Input TSV data file.')
args = parser.parse_args()

# Prepare the functionality of NLP libraries for use
nlp = spacy.load('en', parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

# Load in the original dataset
print("Loading dataset...")
df = pd.read_csv(args.data_file, sep="\t")


# Function for removal of accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# Function for expanding contractions
def expand_contractions(text):

    contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)\
                        if CONTRACTION_MAP.get(match)\
                        else CONTRACTION_MAP.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)

    return expanded_text


# Define function for removing special characters
def remove_special_characters(text, remove_digits=True):
    text.strip()
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text


# Define function for removing stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Define function for text lemmatization
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# Define function for normalizing the text
def normalize_corpus(corpus, contraction_expansion=True,
                     accented_char_removal=True, stopword_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     remove_digits=True):

    normalized_corpus = []
    # Normalize each document in the corpus
    for doc in corpus:

        # Remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # Expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # Remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)

        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # Remove special characters and/or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)

        # Remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus


# Combine the title and the article text
df['full_text'] = df['text'].map(str) + '. ' + df['title']

# Pre-process text and store the same in a new column called "clean text"
print("Normalizing corpus...")
df['clean_text'] = normalize_corpus(df['full_text'])
norm_corpus = list(df['clean_text'])

# Prepare the data frame for modeling
modeling_df = df.drop(['title', 'text', 'url', 'full_text'], axis=1)


# Define function to directly compute the TF-IDF-based feature vectors for documents from the raw documents
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# Split the data into training and testing set
print("Splitting data into train and test...")
X = modeling_df['clean_text']
Y = modeling_df['category']

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

train_corpus, test_corpus = train_X.values.astype('U'), test_X.values.astype('U')
train_labels, test_labels = train_Y, test_Y

# Build the TF-IDF features
print("Constructing TF-IDF features...")
feature_set = []
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)
feature_set.append(('TF-IDF Features', tfidf_train_features, tfidf_test_features))
feature_names = tfidf_vectorizer.get_feature_names()
features = tfidf_train_features.todense()
features_tfidf = np.round(features, 2)

# Fit the Random Forest classifier
print("Training the Random Forest classifier...")
rfc = RandomForestClassifier(random_state=176, n_estimators=100).fit(features, train_labels)

# Export the model as a pickle file
print("Exporting Random Forest classifier...")
pickle.dump(rfc, open('rfc.pickle', 'wb'))

# Prepare and export the test set
test_df = pd.DataFrame(features_tfidf, columns=feature_names)
print("Exporting test set...")
test_df.to_csv('test_data.tsv', sep='\t', index=False)

# Finish and exit the script
print("Done!")
