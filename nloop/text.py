"""
base module for Natural Language [Object-Oriented] Processing
"""
__author__ = ["Siavash Yasini", "Amin Oji"]


import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import spacy

import wordcloud

import gensim
from gensim.corpora import Dictionary
from gensim.models import Phrases, TfidfModel

from nltk import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#TODO: add the equivalent for terminal
from IPython.core.display import HTML, display
import pytextrank
from tqdm.auto import tqdm

import logging
logger = logging.getLogger("Log Message")
logger.setLevel(logging.INFO)

from nloop.lib.topicmodeling import LDA, HDP
from nloop.lib.similarity import Similarity


import re

#########################################################
#                  Text Object
#########################################################

class Docs:

    def __init__(self, text, docs):
        self._docs = docs
        self._keywords = Keywords(text._keywords)

    def __getitem__(self, item):
        return self._docs[item]

    def __repr__(self):
        return repr(self._docs)

    @ property
    def keywords(self):
        return self._keywords


class Keywords:

    def __init__(self, keywords):

        self._keywords = keywords
        self._ranks = [[kw.rank for kw in kws] for kws in self._keywords]
        self._counts = [[kw.count for kw in kws] for kws in self._keywords]
        self._chunks = [[kw.chunks for kw in kws] for kws in self._keywords]


    def __getitem__(self, item):
        return self._keywords[item]

    def __repr__(self):
        return repr(self._keywords)

    @property
    def ranks(self):
        return self._ranks

    @property
    def counts(self):
        return self._counts

    @property
    def chunks(self):
        return self._chunks



class Text:

    nlp = spacy.load("en")


    #nlp = spacy.load("en", disable=["parser", "ner"])

    # Full list here: https://spacy.io/api/annotation
    remove_pos = ['ADP',
                  'ADV',
                  'AUX',
                  'CONJ',
                  'SCONJ',
                  'INTJ',
                  'DET',
                  'PART',
                  'PUNCT',
                  'SYM',
                  'SPACE',
                  'NUM',
                  'X',
                  ]

    tags_re = re.compile(r'<[^>]+>')

    def __init__(self,
                 docs,  # normally a list of lists:
                 fast=False,
                 remove_html_tags=True,
                 lemmatize=True,
                 phrases=True,
                 ):

        """
        Class for loading, processing, and modeling text data

        Parameters
        ----------
        docs: list
            Input raw data.
            Can be a list of docs [ doc1, doc2, ... ] or a pandas dataframe column.
            Each doc is a string.

        fast: bool
            If True, "parser" and "ner" are removed from the nlp pipeline

        remove_html_tags: bool
            If True, <html tags> will be removed before processing the text

        lemmatize: bool
            If True, the processed tokens will be lemmatized

        phrases: bool
            If True, bigrams and trigrams are extracted using gensim.models.Phrases

        """

        self.nlp = Text.nlp

        # disable parser and ner for a fast render
        if fast:
            self.nlp = spacy.load("en", disable=["parser", "ner"])

        # otherwise add pytextrank to the pipeline
        else:
            tr = pytextrank.TextRank()
            # Text.nlp = spacy.load("en")
            self.nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
            self._keywords = []

        print(f"nlp.pipe_names = {self.nlp.pipe_names}")

        # read in the raw documents
        self.raw_docs = docs
        self.n_docs = len(self.raw_docs)

        if remove_html_tags:
            self.raw_docs = self.remove_tags()

        # FIXME: loop has to be True because of bug in spacy pytextrank
        self._docs = Docs(text=self,
                          docs=self._nlp_docs(loop=True))

        #self._raw_tokens = self.get_raw_tokens()

        # tokenize and process
        self._tokens = self.process_tokens(lemmatize=lemmatize, phrases=phrases)

        if "parser" in self.nlp.pipe_names:
            pass
            #self._sentences = self.get_sentences()

        #self._token_counter = self._counter()
        self._clean_docs = self.get_clean_docs()
        self._dictionary = Dictionary(self.tokens)
        self._corpus_bow = self.get_corpus_bow()
        self._corpus_tfidf = self.get_corpus_tfidf()


        self.lda = LDA(corpus=self.corpus_tfidf,
                       dictionary=self.dictionary,
                       tokens=self.tokens)

        self.hdp = HDP(corpus=self.corpus_tfidf,
                       dictionary=self.dictionary,
                       tokens=self.tokens)

        self.similarity = Similarity(corpus=self.corpus_tfidf,
                                     num_features=len(self.dictionary))

    # ------------------------
    #       properties
    # ------------------------

    @property
    def docs(self):
        return self._docs


    @property
    def clean_docs(self):
        return self._clean_docs

    @property
    def raw_tokens(self):
        #self._raw_tokens = list(self._tokenize())
        return self.get_raw_tokens()

    @property
    def tokens(self):
        return self._tokens

    @property
    def token_ids(self):
        #try:
         #   self._token_ids
        #except AttributeError:
        self._token_ids = [[self.dictionary.token2id[token] for token in doc] for doc in
                           self.tokens]
        return self._token_ids
    @property
    def token_counter(self):
        return self._token_counter()

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def corpus_bow(self):
        return self._corpus_bow

    @property
    def corpus_tfidf(self):
        return self._corpus_tfidf
    # ------------------------
    #         methods
    # ------------------------

    def remove_tags(self):
        return [Text.tags_re.sub(" ", doc) for doc in tqdm(self.raw_docs,
                total=self.n_docs,  desc="Removing HTML tags")]


    def _nlp_docs(self, loop=True):

        if not loop:
            docs = list(tqdm(self.nlp.pipe(self.raw_docs), total=self.n_docs,
                               desc="Passing docs through nlp.pipe"))

        else:
            #self._keywords = []
            docs = []
            for doc in tqdm(self.nlp.pipe(self.raw_docs),
                            total=self.n_docs,
                            desc="Passing docs through nlp.pipe loop"):
                #    d = self.nlp(doc)
                keywords = [kw for kw in doc._.phrases]
                self._keywords.append(keywords)
                docs.append(doc)

        return docs

    def get_raw_tokens(self):

        raw_tokens = [[token for token in doc] for doc in tqdm(self.docs,
                        total=self.n_docs, desc="Extracting raw tokens")]

        return raw_tokens


    def process_tokens(self, lemmatize=True, lower=True, phrases=True):

        tokens = [[token for token in raw_token
                   if (not token.is_stop) and (token.pos_ not in Text.remove_pos)]
                  for raw_token in tqdm(self.docs, total=self.n_docs, desc="Processing tokens")]

        if lemmatize:
            tokens = [[token.lemma_ for token in doc] for doc in tokens]
        else:
            tokens = [[token.text for token in doc] for doc in tokens]

        if lower:
            tokens = [[token.lower() for token in doc] for doc in tokens]

        if phrases:
            bigrams = Phrases(tokens, delimiter=b"_", min_count=2)
            trigrams = Phrases(bigrams[tokens], delimiter=b"_", min_count=2)

            # extract bigrams and trigrams

            tokens = [bigrams[doc] for doc in tokens]
            tokens = [trigrams[doc] for doc in tokens]

        return tokens
        # def _tokenize(self):
        #     """generate a list of tokens of each document"""
        #     for doc in self.raw_docs:
        #         doc = doc.replace("\n", " ")
        #         yield gensim.utils.simple_preprocess(doc, deacc=True)
        #
        #
        # # TODO: write another process function using sklearn?
        # def process(self,
        #             remove_stops=True,
        #             make_bigrams=True,
        #             make_trigrams=True,
        #             lemmatize=True,
        #             stem=True,
        #             ):
        #     """Process text.raw_docs using gensim:
        #     (1) remove_pos stopwords, (2) find bigrams and trigrams, (3) lemmatize, and (4) stem"""
        #
        #     # initialize a token generator
        #     Docs = self._tokenize()
        #
        #     print("Removing stopwords...")
        #     if remove_stops:
        #         # Remove stopworsds
        #         Docs = [[word for word in doc if word not in self.stops and not word.isdigit()] for
        #                 doc in tqdm(Docs, total=self.n_docs)]
        #
        #     # TODO: modify this so Docs can be passed as a generator
        #     if make_bigrams:
        #         bigrams = Phrases(Docs, delimiter=b" ", min_count=2)
        #
        #     if make_trigrams:
        #         trigrams = Phrases(bigrams[Docs], delimiter=b" ", min_count=2)
        #
        #
        #     # extract bigrams and trigrams
        #     if make_bigrams:
        #         print("Finding bigrams...")
        #         Docs = [bigrams[doc] for doc in tqdm(Docs, total=self.n_docs)]
        #
        #     if make_trigrams:
        #         #FIXME: Is this actually returning the correct trigrams?
        #         # or is it returning 4-grams?
        #         print("Finding trigrams...")
        #         Docs = [trigrams[doc] for doc in tqdm(Docs, total=self.n_docs)]
        #
        #
        #     if lemmatize:
        #         # lemmatize the n-grams
        #         print("Lemmatizing nouns...")
        #         Docs = [[self.lemmatizer.lemmatize(word, pos="n") for word in doc]
        #                 for doc in tqdm(Docs, total=self.n_docs)]
        #         print("Lemmatizing verbs...")
        #         Docs = [[self.lemmatizer.lemmatize(word, pos="v") for word in doc]
        #                 for doc in tqdm(Docs, total=self.n_docs)]
        #
        #     if stem:
        #         # stem the n-grams
        #         print("Stemming...")
        #         Docs = [[self.stemmer.stem(word) for word in doc]
        #                 for doc in tqdm(Docs, total=self.n_docs)]
        #
        #         print("Done!")
        #     return Docs

    def get_sentences(self):
        """Return a list of spacy sentences"""

        sents = [[sent for sent in doc.sents] for doc in self.docs]

        return list(sents)

    def _token_counter(self):
        """Return the counts of all tokens"""
        return Counter([word for doc in self.tokens for word in doc])

    def _token_frequency(self):
        """Returns the frequency of all tokens"""
        pass

    def show_wordcloud(self, figsize=(10, 10), dpi=100):
        """Plot the processed tokens wordcloud"""

        WC_docs = [" ".join(doc) for doc in self.tokens]
        WC_docs = " ".join(WC_docs)

        WC = wordcloud.WordCloud(background_color='white', scale=2).generate(WC_docs)

        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(WC)
        plt.axis("off")

    def filter_extremes(self,
                        no_below=0,  # FIXME: find the optimal cutoff
                        no_above=1,
                        keep_n=None,
                        keep_tokens=None,
                        inplace=True,
                       ):
        """Construct the tokens dictionary using gensim.corpora.Dictionary

        After constructing the dictionary, this runs dictionary.filter_extremes with the
        following parameters:

        Parameters
        ----------
        no_below : int, optional
            Keep tokens which are contained in at least `no_below` documents (absolute number).
        no_above : float, optional
            Keep tokens which are contained in no more than `no_above` documents
            (fraction of total corpus size, not an absolute number).
        keep_n : int, optional
            Keep only the first `keep_n` most frequent tokens.
        keep_tokens : iterable of str
            Iterable of tokens that **must** stay in dictionary after filtering."""

        dictionary = deepcopy(self.dictionary)
        dictionary.filter_extremes(no_below=no_below,
                                   no_above=no_above,
                                   keep_n=keep_n,
                                   keep_tokens=keep_tokens)

        filtered_tokens = [[token for token in doc if token in dictionary.token2id]
                           for doc in self.tokens]
        if inplace:
            self._tokens = filtered_tokens
            self._clean_docs = self.get_clean_docs()
        else:
            return filtered_tokens

    def bow_transformer(self, doc):
        """transforms the input doc to bag-of-words according to the corpus bow"""
        doc_bow = self.dictionary.doc2bow(doc.split(" "))

        return doc_bow

    def get_corpus_bow(self):
        """Construct the corpus bag of words

        This uses gensim.corpora.dictionary.doc2bow
        """
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.tokens]
        return corpus_bow

    def bow2tfidf(self, bow):

        return TfidfModel(self.corpus_bow)[bow]

    def tfidf_transformer(self, doc):
        """transform the input doc to Tfidf according to the corpus tfidf model"""
        doc_bow = self.bow_transformer(doc)
        doc_tfidf = self.bow2tfidf(doc_bow)
        return doc_tfidf

    def get_corpus_tfidf(self):
        """Construct the corpus Tfidf (term frequency inverse document frequency

        This uses gensim.models.TfidfModel
        """

        bow2tfidf = TfidfModel(self.corpus_bow)
        corpus_tfidf = [bow2tfidf[bow] for bow in self.corpus_bow]
        return corpus_tfidf

    def get_vocab(self):
        """Return the corpus vocabulary by calling all the dictionary values"""

        vocab = [self.dictionary[key] for key in self.dictionary.keys()]
        return vocab

    def get_corpus_id(self):
        """Return the corpus token indices by calling the dictionary.doc2idx"""

        corpus_id = [self.dictionary.doc2idx(doc) for doc in self.tokens]
        return corpus_id

    def get_clean_docs(self):

        return [" ".join(token) for token in tqdm(self.tokens,
                         total=self.n_docs, desc="Putting clean tokens back together")]

    # TODO: fix bug with bigrams and trigrams
    # TODO: add option for printing in terminal
    def search_for_token(self, token, color="red", font_size=5):
        """search the corpus for the given token and highlight/return the documents in which the
        token occurs"""

        token = self.lemmatizer.lemmatize(token, pos="n")
        token = self.lemmatizer.lemmatize(token, pos="v")
        token = self.stemmer.stem(token)

        print(f"Looking for '{token}' in all the raw_docs...")

        # docs_with_token = []
        idx_with_token = []
        Docs = self.tokens

        for idx, doc in enumerate(Docs):
            if token in doc:
                # TODO: debug this kososher
                doc_text = " ".join(doc)

                # add the document index to the list
                print(f"\nDocument # {idx}:")
                idx_with_token.append(idx)

                doc_text = doc_text.replace(token,
                                            f"<b><span style='color:{color}'><font size"
                                            f"={font_size}>{token}</font></span></b>")
                display(HTML(doc_text))


        if len(idx_with_token) == 0:
            print("Nothing found!")

        return idx_with_token









