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
from nloop.lib.utils import lazy_property

import re

#########################################################
#                  Text Object
#########################################################

class Text:

    # Full list here: https://spacy.io/api/annotation
    # remove_pos = ['ADP',
    #               'ADV',
    #               'AUX',
    #               'CONJ',
    #               'SCONJ',
    #               'INTJ',
    #               'DET',
    #               'PART',
    #               'PUNCT',
    #               'SYM',
    #               'SPACE',
    #               'NUM',
    #               'X',
    #               ]

    tags_re = re.compile(r'<[^>]+>')


    #spacy_model = "en"  # switch to "en_core_web_lg" for better results

    def __init__(self,
                 docs,  # normally a list of lists:
                 fast=True,
                 keep_pos=["ADJ", "NOUN", "PROPN", "VERB"],  # https://spacy.io/api/annotation
                 remove_html_tags=True,
                 lemmatize=True,
                 phrases=True,
                 spacy_model="en",  # switch to "en_core_web_lg" for better results
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

        self.nlp = spacy.load(spacy_model)
        print(f'spacy_model: "{spacy_model}"')
        self.keep_pos = keep_pos
        print(f"Only keeping: {self.keep_pos}")

        # disable parser and ner for a fast render
        if fast:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])

        # otherwise add pytextrank to the pipeline
        # this will enable keyword extraction and sentence parser
        else:
            tr = pytextrank.TextRank()
            self.nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)

        # show me whatcha doin' spacy
        print(f"nlp.pipe_names = {self.nlp.pipe_names}")

        # read in the raw documents
        self.raw_docs = docs
        self.n_docs = len(self.raw_docs)

        if remove_html_tags:
            self.raw_docs = self.remove_tags()

        # keywords will be set in self._nlp_docs()
        self._keywords = None

        # FIXME: for extracting keywords set loop=True because of bug in spacy pytextrank
        self._docs = Docs(docs=self._nlp_docs(loop=not fast))

        # tokenize and process
        self._tokens = self.process_tokens(lemmatize=lemmatize, phrases=phrases)

        # self.lda = LDA(corpus=self.corpus_tfidf,
        #                dictionary=self.dictionary,
        #                tokens=self.tokens)
        #
        # self.hdp = HDP(corpus=self.corpus_tfidf,
        #                dictionary=self.dictionary,
        #                tokens=self.tokens)
        #
        self.similarity = Similarity(corpus=self.corpus_tfidf,
                                     num_features=len(self.dictionary))

    # ------------------------
    #       properties
    # ------------------------

    @property
    def docs(self):
        return self._docs

    @property
    def tokens(self):
        return self._tokens

    @lazy_property
    def clean_docs(self):
        return self.get_clean_docs()

    @lazy_property
    def raw_tokens(self):
        #self._raw_tokens = list(self._tokenize())
        return self.get_raw_tokens()

    @property
    def token_ids(self):
        self._token_ids = [[self.dictionary.token2id[token] for token in doc] for doc in
                           self.tokens]
        return self._token_ids

    @property
    def keywords(self):
        return self._keywords

    @property
    def token_counter(self):
        return self._token_counter()

    @lazy_property
    def dictionary(self):
        return Dictionary(self.tokens)

    @lazy_property
    def corpus_bow(self):
        return self.get_corpus_bow()

    @lazy_property
    def corpus_tfidf(self):
        return self.get_corpus_tfidf()

    @lazy_property
    def sentences(self):
        if "parser" in self.nlp.pipe_names:
            return Sentences(self.get_sentences())
        else:
            raise AttributeError("add 'parser' to the nlp.pipes or set fast=False to get "
                                 "sentences.")

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
            # This is currently necessary for extracting keywords because of a bug in spacy
            #print("looping")

            docs = []
            self._keywords = []

            for doc in tqdm(self.nlp.pipe(self.raw_docs),
                            total=self.n_docs,
                            desc="Passing docs through nlp.pipe loop"):

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
                   # TODO: Add like_num option?
                  if (token.pos_ in self.keep_pos) and  (not token.is_stop) and (token.is_alpha)]
                  for raw_token in tqdm(self.docs, total=self.n_docs, desc="Processing tokens")]

        if lemmatize:
            tokens = [[token.lemma_ for token in doc] for doc in tokens]
        else:
            tokens = [[token.text for token in doc] for doc in tokens]

        if lower:
            tokens = [[token.lower() for token in doc] for doc in tokens]

        if phrases:
            # TODO: Add n-gram pattern matching with spacy
            bigrams = Phrases(tokens, delimiter=b"_", min_count=2)
            trigrams = Phrases(bigrams[tokens], delimiter=b"_", min_count=2)

            # extract bigrams and trigrams
            tokens = [bigrams[doc] for doc in tokens]
            tokens = [trigrams[doc] for doc in tokens]

        return tokens


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
            #TODO add dependencies between tokens and raw_tokens and clean_docs
            self._dictionary = dictionary
            self._tokens = filtered_tokens
            self._clean_docs = self.get_clean_docs()
        else:
            return filtered_tokens

    def bow_transformer(self, docs):
        """transforms the input doc to bag-of-words according to the corpus bow"""

        single_doc = False
        if type(docs) is str:
            docs = [docs]
            single_doc = True

        docs_bow = list([self.dictionary.doc2bow(doc.split(" ")) for doc in docs])

        if single_doc:
            docs_bow = docs_bow[0]

        return docs_bow

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
    def search_for_token(self, token, color="#FFFF00", font_size=5, exact=True):
        """search the corpus for the given token and highlight/return the documents in which the
        token occurs"""

        assert exact, "non-exact search is not available"

        print(f"Looking for '{token}' in all the raw_docs...")

        # docs_with_token = []
        idx_with_token = []
        #docs = self.docs

        for idx, doc in enumerate(self.docs):
            if token in doc.text:
                # TODO: debug this kososher
                #doc_text = " ".join(doc.text)
                doc_text = doc.text
                # add the document index to the list
                print(f"\nDocument # {idx}:")
                idx_with_token.append(idx)

                doc_text = doc_text.replace(token,
                                            f"<b><span style='background-color:{color}'><font size"
                                            f"={font_size}>{token}</font></span></b>")

                # doc_text = doc_text.replace(token,
                #                             f"<b><span style='background-color:{color};fontsize"
                #                             f":{font_size}'>{token}</span></b>")
                display(HTML(doc_text))


        if len(idx_with_token) == 0:
            print("Nothing found!")

        return idx_with_token



class Docs:

    def __init__(self, docs):
        self._docs = docs

    def __getitem__(self, item):
        return self._docs[item]

    def __repr__(self):
        return repr(self._docs)

    @property
    def texts(self):
        return [doc.text for doc in self._docs]

    @property
    def ents(self):
        return [doc.ents for doc in self._docs]

    @property
    def noun_chunks(self):
        return [tuple(doc.noun_chunks) for doc in self._docs]

class Keywords:

    def __init__(self, keywords):

        self._keywords = keywords
        self._texts = [[kw.text for kw in kws] for kws in self._keywords]
        self._ranks = [[kw.rank for kw in kws] for kws in self._keywords]
        self._counts = [[kw.count for kw in kws] for kws in self._keywords]
        self._chunks = [[kw.chunks for kw in kws] for kws in self._keywords]


    def __getitem__(self, item):
        return self._keywords[item]

    def __repr__(self):
        return repr(self._keywords)

    @property
    def texts(self):
        return self._texts

    @property
    def ranks(self):
        return self._ranks

    @property
    def counts(self):
        return self._counts

    @property
    def chunks(self):
        return self._chunks


class Sentences:

    def __init__(self, sentences):

        self._sentences = sentences
        #self._ents = [[kw.ents for kw in kws] for kws in self._ents]
        #self._counts = [[kw.count for kw in kws] for kws in self._keywords]
        #self._chunks = [[kw.chunks for kw in kws] for kws in self._keywords]


    def __getitem__(self, item):
        return self._sentences[item]

    def __repr__(self):
        return repr(self._sentences)

    @lazy_property
    def texts(self):
        return [tuple(sent.text for sent in sents) for sents in self._sentences]

    @lazy_property
    def ents(self):
        return [tuple(sent.ents for sent in sents) for sents in self._sentences]

    @lazy_property
    def noun_chunks(self):
        return [tuple(tuple(sent.noun_chunks) for sent in sents) for sents in self._sentences]

    @lazy_property
    def start(self):
        return [[sent.start for sent in sents] for sents in self._sentences]

    @lazy_property
    def end(self):
        return [[sent.end for sent in sents] for sents in self._sentences]

    @lazy_property
    def start_char(self):
        return [[sent.start_char for sent in sents] for sents in self._sentences]

    @lazy_property
    def end_char(self):
        return [[sent.end_char for sent in sents] for sents in self._sentences]








