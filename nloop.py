"""
base module for Natural Language [Object-Oriented] Processing
"""
__author__ = ["Siavash Yasini", "Amin Oji"]

import os
from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import wordcloud

import gensim
from gensim.corpora import Dictionary
from gensim.models import Phrases, TfidfModel, LdaModel, LdaMulticore, CoherenceModel

from nltk import Counter
from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#TODO: add the equivalent for terminal
from IPython.core.display import HTML, display
from tqdm.auto import tqdm
import logging
logger = logging.getLogger("Log Message")
logger.setLevel(logging.INFO)

#########################################################
#                  Text Object
#########################################################

class Text:

    def __init__(self,
                 data,  # normally a list of lists:
                 column=None,  # in case the input is a dataframe
                 ):
        """
        Class for loading, processing, and modeling text data

        Parameters
        ----------
        data: list or pandas.DataFrame
            Input raw data.
            Can be a list of docs [ doc1, doc2, ... ] or a pandas dataframe.
            Each doc is a string.

        column: str
            Selects a column from the input data as data[column].
            Set to None if the input is a list of documents.
        """


        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stops = stopwords.words("english")

        #TODO: add __slots__
        self.docs = data  # pandas dataframe
        if column:
            self.docs = data[column]
        self.n_docs = len(self.docs)

        # tokenize and process
        self._tokens = self.process()
        self._token_counter = self.counter()
        self._dictionary = self.get_dictionary()
        self._corpus_bow = self.get_corpus_bow()
        self._corpus_tfidf = self.get_corpus_tfidf()

        self.lda = LDA(corpus=self.corpus_tfidf,
                       dictionary=self.dictionary,
                       tokens=self.tokens)

    # ------------------------
    #       properties
    # ------------------------

    @property
    def raw_tokens(self):
        self._raw_tokens = list(self._tokenize())
        return self._raw_tokens

    @property
    def tokens(self):
        return list(self._tokens)

    @property
    def token_counter(self):
        return self._token_counter

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

    def _tokenize(self):
        """generate a list of tokens of each document"""
        for doc in self.docs:
            doc = doc.replace("\n", " ")
            yield gensim.utils.simple_preprocess(doc, deacc=True)


    # TODO: write another process function using sklearn?
    def process(self,
                remove_stops=True,
                make_bigrams=True,
                make_trigrams=True,
                lemmatize=True,
                stem=True,
                ):
        """Process text.docs using gensim:
        (1) remove stopwords, (2) find bigrams and trigrams, (3) lemmatize, and (4) stem"""

        # initialize a token generator
        Docs = self._tokenize()

        print("Removing stopwords...")
        if remove_stops:
            # Remove stopworsds
            Docs = [[word for word in doc if word not in self.stops and not word.isdigit()] for
                    doc in tqdm(Docs, total=self.n_docs)]

        # TODO: modify this so Docs can be passed as a generator
        if make_bigrams:
            bigrams = Phrases(Docs, delimiter=b" ", min_count=2)

        if make_trigrams:
            trigrams = Phrases(bigrams[Docs], delimiter=b" ", min_count=2)


        # extract bigrams and trigrams
        if make_bigrams:
            print("Finding bigrams...")
            Docs = [bigrams[doc] for doc in tqdm(Docs, total=self.n_docs)]

        if make_trigrams:
            #FIXME: Is this actually returning the correct trigrams?
            # or is it returning 4-grams?
            print("Finding trigrams...")
            Docs = [trigrams[doc] for doc in tqdm(Docs, total=self.n_docs)]


        if lemmatize:
            # lemmatize the n-grams
            print("Lemmatizing nouns...")
            Docs = [[self.lemmatizer.lemmatize(word, pos="n") for word in doc]
                    for doc in tqdm(Docs, total=self.n_docs)]
            print("Lemmatizing verbs...")
            Docs = [[self.lemmatizer.lemmatize(word, pos="v") for word in doc]
                    for doc in tqdm(Docs, total=self.n_docs)]

        if stem:
            # stem the n-grams
            print("Stemming...")
            Docs = [[self.stemmer.stem(word) for word in doc]
                    for doc in tqdm(Docs, total=self.n_docs)]

            print("Done!")
        return Docs

    def counter(self):
        """Return a gensim counter of all tokens"""
        return Counter([word for doc in self.tokens for word in doc])

    def show_wordcloud(self, figsize=(10, 10), dpi=100):
        """Plot the processed tokens wordcloud"""

        WC_docs = [" ".join(doc) for doc in self.tokens]
        WC_docs = " ".join(WC_docs)

        WC = wordcloud.WordCloud(background_color='white', scale=2).generate(WC_docs)

        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(WC)
        plt.axis("off")

    def get_dictionary(self,
                       no_below=5,  # FIXME: find the optimal cutoff
                       no_above=1,
                       keep_n=None,
                       keep_tokens=None,
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

        dictionary = Dictionary(self.tokens)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n,
                                   keep_tokens=keep_tokens)

        return dictionary

    def get_corpus_bow(self):
        """Construct the corpus bag of words

        This uses gensim.corpora.dictionary.doc2bow
        """
        corpus_bow = [self.dictionary.doc2bow(doc) for doc in self.tokens]
        return corpus_bow

    def get_corpus_tfidf(self):
        """Construct the corpus Tfidf (term frequency inverse document frequency

        This uses gensim.models.TfidfModel
        """

        tfidfmodel = TfidfModel(self.corpus_bow)
        corpus_tfidf = [tfidfmodel[bow] for bow in self.corpus_bow]
        return corpus_tfidf

    def get_vocab(self):
        """Return the corpus vocabulary by calling all the dictionary values"""

        vocab = [self.dictionary[key] for key in self.dictionary.keys()]
        return vocab

    def get_corpus_id(self):
        """Return the corpus token indices by calling the dictionary.doc2idx"""

        corpus_id = [self.dictionary.doc2idx(doc) for doc in self.tokens]
        return corpus_id

    # TODO: fix bug with bigrams and trigrams
    # TODO: add option for printing in terminal
    def search_for_token(self, token, color="red", font_size=5):
        """search the corpus for the given token and highlight/return the documents in which the
        token occurs"""

        token = self.lemmatizer.lemmatize(token, pos="n")
        token = self.lemmatizer.lemmatize(token, pos="v")
        token = self.stemmer.stem(token)

        print(f"Looking for '{token}' in all the docs...")

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
                                            f"<b><span style='color:{color}'><font size={font_size}>{token}</font></span></b>")
                display(HTML(doc_text))

        if len(idx_with_token) == 0:
            print("Nothing found!")

        return idx_with_token


class LDA:

    def __init__(self, corpus, dictionary, tokens):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokens = tokens

    def run(self,
            num_topics=20,
            alpha='symmetric',
            eta=None,
            random_state=0,
            *args,
            **kwargs):

        self.model = LdaMulticore(corpus=self.corpus,
                                  id2word=self.dictionary,
                                  num_topics=num_topics,
                                  alpha=alpha,
                                  eta=eta,
                                  random_state=random_state,
                                  *args,
                                  **kwargs)

        print("Done!\nCheckout lda.model")

    def coherence_score(self, coherence="c_v"):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.tokens,
                                         dictionary=self.dictionary,
                                         coherence=coherence)

        coherence_score = coherence_model.get_coherence()

        return coherence_score


if __name__ == "__main__":

    data_fname = os.path.join(".", "data", "set=physics:astro-ph-from=2007-01-01-to=2008-01-01.csv")
    data = pd.read_csv(data_fname, index_col=0)

    # only work with abstracts that have more than a 100 citations
    data = data[data["n_citations"] > 100]


    # show word cloud of the corpus
    text = Text(data, column="abstract")
    print(text.n_docs)
    # search for a keyword in the corpus and return the index of documents with the keyword in them
    text.lda.run()
    print(text.lda.model.show_topics(5))
    #text.show_wordcloud(dpi=100)

    # data = data.sample(n=100, random_state=0)