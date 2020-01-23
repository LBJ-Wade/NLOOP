from gensim.similarities import MatrixSimilarity, SparseMatrixSimilarity

class Similarity:
    """
    This class provide methods for calculating similarity Matrix and its
    associated index using gensim's MatrixSimilarity (default) and SparseMatrixSimilarity

    Parameters
    ----------
    corpus: list of tuples
        by default, this is the corpus_tfidf, but can be replaced by any other
        list of tuples (word_index, weight)
    num_features: int
        number of features to consider for document vectors

    """
    def __init__(self,
                 corpus,
                 num_features,
                 ):
        self.corpus = corpus
        self.num_features = num_features

        print(self.num_features)

    def index(self, corpus, mode="MatrixSimilarity"):
        if mode == "MatrixSimilarity":
            self._index = MatrixSimilarity(self.corpus, num_features=self.num_features)
        elif mode == "SparseMatrixSimilarity":
            self._index = SparseMatrixSimilarity(self.corpus, num_features=self.num_features)
        else:
            raise TypeError("mode has to be either MatrixSimilarity or SparseMatrixSimilarity")

        return self._index[corpus]

    @property
    def matrix(self):
        return self.index(self.corpus)

