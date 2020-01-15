"""
This module contains clsases for Topic Modeling
"""

from gensim.models import HdpModel, LdaMulticore, CoherenceModel
from sklearn.model_selection import ParameterGrid
import pandas as pd
import pdb

class TopicModeling:

    def coherence_score(self, coherence="c_v"):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.tokens,
                                         dictionary=self.dictionary,
                                         coherence=coherence)

        coherence_score = coherence_model.get_coherence()

        return coherence_score

    def grid_search(self, param_dict, eval_func, args=(), kwargs={}):
        """
        Grid search the eval_func using the provided param_dict in every .run()

        Parameters
        ----------
        param_dict: dict
            dictionary of parameters to be explored
        eval_func: function
            funciton to be evaluated at each grid point
        args: tuple
            arguments to be passed to eval_func
        kwargs: dict
            keyword arguments to be passed to eval_func

        Returns
        -------
        grid dataframe

        example:

        >>> parameters = {'num_topics': [1,10, 50],
                          'gamma_threshold': [0.001, 0.01, 0.1]}
        >>> grid_df = text.lda.grid_search(parameters, text.lda.coherence_score)

        >>> print(df)
        gamma_threshold  num_topics  coherence_score
        0    0.001           1         0.247374
        1    0.001          10         0.299208
        2    0.010           1         0.247374
        3    0.010          10         0.311652
        """

        param_grid = list(ParameterGrid(param_dict))
        grid_df = pd.DataFrame(param_grid)

        grid_list = []

        for i, row in grid_df.iterrows():
            print(i, {**row})
            self.run(**row)
            grid_list.append(eval_func(*args, **kwargs))

        grid_df[eval_func.__name__] = grid_list

        return grid_df


class LDA(TopicModeling):

    def __init__(self, corpus, dictionary, tokens):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokens = tokens

    def run(self,
            num_topics=50,
            alpha='symmetric',
            eta=None,
            decay=0.5,
            offset=1.0,
            gamma_threshold=0.001,
            random_state=0,
            # eval_every=10,
            # iterations=50,
            # workers=None,
            # chunksize=2000,
            # passes=1,
            # batch=False,
            # minimum_probability=0.01,
            # minimum_phi_value=0.01,
            # per_word_topics=False,
            verbose=True,
            *args,
            **kwargs):

        self.model = LdaMulticore(corpus=self.corpus,
                                  id2word=self.dictionary,
                                  num_topics=num_topics,
                                  alpha=alpha,
                                  eta=eta,
                                  random_state=random_state,
                                  decay=decay,
                                  offset=offset,
                                  gamma_threshold=gamma_threshold,
                                  *args,
                                  **kwargs)

        if verbose:
            print("Done!\nCheckout lda.model")

    def visualize(self, mds='pcoa'):
        """
        visualize LDA using pyLDAvis

        see: https://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb#topic=8&lambda=1&term=
        paper: https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf

        Parameters
        ----------
        mds: str
            scaling function
            valild options are ['pcoa', 'tnse', mmds']

        Returns
        -------

        """
        import pyLDAvis
        import pyLDAvis.gensim

        print("Make sure you have pyLDAviz imported in the notebook:\n\n"
              "import pyLDAvis\n"
              "pyLDAvis.enable_notebook()\n")

        ldavis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary, mds=mds)
        pyLDAvis.display(ldavis)


        return ldavis

class HDP(TopicModeling):

    def __init__(self, corpus, dictionary, tokens):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokens = tokens

    def run(self,
            kappa=1.0,
            tau=64.0,
            K=15,
            T=150,
            alpha=1,
            gamma=1,
            eta=0.01,
            scale=1.0,
            var_converge=0.0001,
            outputdir=None,
            random_state=0,
            *args,
            **kwargs):

        self.model = HdpModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              kappa=kappa,
                              tau=tau,
                              K=K,
                              T=T,
                              alpha=alpha,
                              gamma=gamma,
                              eta=eta,
                              scale=scale,
                              var_converge=var_converge,
                              outputdir=outputdir,
                              random_state=random_state,
                              *args,
                              **kwargs)

        print("Done!\nCheckout hdp.model")

