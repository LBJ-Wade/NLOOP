"""
This module contains clsases for Topic Modeling
"""

from gensim.models import HdpModel, LdaMulticore, CoherenceModel


class TopicModeling:

    def coherence_score(self, coherence="c_v"):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=self.tokens,
                                         dictionary=self.dictionary,
                                         coherence=coherence)

        coherence_score = coherence_model.get_coherence()

        return coherence_score


class LDA(TopicModeling):

    def __init__(self, corpus, dictionary, tokens):
        self.corpus = corpus
        self.dictionary = dictionary
        self.tokens = tokens

    def run(self,
            num_topics=20,
            alpha='symmetric',
            eta=None,
            random_state=0,
            verbose=True,
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
            random_state=None,
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

        print("Done!\nCheckout lda.model")

