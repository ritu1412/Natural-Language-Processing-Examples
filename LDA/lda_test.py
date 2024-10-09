"""Latent Dirichlet Allocation

Patrick Wang, 2021
"""
from typing import List

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import numpy as np


def lda_gen(vocabulary: List[str], alpha: np.ndarray, beta: np.ndarray, xi: int) -> List[str]:
    Nd = np.random.poisson(xi) #Document length
    if Nd == 0:
        return []
    theta = np.random.dirichlet(alpha)  #Topic distribution
    words = []
    for i in range(Nd):
        z = np.random.choice(len(theta), p=theta) #Topic assignment
        beta_z = beta[z] #Word distribution
        beta_z /= beta_z.sum() #Normalize
        w = np.random.choice(len(vocabulary), p=beta_z) #word assignment
        words.append(vocabulary[w]) #Add word to document
    return words

def test():
    """Test the LDA generator."""
    vocabulary = [
        "bass", "pike", "deep", "tuba", "horn", "catapult",
    ]
    beta = np.array([
        [0.4, 0.4, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.1, 0.0, 0.3, 0.3],
        [0.3, 0.0, 0.2, 0.3, 0.2, 0.0]
    ])
    alpha = np.array([0.2, 0.2, 0.2])
    xi = 50
    documents = [
        lda_gen(vocabulary, alpha, beta, xi)
        for _ in range(100)
    ]

    # Create a corpus from a list of texts
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(text) for text in documents]
    model = LdaModel(
        corpus,
        id2word=dictionary,
        num_topics=3,
    )
    print(model.alpha)
    print(model.show_topics())


if __name__ == "__main__":
    test()
