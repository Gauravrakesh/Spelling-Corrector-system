from ngram import NGramBase
from config import *
import numpy as np
import pandas as pd

class NoSmoothing(NGramBase):

    def __init__(self):

        super(NoSmoothing, self).__init__()
        self.update_config(no_smoothing)
        
        
    def get_probability(self, context, word):
        count_context = sum(self.ngram_counts[context].values())
        count_word = self.ngram_counts[context][word]
        if count_context == 0:
            return 0
        return count_word / count_context

class AddK(NGramBase):

    def __init__(self, k = 1.0):
        super().__init__()
        self.k = k
        

        self.update_config(add_k)
        
    def get_probability(self, context, word):
        count_context = sum(self.ngram_counts[context].values()) + self.k * len(self.vocab)
        count_word = self.ngram_counts[context][word] + self.k
        return count_word / count_context
        

class StupidBackoff(NGramBase):

    def __init__(self, alpha=0.4):
        super().__init__()
        self.alpha = alpha
        self.update_config(stupid_backoff)
    def get_probability(self, context, word):
        if context in self.ngram_counts and word in self.ngram_counts[context]:
            return self.ngram_counts[context][word] / sum(self.ngram_counts[context].values())
        elif len(context) > 1:
            return self.alpha * self.get_probability(context[1:], word)
        else:
            return 1 / len(self.vocab)

class GoodTuring(NGramBase):

    def __init__(self):
        super().__init__()
        self.update_config(good_turing)
    
    def get_probability(self, context, word):
        count_word = self.ngram_counts[context][word]
        Nc = sum(1 for c in self.ngram_counts[context].values() if c == count_word)
        Nc_next = sum(1 for c in self.ngram_counts[context].values() if c == count_word + 1)
        if Nc == 0:
            return 1 / len(self.vocab)
        adjusted_count = (count_word + 1) * (Nc_next / Nc) if Nc_next > 0 else count_word
        count_context = sum(self.ngram_counts[context].values())
        return adjusted_count / count_context if count_context > 0 else 1 / len(self.vocab)

class Interpolation(NGramBase):

    def __init__(self, lambdas=[0.4, 0.35, 0.25]):
        super().__init__()
        self.lambdas = lambdas
        self.update_config(interpolation)
        
    def get_probability(self, context, word):
        p_ngram = self.ngram_counts[context][word] / sum(self.ngram_counts[context].values()) if context in self.ngram_counts else 0
        p_bigram = self.get_probability(context[1:], word) if len(context) > 1 else 0
        p_unigram = self.ngram_counts[()][word] / self.total_ngrams if word in self.ngram_counts[()] else 1 / len(self.vocab)
        return self.lambdas[0] * p_ngram + self.lambdas[1] * p_bigram + self.lambdas[2] * p_unigram

class KneserNey(NGramBase):

    def __init__(self, discount=0.75):
        super().__init__()
        self.discount=discount
        self.update_config(kneser_ney)

    def get_probability(self, context, word):
        count_context = sum(self.ngram_counts[context].values())
        count_word = self.ngram_counts[context][word]
        unique_contexts = len(set(prev for prev in self.ngram_counts if word in self.ngram_counts[prev]))
        lambda_context = (self.discount / count_context) * unique_contexts if count_context > 0 else 1
        p_continuation = unique_contexts / self.total_ngrams
        p_kn = max(count_word - self.discount, 0) / count_context + lambda_context * p_continuation
        return p_kn if count_context > 0 else p_continuation

if __name__=="__main__":
    ns = NoSmoothing()
    ns.method_name()
