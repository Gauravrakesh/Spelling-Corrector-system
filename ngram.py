import numpy as np
import pandas as pd
from typing import List
import re
from collections import defaultdict

# config.py

class NGramBase:
    def __init__(self, n: int = 2, lowercase: bool = True, remove_punctuation: bool = True):
        """
        Initialize basic n-gram configuration.
        :param n: The order of the n-gram (e.g., 2 for bigram, 3 for trigram).
        :param lowercase: Whether to convert text to lowercase.
        :param remove_punctuation: Whether to remove punctuation from text.
        """
        self.current_config = {}
        
        # change code beyond this point
        #
        self.n = n
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.ngram_counts = defaultdict(int)
        self.vocab = set()
        self.total_ngrams = 0
        

    def method_name(self) -> str:

        # return f"Method Name: {self.current_config['method_name']}"
        # return f"NGram Model: {self.n}-gram"
        return f"Method Name: {self.current_config['method_name']}"


    def fit(self, data: List[List[str]]) -> None:
        """
        Fit the n-gram model to the data.
        :param data: The input data. Each sentence is a list of tokens.
        """
        
        for sentence in data:
            sentence = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i in range(len(sentence) - self.n + 1):
                context = tuple(sentence[i:i + self.n - 1])
                word = sentence[i + self.n - 1]
                self.ngram_counts[context][word] += 1
                self.vocab.add(word)
                self.total_ngrams += 1
 

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        :param text: The input text.
        :return: The list of tokens.
        """
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        if self.lowercase:
            text = text.lower()
        return text.split()

        # raise NotImplementedError

    def prepare_data_for_fitting(self, data: List[str], use_fixed = False) -> List[List[str]]:
        """
        Prepare data for fitting.
        :param data: The input data.
        :return: The prepared data.
        """
        processed = []
        if not use_fixed:
            for text in data:
                processed.append(self.tokenize(self.preprocess(text)))
        else:
            for text in data:
                processed.append(self.fixed_tokenize(self.fixed_preprocess(text)))

        return processed

    def update_config(self, config) -> None:
        """
        Override the current configuration. You can use this method to update
        the config if required
        :param config: The new configuration.
        """
        self.current_config = config

    def preprocess(self, text: str) -> str:
        """
        Preprocess text before n-gram extraction.
        :param text: The input text.
        :return: The preprocessed text.
        """
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        if self.lowercase:
            text = text.lower()
        return text
        # raise NotImplementedError

    def fixed_preprocess(self, text: str) -> str:
        """
        Removes punctuation and converts text to lowercase.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def fixed_tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the input text by splitting at spaces.
        """
        return text.split()

    def perplexity(self, text: str) -> float:
        """
        Compute the perplexity of the model given the text.
        :param text: The input text.
        :return: The perplexity of the model.
        """
        tokens = self.tokenize(text)
        tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_prob = 0
        total_ngrams = 0
        for i in range(len(tokens) - self.n + 1):
            context = tuple(tokens[i:i + self.n - 1])
            word = tokens[i + self.n - 1]
            count_context = sum(self.ngram_counts[context].values())
            count_word = self.ngram_counts[context][word]
            
            if count_context == 0 or count_word == 0:
                return float('inf')
            
            prob = count_word / count_context
            log_prob += -np.log2(prob)
            total_ngrams += 1
            
        return 2 ** (log_prob / total_ngrams) if total_ngrams > 0 else float('inf')
        # raise NotImplementedError

if __name__ == "__main__":
    tester_ngram = NGramBase()
    test_sentence = "This, is a ;test sentence."
