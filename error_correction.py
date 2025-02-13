from smoothing_classes import *
from config import error_correction
import numpy as np
import pandas as pd

class SpellingCorrector:

    def __init__(self):

        self.correction_config = error_correction
        self.internal_ngram_name = self.correction_config['internal_ngram_best_config']['method_name']

        if self.internal_ngram_name == "NO_SMOOTH":
            self.internal_ngram = NoSmoothing()
        elif self.internal_ngram_name == "ADD_K":
            self.internal_ngram = AddK()
        elif self.internal_ngram_name == "STUPID_BACKOFF":
            self.internal_ngram = StupidBackoff()
        elif self.internal_ngram_name == "GOOD_TURING":
            self.internal_ngram = GoodTuring()
        elif self.internal_ngram_name == "INTERPOLATION":
            self.internal_ngram = Interpolation()
        elif self.internal_ngram_name == "KNESER_NEY":
            self.internal_ngram = KneserNey()

        self.internal_ngram.update_config(self.correction_config['internal_ngram_best_config'])
        # self.error_model = defaultdict(lambda: 1e-6)  # Default small probability for unseen typos
        # self.dictionary = set(words.words()) 
        self.vocab = set()

    def fit(self, data: list[str]) -> None:
        """
        Fit the spelling corrector model to the data.
        :param data: The input data.
        """
        
        processed_data = self.internal_ngram.prepare_data_for_fitting(data, use_fixed=True)
        self.internal_ngram.fit(processed_data)
        self.vocab = self.internal_ngram.vocab

    def correct(self, text: List[str]) -> List[str]:
        """
        Correct the input text.
        :param text: The input text.
        :return: The corrected text.
        """
        ## there will be an assertion to check if the output text is of the same
        ## length as the input text
        
        corrected_text = []
        for sentence in text:
            tokens = self.internal_ngram.tokenize(sentence)
            corrected_tokens = []
            for i in range(len(tokens)):
                if i < len(tokens) - 1:
                    context = tuple(tokens[max(0, i - self.internal_ngram.n + 1):i + 1])
                    word = tokens[i + 1]
                    candidates = self.generate_candidates(word)
                    best_candidate = max(candidates, key=lambda x: self.internal_ngram.get_probability(context, x))
                    corrected_tokens.append(best_candidate)
                else:
                    corrected_tokens.append(tokens[i])
            corrected_text.append(" ".join(corrected_tokens))
        return corrected_text

    def generate_candidates(self, word: str) -> List[str]:
        """
        Generate candidate corrections for a given word.
        :param word: The word to generate candidates for.
        :return: A list of candidate corrections.
        """
        candidates = {word}
        letters = 'abcdefghijklmnopqrstuvwxyz'

        # Generate candidates by adding, deleting, replacing, and transposing letters
        for i in range(len(word) + 1):
            # Insertion
            for letter in letters:
                candidates.add(word[:i] + letter + word[i:])
            # Deletion
            if i < len(word):
                candidates.add(word[:i] + word[i+1:])
            # Replacement
            if i < len(word):
                for letter in letters:
                    candidates.add(word[:i] + letter + word[i+1:])
            # Transposition
            if i < len(word) - 1:
                candidates.add(word[:i] + word[i+1] + word[i] + word[i+2:])

        # Filter candidates to only include valid words from the vocabulary
        valid_candidates = [candidate for candidate in candidates if candidate in self.vocab]
        return valid_candidates if valid_candidates else [word]

if __name__ == "__main__":
    corrector = SpellingCorrector()
    # Load training data
    with open("train2.txt", "r") as f:
        training_data = f.readlines()
    corrector.fit(training_data)

    # Load test data
    with open("misspelling_public.txt", "r") as f:
        test_data = f.readlines()

    # Extract incorrect sentences
    incorrect_sentences = [line.split("&&")[1].strip() for line in test_data if "&&" in line]

    # Correct the sentences
    corrected_sentences = corrector.correct(incorrect_sentences)

    # Print the corrected sentences
    for original, corrected in zip(incorrect_sentences, corrected_sentences):
        print(f"Original: {original}")
        print(f"Corrected: {corrected}")
        print()