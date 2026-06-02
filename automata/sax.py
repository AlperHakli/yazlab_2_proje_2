import numpy as np
from scipy.stats import norm


class SAX:
    """
    Symbolic Aggregate approXimation.
    Sayısal değerleri harflere dönüştürür.
    """

    def __init__(self, alphabet_size: int = 3):
        if alphabet_size < 2:
            raise ValueError("alphabet_size en az 2 olmalı.")

        self.alphabet_size = alphabet_size
        self.breakpoints = norm.ppf(
            np.linspace(0, 1, alphabet_size + 1)[1:-1]
        )

        self.alphabet = [chr(97 + i) for i in range(alphabet_size)]

    def transform(self, values):
        values = np.array(values, dtype=float)

        sax_result = []

        for value in values:
            index = np.sum(value > self.breakpoints)
            sax_result.append(self.alphabet[index])

        return "".join(sax_result)