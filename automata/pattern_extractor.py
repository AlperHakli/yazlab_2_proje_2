class PatternExtractor:
    """
    SAX çıktısından sliding window yöntemiyle pattern üretir.
    Örnek:
    sax_string = "aabcc"
    window_size = 3
    patterns = ["aab", "abc", "bcc"]
    """

    def __init__(self, window_size: int = 4):
        if window_size <= 0:
            raise ValueError("window_size 0'dan büyük olmalı.")
        self.window_size = window_size

    def transform(self, sax_string: str):
        patterns = []

        for i in range(len(sax_string) - self.window_size + 1):
            pattern = sax_string[i:i + self.window_size]
            patterns.append(pattern)

        return patterns