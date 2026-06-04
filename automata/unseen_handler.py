class UnseenHandler:
    """
    Eğitimde görülmeyen pattern'ları Levenshtein distance ile
    en yakın bilinen state/pattern'e eşler.
    """

    def __init__(self, known_patterns: set[str]):
        if not known_patterns:
            raise ValueError("known_patterns boş olamaz.")

        self.known_patterns = set(known_patterns)

    @staticmethod
    def levenshtein_distance(a: str, b: str) -> int:
        rows = len(a) + 1
        cols = len(b) + 1

        dp = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            dp[i][0] = i

        for j in range(cols):
            dp[0][j] = j

        for i in range(1, rows):
            for j in range(1, cols):
                cost = 0 if a[i - 1] == b[j - 1] else 1

                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )

        return dp[-1][-1]

    def find_nearest(self, pattern: str):
        best_pattern = None
        best_distance = float("inf")

        for known_pattern in self.known_patterns:
            distance = self.levenshtein_distance(pattern, known_pattern)

            if distance < best_distance:
                best_distance = distance
                best_pattern = known_pattern

        return {
            "original_pattern": pattern,
            "nearest_pattern": best_pattern,
            "distance": best_distance
        }

    def handle(self, pattern: str):
        if pattern in self.known_patterns:
            return {
                "pattern": pattern,
                "status": "known",
                "mapped_to": pattern,
                "distance": 0
            }

        nearest = self.find_nearest(pattern)

        return {
            "pattern": pattern,
            "status": "unseen",
            "mapped_to": nearest["nearest_pattern"],
            "distance": nearest["distance"]
        }