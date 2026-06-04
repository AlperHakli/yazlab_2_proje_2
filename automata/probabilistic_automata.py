from collections import defaultdict


class ProbabilisticAutomata:
    """
    Pattern listesinden state geçişlerini ve geçiş olasılıklarını çıkarır.
    Örnek:
    ["aab", "abc", "bcc"]
    aab -> abc
    abc -> bcc
    """

    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probabilities = defaultdict(dict)
        self.states = set()

    def fit(self, patterns: list[str]):
        if len(patterns) < 2:
            raise ValueError("Automata oluşturmak için en az 2 pattern gerekir.")

        self.states = set(patterns)

        for current_state, next_state in zip(patterns[:-1], patterns[1:]):
            self.transition_counts[current_state][next_state] += 1

        self._calculate_probabilities()

    def _calculate_probabilities(self):
        for state, next_states in self.transition_counts.items():
            total = sum(next_states.values())

            for next_state, count in next_states.items():
                self.transition_probabilities[state][next_state] = count / total

    def get_transition_probability(self, current_state: str, next_state: str) -> float:
        return self.transition_probabilities.get(current_state, {}).get(next_state, 0.0)

    def get_states(self):
        return self.states

    def get_transition_table(self):
        rows = []

        for current_state, next_states in self.transition_probabilities.items():
            for next_state, probability in next_states.items():
                rows.append({
                    "current_state": current_state,
                    "next_state": next_state,
                    "probability": probability
                })

        return rows