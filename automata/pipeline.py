from automata.paa import PAA
from automata.sax import SAX
from automata.pattern_extractor import PatternExtractor
from automata.probabilistic_automata import ProbabilisticAutomata
from automata.unseen_handler import UnseenHandler
from automata.explainability import ExplainabilityEngine


class AutomataPipeline:
    """
    PC1 zaman serisini alır:
    PAA -> SAX -> Pattern -> Automata -> Explainability
    zincirini çalıştırır.
    """

    def __init__(self, paa_window_size=4, pattern_window_size=4, alphabet_size=3):
        self.paa = PAA(window_size=paa_window_size)
        self.sax = SAX(alphabet_size=alphabet_size)
        self.pattern_extractor = PatternExtractor(window_size=pattern_window_size)

        self.automata = ProbabilisticAutomata()
        self.unseen_handler = None
        self.explainer = None
        self.train_patterns = None

    def fit(self, train_series):
        paa_values = self.paa.transform(train_series)
        sax_string = self.sax.transform(paa_values)
        patterns = self.pattern_extractor.transform(sax_string)

        self.train_patterns = patterns
        self.automata.fit(patterns)

        self.unseen_handler = UnseenHandler(self.automata.get_states())
        self.explainer = ExplainabilityEngine(self.automata)

        return {
            "paa_values": paa_values,
            "sax_string": sax_string,
            "patterns": patterns,
            "states": self.automata.get_states(),
            "transition_table": self.automata.get_transition_table(),

        }

    def predict(self, test_series):
        if self.unseen_handler is None or self.explainer is None:
            raise ValueError("Önce fit() çalıştırılmalı.")

        paa_values = self.paa.transform(test_series)
        sax_string = self.sax.transform(paa_values)
        test_patterns = self.pattern_extractor.transform(sax_string)

        explanations = []

        for i in range(len(test_patterns) - 1):
            current_state_info = self.unseen_handler.handle(test_patterns[i])
            next_state_info = self.unseen_handler.handle(test_patterns[i + 1])

            current_state = current_state_info["mapped_to"]
            incoming_pattern = test_patterns[i + 1]
            mapped_to = next_state_info["mapped_to"]

            explanation = self.explainer.explain(
                current_state=current_state,
                incoming_pattern=incoming_pattern,
                status=next_state_info["status"],
                mapped_to=mapped_to,
                distance=next_state_info["distance"]
            )

            explanation["time_step"] = i + 1
            explanation["distance"] = next_state_info["distance"]

            explanations.append(explanation)

        return explanations