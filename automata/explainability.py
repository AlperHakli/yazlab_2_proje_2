from settings import settings
class ExplainabilityEngine:

    def __init__(self, automata):
        self.automata = automata

    def explain(
            self,
            current_state: str,
            incoming_pattern: str,
            status: str,
            mapped_to: str
    ):

        transition_probability = self.automata.get_transition_probability(
            current_state,
            mapped_to
        )

        if transition_probability < settings.AUTOMATA_THRESHOLD:
            decision = "anomaly"
        else:
            decision = "normal"

        return {
            "current_state": current_state,
            "pattern": incoming_pattern,
            "status": status,
            "mapped_to": mapped_to,
            "transition_probability": transition_probability,
            "path_probability": transition_probability,
            "decision": decision,
            "confidence": transition_probability
        }