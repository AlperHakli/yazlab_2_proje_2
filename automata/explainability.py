from settings import settings

class ExplainabilityEngine:
    def __init__(self, automata):
        self.automata = automata
        # kümülatif path probability takibi için başlangıç değeri (1.0)
        self.cumulative_path_probability = 1.0

    def reset_path(self):
        """her yeni test serisi analizinde kümülatif olasılığı sıfırlamak için kullanılır."""
        self.cumulative_path_probability = 1.0

    def explain(
            self,
            current_state: str,
            incoming_pattern: str,
            status: str,
            mapped_to: str,
            distance: int = 0
    ):
        # anlık geçiş olasılığını al P(Si -> Sj)
        transition_probability = self.automata.get_transition_probability(
            current_state,
            mapped_to
        )

        #ardışık geçiş olasılıklarının çarpımı
        prob_for_multiplier = transition_probability if transition_probability > 0 else 1e-4
        self.cumulative_path_probability *= prob_for_multiplier

        # karar mekanizması treshold kontrolü yapılır
        if transition_probability < settings.AUTOMATA_THRESHOLD:
            decision = "anomaly"
        else:
            decision = "normal"

        # proje isterlerindeki zorunlu json şablonu
        return {
            "state": current_state,
            "pattern": incoming_pattern,
            "status": status,
            "mapped_to": mapped_to,
            "probability": float(transition_probability),
            "path_probability": float(self.cumulative_path_probability),
            "decision": decision,
            "confidence": float(transition_probability),
            "distance": distance
        }