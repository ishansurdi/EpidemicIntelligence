class OutbreakClassifier:
    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold

    def probability(self, recent_growth: float, mobility_signal: float) -> float:
        score = 0.45 + (0.4 * recent_growth) + (0.25 * mobility_signal)
        return min(max(score, 0.0), 1.0)

    def label(self, probability: float) -> int:
        return int(probability >= self.threshold)
