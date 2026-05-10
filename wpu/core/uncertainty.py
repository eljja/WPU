def multiply_confidence(old_confidence: float, observation_confidence: float) -> float:
    """Small v1 uncertainty rule used before Bayesian filtering is introduced."""
    return max(0.0, min(1.0, float(old_confidence) * float(observation_confidence)))
