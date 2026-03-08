class DummyModel:
    def predict_proba(self, X):
        # Return dummy probabilities
        n = len(X)
        return [[0.5, 0.5]] * n