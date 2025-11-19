from sklearn.ensemble import IsolationForest

def train_isolation_forest(X):
    model = IsolationForest(
        n_estimators = 200,
        contamination= 0.03,
        random_state=42
    )
    model.fit(X)
    return model
def predict_anomalies(model, X):
    preds = model.predict(X)
    scores = model.decision_function(X)
    return preds, scores