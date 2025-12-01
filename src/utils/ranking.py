import pandas as pd

def compute_customer_scores(df, model, method="iso"):
    scores = []

    for cid in df["customer_id"].unique():
        cust_df = df[df["customer_id"] == cid].sort_values("date")

        if len(cust_df) < 10:
            continue  # ignore useless samples

        X = cust_df["consumption"].values.reshape(-1, 1)

        if method == "iso":
            score = -model.score_samples(X).mean()

        elif method == "ae":
            import torch
            X_tensor = torch.tensor(X).float().unsqueeze(0)
            recon = model(X_tensor).detach().numpy().flatten()
            score = ((X.flatten() - recon) ** 2).mean()

        else:
            raise ValueError("Unknown method")

        scores.append([cid, score])

    return pd.DataFrame(scores, columns=["customer_id", "anomaly_score"])