import matplotlib.pyplot as plt

def plot_consumption(df, customer_id):
    temp = df[df['customer_id'] == customer_id]
    plt.figure(figsize=(12,4))
    plt.plot(temp['timestamp'], temp['consumption'])
    plt.title(f'Consumption Pattern - Customer {customer_id}')
    plt.tight_layout()
    plt.show()

def plot_anomaly_scores(scores):
    plt.figure(figsize=(10,3))
    plt.plot(scores)
    plt.title('Anomaly Scores timeline')
    plt.tight_layout()
    plt.show()