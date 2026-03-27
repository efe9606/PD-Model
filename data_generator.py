import numpy as np
import pandas as pd

def generate_data(n=1000, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        'loan_amount': np.random.normal(50000, 15000, n),
        'income': np.random.normal(70000, 20000, n),
        'credit_score': np.random.normal(650, 50, n),
        'age': np.random.randint(21, 65, n),
        'default': np.random.binomial(1, 0.1, n)
    })

    df["loan_to_income"] = df["loan_amount"] / df["income"]

    return df