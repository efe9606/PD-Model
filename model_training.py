from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(df):
    """
    Train a Logistic Regression PD model.

    Returns:
        model: Trained logistic regression model
        X_test: Test features
        y_test: Test labels
    """

    features = ["loan_amount", "income", "credit_score", "age", "loan_to_income"]

    X = df[features]
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    return model, X_test, y_test