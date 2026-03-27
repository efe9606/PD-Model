from data_generator import generate_data
from model_training import train_model
from evaluation import evaluate_model

def main():
    df = generate_data()
    model, X_test, y_test = train_model(df)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()