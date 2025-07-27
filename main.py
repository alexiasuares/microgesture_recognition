from src.data_processing import load_process_data
from src.model_training import train_and_evaluate_model

def main():
    csv_file = 'data/attention_detection_dataset_v1.csv'

    X_train, X_val, X_test, y_train, y_val, y_test = load_process_data(csv_file)
    model = train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == '__main__':
    main()
    