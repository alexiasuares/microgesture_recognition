import pandas as pd
from sklearn.model_selection import train_test_split


def load_process_data(csv_path):
    df = pd.read_csv(csv_path)

    # Seleção de colunas relevantes
    feature_cols = [
        'no_of_face', 'face_x', 'face_y', 'face_w', 'face_h', 'face_con',
        'no_of_hand', 'pose_x', 'pose_y', 'phone', 'phone_x', 'phone_y',
        'phone_w', 'phone_h', 'phone_con'
    ]

    # Features e rótulos
    X = df[feature_cols].fillna(0)
    y = df['label']

    # Divisão dos dados: 70% treino, 15% validação, 15% teste
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test
