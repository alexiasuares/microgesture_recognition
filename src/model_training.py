import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test):
    # 1. Treinar modelo XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # 2. Avaliação na validação
    y_pred_val = model.predict(X_val)
    print("=== Avaliação na Validação ===")
    print("Acurácia:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val))

    # 3. Avaliação no teste
    y_pred_test = model.predict(X_test)
    print("=== Avaliação no Teste ===")
    print("Acurácia:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))

    # 4. Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Atento', 'Cansado'],
                yticklabels=['Atento', 'Cansado'])
    plt.title("Matriz de Confusão - Teste")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

    # 5. Importância das features
    feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"f{i}" for i in range(X_train.shape[1])]
    importances = model.feature_importances_
    pd.Series(importances, index=feature_names).sort_values().plot(kind='barh')
    plt.title("Importância das Features - XGBoost")
    plt.xlabel("Importância")
    plt.show()

    return model