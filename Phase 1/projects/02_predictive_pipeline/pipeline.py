import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Data Generation (Simulate Housing Data)
def get_data():
    print("Generating Synthetic Data...")
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    feature_names = [f"feat_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some non-linearity (Simulate real world)
    df['feat_0'] = df['feat_0'] ** 2 
    
    return train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning (XGBoost)
def tune_xgboost(X_train, y_train):
    print("Tuning XGBoost with Optuna...")
    
    def objective(trial):
        param = {
            'verbosity': 0,
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'n_estimators': 100
        }
        
        reg = xgb.XGBRegressor(**param)
        score = cross_val_score(reg, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        return score.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # Low trial count for demo speed
    
    print(f"Best params: {study.best_params}")
    return xgb.XGBRegressor(**study.best_params)

# 3. The Grand Pipeline
def build_and_train():
    X_train, X_test, y_train, y_test = get_data()
    
    # Base Models
    xgb_model = tune_xgboost(X_train, y_train)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    svm_model = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    knn_model = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5))])
    enet_model = Pipeline([('scaler', StandardScaler()), ('enet', ElasticNet(alpha=0.1))])
    
    # Stacking (Ensemble of 5)
    print("Training Stacked Regressor (XGB, RF, SVR, KNN, ElasticNet)...")
    estimators = [
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('svm', svm_model),
        ('knn', knn_model),
        ('enet', enet_model)
    ]
    
    # Final Estimator (Meta Learner) uses Ridge Regression to blend
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge()
    )
    
    stack.fit(X_train, y_train)
    
    # Evaluate
    preds = stack.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"--- Final Evaluation ---")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save
    joblib.dump(stack, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    build_and_train()
