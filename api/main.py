#main.py
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
# enregistrement du meilleur modèle final
best_model_name = results_df_grid.iloc[0]["model"]
print("Best model:", best_model_name)
final_model = best_models[best_model_name]
mlflow.sklearn.log_model(final_model,name="model",registered_model_name=f"HomeCredit_Scoring_final_{best_model_name.replace(' ', '_')}")
i

model_name = "HomeCredit_Scoring_final_XGBoost"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")