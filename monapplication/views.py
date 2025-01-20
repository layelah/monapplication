import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import CSVUploadForm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import optuna

# Fonction d'évaluation des modèles
@csrf_exempt
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F_score': f1_score(y_test, y_pred),
        'Confusion_Matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

# Fonction pour traiter le fichier CSV et entraîner un modèle
@csrf_exempt
def process_file_and_train_model(uploaded_file):
    try:
        dataset = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement du fichier: {e}")
    
    # Traitement des données
    x = dataset.drop(columns=["legitimate"])
    y = dataset["legitimate"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Modèle sans optimisation
    mon_model = DecisionTreeClassifier(random_state=42)
    mon_model.fit(x_train, y_train)

    # Optimisation avec Optuna
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }
        model = DecisionTreeClassifier(**params, random_state=42)
        model.fit(x_train, y_train)
        return f1_score(y_test, model.predict(x_test))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    optimized_model = DecisionTreeClassifier(**best_params, random_state=42)
    optimized_model.fit(x_train, y_train)

    # Évaluation des modèles
    baseline_results = evaluate_model(mon_model, x_test, y_test)
    optimized_results = evaluate_model(optimized_model, x_test, y_test)

    return baseline_results, optimized_results, best_params

# Vue principale
@csrf_exempt
def mavue(request):
    baseline_results = None
    optimized_results = None
    best_params = None

    # Vérifier si le formulaire a été soumis
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']

        try:
            baseline_results, optimized_results, best_params = process_file_and_train_model(uploaded_file)
        except ValueError as e:
            return HttpResponse(str(e))
        
        # Rediriger après le traitement pour éviter la soumission multiple
         
    
    # Formulaire de téléchargement
    form = CSVUploadForm()

    context = {
        'form': form,
        'baseline': baseline_results,
        'optimized': optimized_results,
        'best_params': best_params,
    }
    
    return render(request, 'monapplication/mapage.html', context)
