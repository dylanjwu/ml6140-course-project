from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from models import get_data

filename = '/Users/dylanwu/Development/cs_6140/project/data/tfidf_pca_100features_1000rows.csv'
X_train, y_train = get_data(filename=filename)

mlp = MLPClassifier(max_iter=1000)

# Define the hyperparameters to tune
param_grid = {
    # 'hidden_layer_sizes': [(i,j,k) for i in range(95, 101) for j in range(95, 101) for k in range(95, 101)],
    'hidden_layer_sizes': [(i,j) for i in range(90, 102) for j in range(90, 102)],
    # 'hidden_layer_sizes': [(i) for i in range(60, 101)],
    'activation': ['relu'],
    'learning_rate': ['constant']
}

grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found:")
print(grid_search.best_params_)
print("Best cross-validation score:")
print(grid_search.best_score_)

