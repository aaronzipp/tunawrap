# TunaWrap

A simple wrapper around Optuna for hyperparameter optimization of scikit-learn compatible models.

## Note

This is a work in progress.

## Usage

TunaWrap supports two approaches for providing models:

### Approach 1: Using `model` (with sklearn's clone)

```python
from tunawrap import TunaWrap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score

# Define parameters to optimize
parameters = {
    'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
    'max_depth': {'type': 'int', 'low': 2, 'high': 20},
    'min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
}

# Create a base model (set random_state for reproducibility)
model = RandomForestClassifier(random_state=42)

# Create wrapper and optimize
wrapper = TunaWrap(
    model=model,
    parameters=parameters,
    X=X_train,
    y=y_train,
    scorer=make_scorer(accuracy_score),
    folds=5
)

best_params = wrapper.cook(n_trials=100)
```

### Approach 2: Using `model_factory` (function that creates models)

```python
from tunawrap import TunaWrap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Define parameters to optimize
parameters = {
    'classifier__n_estimators': {'type': 'int', 'low': 10, 'high': 200},
    'classifier__max_depth': {'type': 'int', 'low': 2, 'high': 20},
    'classifier__min_samples_split': {'type': 'int', 'low': 2, 'high': 10}
}

# Create wrapper and optimize using a model factory
wrapper = TunaWrap(
    model_factory=lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ]),
    parameters=parameters,
    X=X_train,
    y=y_train,
    scorer='accuracy',
    folds=5
)

best_params = wrapper.cook(n_trials=100)
```

**Note:** You must provide either `model` or `model_factory`, but not both.

## Parameter Types

- `int`: Integer parameters with `low` and `high` bounds
- `float`: Float parameters with `low` and `high` bounds
- `categorical`: Categorical parameters with `choices` list
