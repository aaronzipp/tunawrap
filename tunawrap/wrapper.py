"""
This module provides the TunaWrap class, which wraps around optuna for models using
the scikit-learn API. It allows for easy hyperparameter optimization using the Optuna library.
"""

from typing import TYPE_CHECKING, Callable

import optuna
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

if TYPE_CHECKING:
    import pandas as pd


class TunaWrap:
    """A wrapper around Optuna for hyperparameter optimization of scikit-learn compatible models.

    TunaWrap simplifies the process of finding optimal hyperparameters for machine learning
    models by providing a clean interface to Optuna's optimization capabilities. It supports
    two approaches for providing models: using sklearn's clone() or a factory function.

    Attributes:
        study (optuna.Study): The Optuna study object containing optimization results.
            Available after calling cook().

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.metrics import make_scorer, accuracy_score
        >>>
        >>> # Using the clone approach
        >>> model = RandomForestClassifier(random_state=42)
        >>> parameters = {
        ...     'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
        ...     'max_depth': {'type': 'int', 'low': 5, 'high': 30}
        ... }
        >>> wrapper = TunaWrap(
        ...     model=model,
        ...     parameters=parameters,
        ...     X=X_train,
        ...     y=y_train,
        ...     scorer=make_scorer(accuracy_score),
        ...     folds=5,
        ...     random_state=42
        ... )
        >>> best_params = wrapper.cook(n_trials=100)
    """

    def __init__(
        self,
        parameters: dict,
        X: "pd.DataFrame",
        y: "pd.Series",
        scorer: Callable,
        folds: int | None = None,
        cv=None,
        model=None,
        model_factory: Callable | None = None,
        random_state: int | None = None,
    ):
        """Initialize the TunaWrap optimizer.

        Args:
            parameters: Dictionary defining the hyperparameter search space.
                Each key is a parameter name, and each value is a dict with:
                - 'type': Either 'int', 'float', or 'categorical'
                - For 'int'/'float': 'low' and 'high' bounds
                - For 'categorical': 'choices' list
                Example:
                    {
                        'n_estimators': {'type': 'int', 'low': 10, 'high': 200},
                        'max_depth': {'type': 'int', 'low': 5, 'high': 30},
                        'criterion': {'type': 'categorical', 'choices': ['gini', 'entropy']}
                    }
            X: Training features.
            y: Training target labels.
            scorer: Scoring function (e.g., from sklearn.metrics.make_scorer).
            folds: Number of cross-validation folds. If specified, creates a KFold splitter.
                Ignored if cv is provided. Defaults to None.
            cv: A cross-validation splitter object (e.g., StratifiedKFold, KFold, etc.).
                If provided, this takes precedence over folds parameter. Defaults to None.
            model: A scikit-learn compatible model instance. TunaWrap will use
                sklearn.base.clone() to create fresh copies for each trial. Cannot be
                used together with model_factory. Defaults to None.
            model_factory: A callable that returns a fresh model instance for each trial.
                Cannot be used together with model. Defaults to None.
            random_state: Random seed for reproducibility. Controls both the KFold
                cross-validation splits (if folds is used) and Optuna's TPE sampler.
                If None, results will vary between runs. Defaults to None.

        Raises:
            ValueError: If both model and model_factory are provided, or if neither is
                provided, or if neither folds nor cv is provided.
        """
        # Support both approaches: model (with clone) or model_factory
        if model is not None and model_factory is not None:
            raise ValueError("Provide either 'model' or 'model_factory', not both")
        if model is None and model_factory is None:
            raise ValueError("Must provide either 'model' or 'model_factory'")
        if folds is None and cv is None:
            raise ValueError("Must provide either 'folds' or 'cv'")

        self.model = model
        self.model_factory = model_factory
        self.parameters = parameters
        self.X = X
        self.y = y
        self.scorer = scorer
        self.folds = folds
        self.cv = cv
        self.random_state = random_state

        if cv is not None:
            self.cv_splitter = cv
            self.cv_splits = list(self.cv_splitter.split(X, y))
        elif folds is not None:
            self.cv_splitter = KFold(
                n_splits=folds, shuffle=True, random_state=self.random_state
            )
            self.cv_splits = list(self.cv_splitter.split(X, y))
        else:
            self.cv_splits = None

    def cook(self, n_trials=20, timeout_seconds=None, debug=False, study=None):
        """Run hyperparameter optimization using Optuna.

        This method performs Bayesian optimization using Optuna's TPE (Tree-structured
        Parzen Estimator) sampler to find the best hyperparameters for the model. Each
        trial evaluates a different set of hyperparameters using cross-validation.

        Args:
            n_trials: Number of optimization trials to run. Each trial tests a different
                combination of hyperparameters. Defaults to 20.
            timeout_seconds: Maximum time in seconds for the entire optimization.
                If specified, optimization stops after this duration even if n_trials
                haven't completed. Defaults to None (no timeout).
            debug: If True, prints detailed debugging information for each trial, including
                model IDs, parameters before/after setting, and cross-validation scores.
                Useful for troubleshooting. Defaults to False.
            study: An existing Optuna study to continue optimization from. If provided,
                this study will be used instead of creating a new one. Useful for
                warm-starting with enqueued trials or resuming previous optimizations.
                Defaults to None (creates new study).

        Returns:
            The best hyperparameters found during optimization.

        Example:
            >>> # Basic usage
            >>> best_params = wrapper.cook(n_trials=100, timeout_seconds=600)
            >>> print(f"Best parameters: {best_params}")
            >>> print(f"Best score: {wrapper.study.best_value}")

            >>> # Warm-start with previous best parameters
            >>> study = optuna.create_study(direction="maximize")
            >>> study.enqueue_trial({"n_estimators": 100, "max_depth": 5})
            >>> best_params = wrapper.cook(n_trials=100, study=study)
        """

        def objective(trial: optuna.Trial):
            trial_parameters = self._parse_parameters(trial)

            # Use clone if model was provided, otherwise use factory
            if self.model is not None:
                trial_model = clone(self.model)
            else:
                trial_model = self.model_factory()

            if debug:
                print(f"\n--- Trial {trial.number} Debug Info ---")
                if self.model is not None:
                    print(f"Using clone() approach with base model: {self.model}")
                else:
                    print(f"Using factory approach: {self.model_factory}")
                print(f"Created model id: {id(trial_model)}")
                print(f"Trial parameters to set: {trial_parameters}")

                # Get relevant params before setting
                before_params = {
                    k: trial_model.get_params()[k] for k in trial_parameters.keys()
                }
                print(f"Before set_params: {before_params}")

            trial_model.set_params(**trial_parameters)

            if debug:
                # Get relevant params after setting
                after_params = {
                    k: trial_model.get_params()[k] for k in trial_parameters.keys()
                }
                print(f"After set_params: {after_params}")
                print(f"Parameters changed? {before_params != after_params}")

            if self.cv_splits is not None:
                scores = cross_val_score(
                    trial_model,
                    self.X,
                    self.y,
                    cv=self.cv_splits,
                    scoring=self.scorer,
                )
                if debug:
                    print(f"CV scores: {scores}")
                    print(f"Mean score: {scores.mean()}")
                return scores.mean()
            else:
                raise NotImplementedError(
                    "Only cross-validation is implemented for now. Please specify either 'folds' or 'cv'."
                )

        # Use provided study or create a new one
        if study is not None:
            self.study = study
        else:
            self.study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
            )
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

        return self.study.best_params

    def _parse_parameters(self, trial: optuna.Trial) -> dict:
        """Parse the parameter configuration and suggest values for a trial.

        This internal method converts the parameter configuration dictionary into
        actual parameter values suggested by Optuna for the current trial. It supports
        integer, float, and categorical parameter types.

        Args:
            trial: The Optuna trial object used to suggest parameter values.

        Returns:
            A dictionary mapping parameter names to their suggested values for this trial.

        Raises:
            ValueError: If an unsupported parameter type is encountered.
        """
        trial_parameters = {}
        for parameter_name, config in self.parameters.items():
            if config["type"] == "float":
                trial_parameters[parameter_name] = trial.suggest_float(
                    parameter_name, config["low"], config["high"]
                )
            elif config["type"] == "int":
                trial_parameters[parameter_name] = trial.suggest_int(
                    parameter_name, config["low"], config["high"]
                )
            elif config["type"] == "categorical":
                trial_parameters[parameter_name] = trial.suggest_categorical(
                    parameter_name, config["choices"]
                )
            else:
                raise ValueError(
                    f"Unsupported parameter type: {config['type']} for {parameter_name}"
                )
        return trial_parameters
