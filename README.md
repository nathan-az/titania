# Titania
Titania provides a framework for distributed training of Python models, leveraging Spark for parallelisation. Key usages I have found are:
* Better parallelisation during hyperparameter optimisation
* Faster training of ensemble models
* Faster feature set tests
* Parallel training of independent models predicting in the same pipeline (e.g. propensity -> value models)

For very unbalanced binary classification problems, Titania supports training using re-balanced datasets. At prediction-time, probability calibration as described in [1] is performed to ensure conformity of the score distribution with that of the original unbalanced dataset. This functionality is not currently available for multi-class classification.

Please note that this codebase is a work in progress. As such, usage patterns are in heavy flux. 

Effort is being made to keep models (and submodels) consistent with the sklearn API, while requiring minimal effort from a user perspective. Primary tasks for the user are creation of the training manager class and conformity of the training DataFrame. Both are described below in **Usage**.

## Usage
### Data
Training data for all models to be trained. If you plan to train k models (e.g. trying k hyperparameter configurations, or ensembling k models), a `pyspark.sql.DataFrame` must be prepared as follows:
* All data for each of the k models is contained in the DataFrame
* Each row belongs to only 1 of the k models
* The DataFrame should contain a `dataset_id` column, ranging from 0 to k-1, indicating which model each row belongs to
* The DataFrame should contain all features and the label column

### Manager Class
A "training manager" is a class created to conform to Titania's requirements in parallelisation. (example: [LGBMClassifierManager](https://github.com/nathan-az/titania/blob/1690fcc74287f8862893dd5fe1b1fdc98f6852ed/titania/trainers/algorithm_managers.py#L87)):
* A `train` function must be available
  * Its signature must conforming to `(df, label_col, **training_kwargs)`. `training_kwargs` should include any training-time arguments for your particular algorithm. 
  * The function itself should conduct the training and include any overhead activities. 
* A `binary_predict_proba` function is available
  * This function should output a one-dimensional array containing the predicted probability of a `success` of your given label. This requirement will be removed with the future support of multi-class classification. 

### ModelSpec
The ModelSpec is a simple dataclass used to provide the initialisation specifications for a model. It should contain:
* `base_classifier_class`: the class `type` conforming to the above restrictions
* `model_init_kwargs`: any arguments required when initialising the model (i.e. the model should be initialisable using `base_classifier_class(**model_init_kwargs)`
* `original_alpha`: optional for the case where the dataset provided for a given model was rebalanced and no longer represents the original distribution. If `original_alpha` is set, the class will be modified to include additional functionality so that during training, the `sampled_alpha` is calculated, and during prediction-time, outputs are calibrated as in [1].

### EnsembledClassifier
The EnsembledClassifier has a simple API, taking only two arguments during initialisation:
* model_specs: either a single model specification to be used to instantiate all models, or a list of model specs, where `len(model_specs) == num_datasets`
* model_name: the name of the ensemble to be trained. This is used to identify temporarily saved pickled models from each UDF to then reload into the EnsembledClassifier afterwards, as well as in prediction columns.

## To-do
### High Priority
* Multiple training-time arguments - intention is to add support for passing of different training-time arguments. This will likely be added as an additional attribute in ModelSpec.

### Medium Priority
* Multi-class classification - remove all dependencies limiting functionality to binary classification problems.
* Setup for pypi for easy install - will be done once code is tested and usage pattern is more stable

### Low Priority
* Implement more training managers for common libraries (e.g. XGBoost, sklearn RandomForestClassifier, etc.).
* Implement functions for local training and prediction
* Add more examples in `examples` and readme

## Sources
[[1] Calibrating Probability with Undersampling for Unbalanced Classification (A Dal Pozzolo)](https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf)
