from psmltrain.trainers.training_managers import LightGBMTrainingManager


def lightgbm_training_function(
    df, model_params, label_col, dataset_params, dataset_type_col, use_early_stopping
):
    training_manager = LightGBMTrainingManager(model_params=model_params)
    training_manager = training_manager.train(
        df=df,
        label_col=label_col,
        dataset_params=dataset_params,
        dataset_type_col=dataset_type_col,
        use_early_stopping=use_early_stopping,
    )
    return training_manager
