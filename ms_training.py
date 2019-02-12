import os
from Utils.RunMode import RunMode
from Ms2016TrainingSettings import Ms2016TrainingSettings as Settings


# Initialize settings and get instances of classes
settings = Settings()
logger = settings.logger
plots = settings.plots
dataset = settings.dataset
generator = settings.generator
model = settings.model

# Create simulations directory
if not os.path.exists(settings.simulation_folder):
    os.makedirs(settings.simulation_folder)

# Initialize logger
logger.start(settings.output_logs_path)
logger.log(settings.log_message)

# Build and compile model
model.build()
model.compile()
if settings.load_weights:
    model.load_weights()
logger.log(model.summary())

# Split data to training/validation/test and folds
generator.split_data()

# Train or test model for each fold
for fold in range(settings.folds):

    logger.log("\nFold: {0}".format(fold))
    if settings.leave_out:
        leave_out_unique_values = generator.test_info[fold][settings.leave_out_param].unique()
        logger.log("Leave out values are: {0}".format(leave_out_unique_values))

    # Update simulation directories for current fold
    fold_simulation_folder = os.path.join(settings.simulation_folder, str(fold))
    if not os.path.exists(fold_simulation_folder):
        os.makedirs(fold_simulation_folder)

    settings.output_plots_directory = fold_simulation_folder
    settings.load_weights_path = os.path.join(fold_simulation_folder, settings.load_weights_name)
    settings.training_log_path = os.path.join(fold_simulation_folder, settings.training_log_name)

    # Save list of training/validation/test examples
    generator.train_info[fold].to_json(os.path.join(fold_simulation_folder, settings.train_data_file_name))
    generator.val_info[fold].to_json(os.path.join(fold_simulation_folder, settings.val_data_file_name))
    generator.test_info[fold].to_json(os.path.join(fold_simulation_folder, settings.test_data_file_name))

    if settings.train_model:
        # Update simulation directories for training
        settings.save_weights_path = os.path.join(fold_simulation_folder, settings.save_weights_name)

        # Update callbacks to make effect of directories change
        settings.callbacks = [settings.callbacks_container.checkpoint(),
                              settings.callbacks_container.csv_logger(),
                              settings.callbacks_container.early_stopping(),
                              settings.callbacks_container.reduce_lr_onplateu()]
        model.build()
        model.compile()
        model.fit(fold=fold)  # Train model
    else:
        # Visualize training metrics
        plots.training_plot()

        # Calculate predictions on training data
        train_predictions = model.predict(run_mode=RunMode.TRAINING, fold=fold)
        train_data = generator.get_data(run_mode=RunMode.TRAINING, fold=fold)

        # Evaluate model and calculate predictions on test data
        test_predictions = model.predict(run_mode=RunMode.TEST, fold=fold)
        test_evaluations = model.evaluate(run_mode=RunMode.TEST, fold=fold)
        test_data = generator.get_data(run_mode=RunMode.TEST, fold=fold)

        # Apply postprocessing
        test_predictions = dataset.apply_postprocessing(test_predictions, test_data,
                                                        train_predictions, train_data,
                                                        {"train_info": generator.train_info[fold],
                                                        "val_info": generator.val_info[fold],
                                                        "test_info": generator.test_info[fold]})

        # Calculate metrics
        dataset.calculate_fold_metrics(test_predictions, test_data, test_evaluations, train_predictions, train_data,
                                       fold,
                                       {"train_info": generator.train_info[fold],

                                        "val_info": generator.val_info[fold],
                                        "test_info": generator.test_info[fold]})

        # Save tested data
        dataset.save_tested_data(generator.test_info[fold])

if not settings.train_model:
    dataset.log_metrics()  # Log metrics

logger.end()
