import os
from Utils.RunMode import RunMode
from MsSegmentationSettings import MsSegmentationSettings as Settings


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

    logger.log("\nFold: " + str(fold))

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

    # Reinitialize weights and train
    if settings.train_model:
        # Update simulation directories for training
        settings.save_weights_path = os.path.join(fold_simulation_folder, settings.save_weights_name)

        # Update callbacks to make effect of directories change
        settings.callbacks = [settings.callbacks_container.checkpoint(),
                              settings.callbacks_container.csv_logger(),
                              settings.callbacks_container.early_stopping(),
                              settings.callbacks_container.reduce_lr_onplateu()]
        model.initialize()
        model.fit(fold=fold)

    # Visualize training metrics
    if not settings.train_model:
        plots.training_plot()

    if not settings.train_model:
        # Calculate predictions on training data
        train_predictions = model.predict(run_mode=RunMode.TRAINING, fold=fold)
        train_data = generator.get_data(run_mode=RunMode.TRAINING, fold=fold)

        # Evaluate model and calculate predictions on test data
        test_predictions = model.predict(run_mode=RunMode.TEST, fold=fold)
        test_evaluations = model.evaluate(run_mode=RunMode.TEST, fold=fold)
        test_data = generator.get_data(run_mode=RunMode.TEST, fold=fold)

        # Calculate metrics
        # dataset.calculate_fold_metrics(test_predictions, test_data, test_evaluations, train_predictions, train_data)

        # Save tested data
        # dataset.save_tested_data(generator.test_info[fold])

# Log metrics
dataset.log_metrics()

logger.end()
