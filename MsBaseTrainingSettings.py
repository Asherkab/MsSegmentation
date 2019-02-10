import os
import abc
import numpy as np
from keras.optimizers import Adam
from keras.regularizers import l2
from Plots.Plots import Plots
from Logger.Logger import Logger
from DataUtils.DataUtils import DataUtils
from Generator.Generator import Generator
from KerasLossFunctions.KerasLossFunctions import KerasLossFunctions
from Metrics.Metrics import Metrics
from Metrics.KerasMetrics import KerasMetrics
from KerasCallbacks.KerasCallbacks import KerasCallbacks
from KerasModels.CenModel import CenModel as Model
from MsMaskTypes import MsMaskTypes as MaskTypes


class MsBaseTrainingSettings(abc.ABC):

    def __init__(self):

        # Augmentation settings
        self.rotation = 5

        # Data generator settings
        self.data_random_seed = 2018
        self.folds = 5
        self.train_split = 0.8
        self.test_split = 0.1  # useful only in 1-fold setting
        self.leave_out = True  # allows to choose for test data with unique values of 'self.leave_out_param'
        self.leave_out_param = 'patient'
        self.leave_out_values = [1]  # useful only in 1-fold setting
        self.generator = Generator(self)

        # Data preprocessing settings
        self.preload_data = True
        self.preload_labels = True
        self.crop = True
        self.crop_offset = [[0, 1], [0, 1], [0, 0]]

        # Mask preprocessing settings
        self.training_mask_type = MaskTypes.EXPERT_1

        # Data postprocessing settings
        self.find_thr_mask_type = MaskTypes.EXPERT_1
        self.opt_thr = 0.01
        self.find_opt_thr = True
        self.thrs_to_check = np.arange(0.01, 1, 0.01)
        self.data_utils = DataUtils(self)

        # Output settings
        self.simulation_folder = "../../Simulations/MsSegmentation/test"
        self.train_data_file_name = "train_data.json"
        self.val_data_file_name = "val_data.json"
        self.test_data_file_name = "test_data.json"

        # Model architecture settings
        self.model = Model(self)
        self.kernel_regularizer = l2(0.0001)
        self.kernel_initializer = "glorot_normal"

        # Model compilation settings
        self.losses = KerasLossFunctions(self)
        self.keras_metrics_container = KerasMetrics(self)
        self.optimizer = Adam(lr=0.0001)
        self.loss = self.losses.dice_coef_loss()
        self.keras_metrics = [self.keras_metrics_container.dice_coef,
                              self.keras_metrics_container.recall,
                              self.keras_metrics_container.precision]

        # Model training settings
        self.load_weights = False
        self.load_weights_name = "best_weights.h5"
        self.load_weights_path = os.path.join(self.simulation_folder, self.load_weights_name)
        self.train_model = True

        # Callbacks settings
        self.callbacks_container = KerasCallbacks(self)
        self.save_weights_name = "best_weights.h5"  # "weights_{epoch:03d}.h5"
        self.save_weights_path = os.path.join(self.simulation_folder, self.save_weights_name)
        self.save_best_only = True
        self.training_log_name = "metrics.log"
        self.training_log_path = os.path.join(self.simulation_folder, self.training_log_name)
        self.monitor = "val_loss"
        self.early_stopping_patience = 1500
        self.reduce_lr_factor = 0.9
        self.reduce_lr_patience = 10
        self.reduce_lr_min_lr = 0.00001
        self.callbacks = [self.callbacks_container.checkpoint(),
                          self.callbacks_container.csv_logger(),
                          self.callbacks_container.early_stopping(),
                          self.callbacks_container.reduce_lr_onplateu()]

        # Logger settings
        self.logger = Logger(self)
        self.output_logs_name = "results.log"
        self.output_logs_path = os.path.join(self.simulation_folder, self.output_logs_name)
        self.log_message = ""

        # Plot settings
        self.plots = Plots(self)
        self.output_plots_directory = self.simulation_folder
        self.plot_metrics = ["loss", "dice_coef", "recall", "precision"]

        # Metrics settings
        self.metrics_container = Metrics(self)
