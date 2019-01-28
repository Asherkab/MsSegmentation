import os
from keras.optimizers import Adam
from keras.regularizers import l2
from Plots.Plots import Plots
from Logger.Logger import Logger
from DataUtils.DataUtils import DataUtils
from Generator.Generator import Generator
from KerasLossFunctions.LossFunctions import LossFunctions
from KerasMetrics.Metrics import Metrics
from KerasCallbacks.Callbacks import Callbacks
from KerasModels.CenModel import CenModel as Model
from MsSegmentationDataset import MsSegmentationDataset as Dataset


class MsSegmentationSettings(object):

    def __init__(self):

        # Dataset settings
        self.data_folder = "../../Datasets/MS_Longitudinal_ISBI2015/training/axial_slices/"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        # Data preprocessing settings
        self.filters = {"expert_1": {"min_open": 1}}
        self.preload_data = True
        self.preload_labels = True
        self.crop = True
        self.crop_offset = [[0, 1], [0, 1], [0, 0]]

        # 0 - 1st rater
        # 1 - 2nd rater
        # 2 - union
        # 3 - intersection
        # 4 - binary intersection region, soft union region
        # 5 - 3d dilated 1st rater
        # 6 - 3d conditionally dilated (constant iterations number)
        # 7 - soft filtered union
        # 8 - binary intersection, soft union, soft expansion
        # 9 - gradual soft union
        # 10 - 2d conditionally dilated soft 1st rater (constant data threshold)
        # 11 - 2d dilated soft filtered 2nd rater (constant iterations number)
        self.mask_type = 2

        self.max_soft_label = 1
        self.min_soft_label = 0.3
        self.dilation_iterations = 6
        self.dilation_thr = 0.2
        self.data_thr = 0.99  # threshold of flair modality to extract candidates
        self.min_candidates = 1  # use slice with more than 'min_candidates'
        self.data_utils = DataUtils(self)
        self.dataset = Dataset(self)

        # Data postprocessing settings
        self.opt_thr = 0.5
        self.find_opt_thr = True

        # Augmentation settings
        self.rot_angle = 5.0

        # Data generator settings
        self.data_random_seed = 2018
        self.balance = False
        self.folds = 5
        self.train_split = 0.8
        self.test_split = 0.1
        self.generator = Generator(self)

        # Output settings
        self.simulation_folder = "../../Simulations/MsSegmentation/test"
        self.train_data_file_name = "train_data.json"
        self.val_data_file_name = "val_data.json"
        self.test_data_file_name = "test_data.json"

        self.training_logs = ""
        self.output_segmentations = ""

        # Model architecture settings
        self.model = Model(self)
        self.kernel_regularizer = l2(0.0001)
        self.kernel_initializer = "glorot_normal"

        # Model compilation settings
        self.input_shape = (216, 180, 4)
        self.losses = LossFunctions(self)
        self.metric = Metrics(self)
        self.optimizer = Adam(lr=0.0001)
        self.loss = self.losses.dice_coef_loss()
        self.metrics = [self.metric.dice_coef, self.metric.recall, self.metric.precision]

        # Model training settings
        self.load_weights = False
        self.load_weights_name = "best_weights.h5"
        self.load_weights_path = os.path.join(self.simulation_folder, self.load_weights_name)
        self.train_model = True
        self.epochs = 7000
        self.steps_per_epoch = 50
        self.batch_size = 32
        self.val_steps = 15

        # Callbacks settings
        self.callbacks_container = Callbacks(self)
        self.save_weights_name = "weights_{epoch:03d}.h5"
        self.save_weights_path = os.path.join(self.simulation_folder, self.save_weights_name)
        self.save_best_only = True
        self.training_log_name = "metrics.log"
        self.training_log_path = os.path.join(self.simulation_folder, self.training_log_name)
        self.monitor = "val_loss"
        self.early_stopping_patience = 50
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
        self.small_objects = 18  # elements of less size removed during postprocessing
        self.smallest_lesion_size = 18  # elements of less size removed before LTPR / FTPR calculations
        self.min_overlap = 1  # if overlap between prediction and gt is more than 'min_overlap' it's a true prediction
        self.epsilon = 1e-6
