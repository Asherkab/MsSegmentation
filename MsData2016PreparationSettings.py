import os
from Logger.Logger import Logger
from Plots.Plots import Plots
from Metrics.Metrics import Metrics
from DataUtils.DataUtils import DataUtils
from MsUtils import MsUtils as Utils


class MsDataPreparationSettings(object):

    def __init__(self):

        # Dataset settings
        self.data = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData"
        self.masks = "../../Datasets/MMSEG_MICCAI2016/UnprocessedTrainingData"
        self.data_folder = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData/axial_slices"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        self.cases = ["01016SACH",
                      "01038PAGU",
                      "01039VITE",
                      "07001MOEL",
                      "07003SATH",
                      "07010NABO",
                      "08002CHJE",
                      "08027SYBR",
                      "08029IVDI",
                      "01040VANE",
                      "07040DORE",
                      "08031SEVE",
                      "01042GULE",
                      "07043SEME",
                      "08037ROGU"]
        self.modalities = [0, 1, 2, 3, 4]
        self.experts_num = 7

        # Data utils settings
        self.resize_shape = (181, 217, 181)
        self.min_clip_value = None
        self.max_clip_value = None
        self.data_utils = DataUtils(self)

        # Ms utils settings
        self.connected_to_original_mask = False
        self.dilation_thr = 0.2
        self.initial_data_thr = 0.99
        self.dilation_iterations = 6
        self.soft_label = 0.3
        self.ms_utils = Utils(self)

        # Plot settings
        self.plot_examples = False
        self.output_plots_directory = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData/plots"
        self.plots = Plots(self)

        # Logger settings
        self.output_logs_name = "data_characteristics.log"
        self.output_logs_path = os.path.join(self.data_folder, self.output_logs_name)
        self.logger = Logger(self)

        # Metrics settings
        self.calculate_inter_rater_metrics = False
        self.metrics_container = Metrics(self)
