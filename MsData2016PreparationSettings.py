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
        self.masks = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData"
        self.data_folder = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData/axial_slices_dilation_0.2_label_0.3_with_soft_staple_simplified_implemented"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        self.cases = ["01016SACH_reg",
                      "01038PAGU_reg",
                      "01039VITE_reg",
                      "01040VANE_reg",
                      "01042GULE_reg",
                      "07001MOEL_reg",
                      "07003SATH_reg",
                      "07010NABO_reg",
                      "07040DORE_reg",
                      "07043SEME_reg",
                      "08002CHJE_reg",
                      "08027SYBR_reg",
                      "08029IVDI_reg",
                      "08031SEVE_reg",
                      "08037ROGU_reg"]
        self.modalities = [0, 1, 2, 3, 4]
        self.experts_num = 10  # 7 experts(1-7), consensus(0), intersection(8), staple(9)

        # Data utils settings
        self.min_clip_value = None
        self.max_clip_value = None
        self.data_utils = DataUtils(self)

        # Ms utils settings
        self.connected_to_original_mask = True
        self.dilation_thr = 0.4
        self.initial_data_thr = 0.99
        self.dilation_iterations = 3
        self.soft_label = 0.3
        self.apply_staple_threshold = True
        self.staple_threshold = 0.5
        self.ms_utils = Utils(self)

        # Plot settings
        self.plot_examples = True
        self.output_plots_directory = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData/plots_mask_average_heatmap_on_flair"
        self.plots = Plots(self)

        # Logger settings
        self.output_logs_name = "data_characteristics.log"
        self.output_logs_path = os.path.join(self.data_folder, self.output_logs_name)
        self.logger = Logger(self)

        # Metrics settings
        self.calculate_inter_rater_metrics = True
        self.metrics_container = Metrics(self)
