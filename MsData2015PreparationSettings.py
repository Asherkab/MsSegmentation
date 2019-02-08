import os
from Plots.Plots import Plots
from MsUtils import MsUtils as Utils


class MsDataPreparationSettings(object):

    def __init__(self):

        # Dataset settings
        self.masks = "../../Datasets/MS_Longitudinal_ISBI2015/training/masks_A/"
        self.data = "../../Datasets/MS_Longitudinal_ISBI2015/training/matched_A/"
        self.data_folder = "../../Datasets/MS_Longitudinal_ISBI2015/training/axial_slices_dilated/"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        self.cases = [[1, [1, 2, 3, 4]],
                      [2, [1, 2, 3, 4]],
                      [3, [1, 2, 3, 4, 5]],
                      [4, [1, 2, 3, 4]],
                      [5, [1, 2, 3, 4]]]
        self.modalities = [0, 1, 2, 3]

        # Utils settings
        self.connected_to_original_mask = False
        self.dilation_thr = 0.2
        self.initial_data_thr = 0.99
        self.dilation_iterations = 6
        self.soft_label = 0.3
        self.utils = Utils(self)

        # Plot settings
        self.plots = Plots(self)
        self.output_plots_directory = self.data_folder
