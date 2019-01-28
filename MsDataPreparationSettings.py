import os
from MsSegmentationUtils import MsSegmentationUtils as Utils


class MsDataPreparationSettings(object):

    def __init__(self):

        # Dataset settings
        self.masks = "../../Datasets/MS_Longitudinal_ISBI2015/training/masks_A/"
        self.data = "../../Datasets/MS_Longitudinal_ISBI2015/training/matched_A/"
        self.data_folder = "../../Datasets/MS_Longitudinal_ISBI2015/training/axial_slices/"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        self.cases = [[1, [1, 2, 3, 4]],
                      [2, [1, 2, 3, 4]],
                      [3, [1, 2, 3, 4, 5]],
                      [4, [1, 2, 3, 4]],
                      [5, [1, 2, 3, 4]]]
        self.modalities = [0, 1, 2, 3]

        self.utils = Utils(self)
