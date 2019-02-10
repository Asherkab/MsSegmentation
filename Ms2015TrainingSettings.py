import os
from MsBaseTrainingSettings import MsBaseTrainingSettings
from MsMaskTypes import MsMaskTypes as MaskTypes
from Ms2015Dataset import Ms2015Dataset as Dataset


class Ms2015TrainingSettings(MsBaseTrainingSettings):

    def __init__(self):

        # 2015 dataset relative settings
        self.data_folder = "../../Datasets/MS_Longitudinal_ISBI2015/training/axial_slices/"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        # 2015 dataset relative preprocessing settings
        self.filters = {"expert_1": {"min_open": 1}}
        self.training_mask_type = MaskTypes.EXPERT_1

        # 2015 dataset relative postprocessing settings
        self.find_thr_mask_type = MaskTypes.EXPERT_1
        self.calculate_metrics_mask_types = [MaskTypes.EXPERT_1,
                                             MaskTypes.EXPERT_2]

        # 2015 dataset relative model compilation settings
        self.input_shape = (216, 180, 4)

        # 2015 dataset relative model training settings
        self.epochs = 1500
        self.steps_per_epoch = 32
        self.batch_size = 32
        self.val_steps = 16


        super(Ms2015TrainingSettings, self).__init__()
        self.dataset = Dataset(self)
