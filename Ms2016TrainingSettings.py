import os
from MsBaseTrainingSettings import MsBaseTrainingSettings
from MsMaskTypes import MsMaskTypes as MaskTypes
from Ms2016Dataset import Ms2016Dataset as Dataset


class Ms2016TrainingSettings(MsBaseTrainingSettings):

    def __init__(self):

        # 2016 dataset relative settings
        self.data_folder = "../../Datasets/MMSEG_MICCAI2016/PreprocessedTrainingData/axial_slices_dilation_0.3_label_0.3"
        self.data_definition_file_name = "ms_info.json"
        self.data_definition_file_path = os.path.join(self.data_folder, self.data_definition_file_name)

        # 2016 dataset relative preprocessing settings
        self.filters = {"mask_0": {"min_open": 10}}
        self.training_mask_type = MaskTypes.INTERSECTION_DILATED

        # 2016 dataset relative postprocessing settings
        self.find_thr_mask_type = MaskTypes.INTERSECTION
        self.calculate_metrics_mask_types = [MaskTypes.INTERSECTION,
                                             MaskTypes.STAPLE,
                                             MaskTypes.EXPERT_1,
                                             MaskTypes.EXPERT_2,
                                             MaskTypes.EXPERT_3,
                                             MaskTypes.EXPERT_4,
                                             MaskTypes.EXPERT_5,
                                             MaskTypes.EXPERT_6,
                                             MaskTypes.EXPERT_7]

        # 2016 dataset relative model compilation settings
        self.input_shape = (216, 180, 5)

        # 2016 dataset relative model training settings
        self.epochs = 3000
        self.steps_per_epoch = 16
        self.batch_size = 32
        self.val_steps = 8

        super(Ms2016TrainingSettings, self).__init__()
        self.dataset = Dataset(self)
