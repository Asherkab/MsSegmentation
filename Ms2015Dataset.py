import numpy as np
from MsMaskTypes import MsMaskTypes as MaskTypes
from MsBaseDataset import MsBaseDataset


class Ms2015Dataset(MsBaseDataset):

    def __init__(self, settings):
        super(Ms2015Dataset, self).__init__(settings)

    def _get_path_from_info_row(self, data_dir, info_row):
        slice_path = {"data": str.format("{0}/data_p{1}_t{2}_s{3}.npy", data_dir, info_row["patient"],
                                         info_row["time"], info_row["slice"]),
                      "mask": str.format("{0}/mask_p{1}_t{2}_s{3}.npy", data_dir, info_row["patient"],
                                         info_row["time"], info_row["slice"])}

        return slice_path

    def _get_label(self, info_row):
        mask = np.load(info_row["path"]["mask"])

        if self.settings.crop:
            mask = self.settings.data_utils.crop_center(mask)

        if self.settings.training_mask_type == MaskTypes.EXPERT_1:
            mask = mask[:, :, 0]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2:
            mask = mask[:, :, 1]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_1_DILATED:
            mask = mask[:, :, 2]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2_DILATED:
            mask = mask[:, :, 3]
        elif self.settings.training_mask_type == MaskTypes.INTERSECTION:
            mask = np.logical_and(mask[:, :, 0], mask[:, :, 1])
        elif self.settings.training_mask_type == MaskTypes.UNION:
            mask = np.logical_or(mask[:, :, 0], mask[:, :, 1])

        mask = np.expand_dims(mask, 2)

        return mask
