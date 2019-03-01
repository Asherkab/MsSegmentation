import numpy as np
from MsMaskTypes import MsMaskTypes as MaskTypes
from MsBaseDataset import MsBaseDataset


class Ms2016Dataset(MsBaseDataset):

    def __init__(self, settings):
        super(Ms2016Dataset, self).__init__(settings)

    def _get_path_from_info_row(self, data_dir, info_row):
        slice_path = {"data": str.format("{0}/data_p{1}_s{2}.npy", data_dir, info_row["patient"],
                                         info_row["slice"]),
                      "mask": str.format("{0}/mask_p{1}_s{2}.npy", data_dir, info_row["patient"],
                                         info_row["slice"])}

        return slice_path

    def _get_label(self, info_row):
        mask = np.load(info_row["path"]["mask"])

        if self.settings.crop:
            mask = self.settings.data_utils.crop_center(mask)

        if self.settings.training_mask_type == MaskTypes.STAPLE:
            mask = mask[:, :, 0]
        elif self.settings.training_mask_type == MaskTypes.STAPLE_DILATED:
            mask = mask[:, :, 9]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_1:
            mask = mask[:, :, 1]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2:
            mask = mask[:, :, 2]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_3:
            mask = mask[:, :, 3]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_4:
            mask = mask[:, :, 4]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_5:
            mask = mask[:, :, 5]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_6:
            mask = mask[:, :, 6]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_7:
            mask = mask[:, :, 7]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_1_DILATED:
            mask = mask[:, :, 10]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2_DILATED:
            mask = mask[:, :, 11]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_3_DILATED:
            mask = mask[:, :, 12]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_4_DILATED:
            mask = mask[:, :, 13]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_5_DILATED:
            mask = mask[:, :, 14]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_6_DILATED:
            mask = mask[:, :, 15]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_7_DILATED:
            mask = mask[:, :, 16]
        elif self.settings.training_mask_type == MaskTypes.INTERSECTION:
            mask = mask[:, :, 8]
        elif self.settings.training_mask_type == MaskTypes.INTERSECTION_DILATED:
            mask = mask[:, :, 17]
        elif self.settings.training_mask_type == MaskTypes.UNION:
            temp_mask = np.logical_or(mask[:, :, 1], mask[:, :, 2])
            temp_mask = np.logical_or(temp_mask, mask[:, :, 3])
            temp_mask = np.logical_or(temp_mask, mask[:, :, 4])
            temp_mask = np.logical_or(temp_mask, mask[:, :, 5])
            temp_mask = np.logical_or(temp_mask, mask[:, :, 6])
            mask = np.logical_or(temp_mask, mask[:, :, 7])

        mask = np.expand_dims(mask, 2)

        return mask
