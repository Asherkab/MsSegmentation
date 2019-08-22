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
        full_mask = np.load(info_row["path"]["mask"])

        if self.settings.crop:
            full_mask = self.settings.data_utils.crop_center(full_mask)

        mask = np.zeros_like(full_mask[:, :, 0])
        if self.settings.training_mask_type == MaskTypes.LOP_STAPLE:
            mask = full_mask[:, :, 0]
        elif self.settings.training_mask_type == MaskTypes.LOP_STAPLE_DILATED:
            mask = full_mask[:, :, 10]
        elif self.settings.training_mask_type == MaskTypes.STAPLE:
            mask = full_mask[:, :, 9]
        elif self.settings.training_mask_type == MaskTypes.STAPLE_DILATED:
            mask = full_mask[:, :, 19]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_1:
            mask = full_mask[:, :, 1]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2:
            mask = full_mask[:, :, 2]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_3:
            mask = full_mask[:, :, 3]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_4:
            mask = full_mask[:, :, 4]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_5:
            mask = full_mask[:, :, 5]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_6:
            mask = full_mask[:, :, 6]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_7:
            mask = full_mask[:, :, 7]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_1_DILATED:
            mask = full_mask[:, :, 11]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_2_DILATED:
            mask = full_mask[:, :, 12]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_3_DILATED:
            mask = full_mask[:, :, 13]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_4_DILATED:
            mask = full_mask[:, :, 14]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_5_DILATED:
            mask = full_mask[:, :, 15]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_6_DILATED:
            mask = full_mask[:, :, 16]
        elif self.settings.training_mask_type == MaskTypes.EXPERT_7_DILATED:
            mask = full_mask[:, :, 17]
        elif self.settings.training_mask_type == MaskTypes.INTERSECTION:
            mask = full_mask[:, :, 8]
        elif self.settings.training_mask_type == MaskTypes.INTERSECTION_DILATED:
            mask = full_mask[:, :, 18]
        elif self.settings.training_mask_type == MaskTypes.UNION:
            temp_mask = np.logical_or(full_mask[:, :, 1], full_mask[:, :, 2])
            temp_mask = np.logical_or(temp_mask, full_mask[:, :, 3])
            temp_mask = np.logical_or(temp_mask, full_mask[:, :, 4])
            temp_mask = np.logical_or(temp_mask, full_mask[:, :, 5])
            temp_mask = np.logical_or(temp_mask, full_mask[:, :, 6])
            mask = np.logical_or(temp_mask, full_mask[:, :, 7])
        elif self.settings.training_mask_type == MaskTypes.AVERAGE:
            temp_mask = full_mask[:, :, 1] + full_mask[:, :, 2]
            temp_mask += full_mask[:, :, 3]
            temp_mask += full_mask[:, :, 4]
            temp_mask += full_mask[:, :, 5]
            temp_mask += full_mask[:, :, 6]
            temp_mask += full_mask[:, :, 7]
            mask = temp_mask / 7
        elif self.settings.training_mask_type == MaskTypes.AVERAGE_DILATED:
            temp_mask = full_mask[:, :, 11] + full_mask[:, :, 12]
            temp_mask += full_mask[:, :, 13]
            temp_mask += full_mask[:, :, 14]
            temp_mask += full_mask[:, :, 15]
            temp_mask += full_mask[:, :, 16]
            temp_mask += full_mask[:, :, 17]
            mask = temp_mask / 7

        mask = np.expand_dims(mask, 2)

        return mask
