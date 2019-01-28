import numpy as np
from DatasetUtils.BaseDataset import BaseDataset


class MsSegmentationDataset(BaseDataset):

    def _get_path_from_info_row(self, data_dir, info_row):
        slice_path = {"data": str.format("{0}/data_p{1}_t{2}_s{3}.npy", data_dir, info_row["patient"],
                                         info_row["time"], info_row["slice"]),
                      "mask": str.format("{0}/mask_p{1}_t{2}_s{3}.npy", data_dir, info_row["patient"],
                                         info_row["time"], info_row["slice"])}

        return slice_path

    def _get_data(self, info_row):
        data = np.load(info_row["path"]["data"])

        if self.settings.crop:
            data = self.settings.data_utils.crop_center(data)

        return data

    def _get_label(self, info_row):
        mask = np.load(info_row["path"]["mask"])

        if self.settings.crop:
            mask = self.settings.data_utils.crop_center(mask)

        mask = mask[:, :, 0]  # 1st expert mask
        mask = np.expand_dims(mask, 2)

        return mask

    def save_tested_data(self, info_row, is_true_classified, title=""):
        pass
