import os
import numpy as np
from skimage import morphology
from scipy import ndimage as nd


class MsUtils(object):

    def __init__(self, settings):
        self.settings = settings

    def get_mask2015_paths(self):

        mask_paths = []
        template = list("training0X_0X_maskX.npy")
        for patient in self.settings.cases:
            template[9] = str(patient[0])
            for time in patient[1]:
                template[12] = str(time)

                template[18] = '1'
                path1 = self.settings.masks + "".join(template)

                template[18] = '2'
                path2 = self.settings.masks + "".join(template)

                mask_paths.append([path1, path2])

        return mask_paths

    def get_data2015_paths(self):

        num_modalities = len(self.settings.modalities)
        templates = [list("Person0X_Time0X_FLAIR.npy"),
                     list("Person0X_Time0X_MPRAGE.npy"),
                     list("Person0X_Time0X_PD.npy"),
                     list("Person0X_Time0X_T2.npy")]
        templates = [templates[modality] for modality in self.settings.modalities]

        data_paths = []
        for patient in self.settings.cases:
            for i in range(num_modalities):
                templates[i][7:8] = str(patient[0])
            for time in patient[1]:
                for i in range(num_modalities):
                    templates[i][14:15] = str(time)
                paths = [self.settings.data + "".join(template) for template in templates]
                data_paths.append(paths)

        return data_paths

    def get_mask2016_paths(self):

        paths_masks = []
        template = list("ManualSegmentation_X.nii.gz")
        for case in self.settings.cases:
            paths_for_case = []
            for expert in range(1, self.settings.experts_num + 1):
                template[19] = str(expert)
                paths_for_case.append(os.path.join(self.settings.masks, case, "".join(template)))

            paths_masks.append(paths_for_case)

        return paths_masks

    def get_data2016_paths(self):

        names = ["DP_preprocessed.nii.gz",
                 "FLAIR_preprocessed.nii.gz",
                 "GADO_preprocessed.nii.gz",
                 "T1_preprocessed.nii.gz",
                 "T2_preprocessed.nii.gz"]
        names = [names[modality] for modality in self.settings.modalities]

        paths_data = []
        for case in self.settings.cases:
            paths = [os.path.join(self.settings.data, case, name) for name in names]
            paths_data.append(paths)

        return paths_data

    def dilate_mask(self, mask, data):

        dilation_thr = self.settings.dilation_thr
        data_thr = self.settings.initial_data_thr
        mask_constructed = False
        mask_dilated = nd.binary_dilation(mask, iterations=self.settings.dilation_iterations)
        while not mask_constructed:
            mask_candidates = data > data_thr
            mask_dilated_cur = mask_dilated - mask
            mask_dilated_cur = np.logical_and(mask_candidates, mask_dilated_cur)
            mask_dilated_cur = mask_dilated_cur.astype(np.byte)

            if self.settings.connected_to_original_mask:
                mask_dilated_labeled, labels_num = morphology.label(np.logical_or(mask, mask_dilated_cur),
                                                                    connectivity=1, return_num=True)
                connected_to_original_mask_labeled = np.multiply(mask_dilated_labeled, mask)
                for lbl in range(1, labels_num + 1):
                    if np.sum(connected_to_original_mask_labeled == lbl) == 0:
                        mask_dilated_cur[mask_dilated_labeled == lbl] = 0

            ratio = np.sum(mask_dilated_cur) / np.sum(mask)
            if ratio < dilation_thr:
                data_thr = data_thr - 0.01
            else:
                mask_dilated = mask_dilated_cur
                mask_constructed = True

        mask = mask + self.settings.soft_label * mask_dilated
        return mask
