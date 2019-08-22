import os
import numpy as np
import pandas as pd
from MsData2016PreparationSettings import MsDataPreparationSettings as Settings


# Initialize settings and utils
settings = Settings()
logger = settings.logger
data_utils = settings.data_utils
ms_utils = settings.ms_utils

# Get paths of data and masks
mask_paths = ms_utils.get_mask2016_paths()
data_paths = ms_utils.get_data2016_paths()

logger.start(settings.output_logs_path)  # initialize logger

mean_dice_matrix = np.zeros(shape=(7, 7))

# Iterate over cases
ms_info = pd.DataFrame()  # create empty MS info table
for patient_idx in range(len(data_paths)):

    data = [ms_utils.load_and_rotate_nifti(path) for path in data_paths[patient_idx]]  # load all data modalities

    # Normalize data
    normalized_data = []
    for mod_idx, mod in enumerate(data):
        flatten_mod = mod.flatten()
        flatten_mod.sort()
        one_percent_idx = int(flatten_mod.shape[0] / 1000)
        settings.min_clip_value = flatten_mod[one_percent_idx]
        settings.max_clip_value = flatten_mod[-one_percent_idx]
        normalized_mod = data_utils.clip_and_normalize(mod)
        normalized_data.append(normalized_mod)
    data = normalized_data

    masks = [ms_utils.load_and_rotate_nifti(path) for path in mask_paths[patient_idx]]  # load all experts masks

    # Set MS lesions label to be 1
    binary_masks = []
    for mask in masks:
        binary_mask = mask
        binary_mask[mask > 0] = 1
        binary_masks.append(binary_mask)
    masks = binary_masks

    # Create intersection mask
    intersection_mask = np.logical_and(masks[1], masks[2])
    for mask_idx in range(3, len(masks)):
        intersection_mask = np.logical_and(intersection_mask, masks[mask_idx])
    intersection_mask = intersection_mask.astype(np.byte)
    masks = masks + [intersection_mask]

    # Create STAPLE mask
    staple_mask, sensitivity, specificity = ms_utils.staple([masks[1], masks[2], masks[3], masks[4], masks[5],
                                                             masks[6], masks[7]], weights_=None)
    masks = masks + [staple_mask]

    # Calculate inter-rater metrics
    if settings.calculate_inter_rater_metrics:
        dice_matrix = np.zeros(shape=(settings.experts_num, settings.experts_num))
        agreement_matrix = np.zeros(shape=(settings.experts_num, settings.experts_num))
        for row in range(settings.experts_num):
            for col in range(settings.experts_num):
                dice_matrix[row, col] = settings.metrics_container.dice_coef(masks[row], masks[col])
                agreement_matrix[row, col] = settings.metrics_container.precision(masks[row], masks[col])

        mean_dice_matrix += dice_matrix

        logger.log("\n\nPatient {0}".format(patient_idx))
        logger.log("\nDice Matrix:")
        logger.log(dice_matrix)
        logger.log("\nAgreement Matrix:")
        logger.log(agreement_matrix)

    # Create dilated masks
    dilated_masks = [ms_utils.dilate_mask(mask, data[1]) for mask in masks]
    masks = masks + dilated_masks

    # Create STAPLE mask from dilated raters' masks
    staple_mask, sensitivity, specificity = ms_utils.soft_staple_simplified([masks[11], masks[12], masks[13], masks[14],
                                                                             masks[15], masks[16], masks[17]])
    staple_mask[staple_mask < 0.01] = 0
    masks[-1] = staple_mask

    # Iterate over slices
    for slice_idx in range(data[0].shape[2]):  # axial

        # Create data and mask sample
        data_sample = [modality_volume[slice_idx, :, :] for modality_volume in data]
        data_sample = np.moveaxis(np.array(data_sample), 0, -1)

        mask_sample = [mask[slice_idx, :, :] for mask in masks]
        mask_sample = np.moveaxis(np.array(mask_sample), 0, -1)

        if settings.plot_examples and 80 < slice_idx < 140:
            settings.plots.overlay_plot(data_sample[:, :, 1], overlay_1=mask_sample[:, :, 1],
                                        overlay_2=mask_sample[:, :, 9],
                                        name="{0}_{1}".format(patient_idx, slice_idx))

        # Save data and mask sample
        data_name = "data_p{0}_s{2}.npy".format(patient_idx, 0, slice_idx)
        np.save(os.path.join(settings.data_folder, data_name), data_sample)

        mask_name = "mask_p{0}_s{2}.npy".format(patient_idx, 0, slice_idx)
        np.save(os.path.join(settings.data_folder, mask_name), mask_sample)

        # Fill info row
        ms_info_row = dict()
        ms_info_row["patient"] = patient_idx
        ms_info_row["slice"] = slice_idx

        for mask_idx in range(settings.experts_num):
            ms_info_row["mask_{0}".format(mask_idx)] = np.sum(mask_sample[:, :, mask_idx])

        for mask_idx in range(settings.experts_num, settings.experts_num * 2):
            ms_info_row["dilated_mask_{0}".format(mask_idx - settings.experts_num)] = np.sum(mask_sample
                                                                                             [:, :, mask_idx])

        ms_info = ms_info.append(ms_info_row, ignore_index=True)  # add sample info to table

ms_info.to_json(settings.data_definition_file_path)  # save MS info table
logger.end()  # close logger
