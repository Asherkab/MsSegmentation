import os
import numpy as np
import pandas as pd
from MsData2016PreparationSettings import MsDataPreparationSettings as Settings


# Initialize settings and utils
settings = Settings()
logger = settings.logger
data_utils = settings.data_utils
ms_utils = settings.ms_utils

logger.start(settings.output_logs_path)  # initialize logger

# Get paths of data and masks
mask_paths = ms_utils.get_mask2016_paths()
data_paths = ms_utils.get_data2016_paths()

# Iterate over cases
ms_info = pd.DataFrame()  # create empty MS info table
for patient_idx in range(len(data_paths)):

    data = [ms_utils.load_and_rotate_nifti(path) for path in data_paths[patient_idx]]  # load all data modalities
    data = [data_utils.resize(vol, interpolation_order=1) for vol in data]  # resize data volumes

    # Normalize data
    normalized_data = []
    for mod in data:
        settings.min_clip_value = np.min(mod)
        settings.max_clip_value = np.max(mod)
        normalized_mod = data_utils.clip_and_normalize(mod)
        normalized_data.append(normalized_mod)
    data = normalized_data

    masks = [ms_utils.load_and_rotate_nifti(path) for path in mask_paths[patient_idx]]  # load all experts masks
    masks = [data_utils.resize(vol, interpolation_order=0) for vol in masks]  # resize masks volumes

    # Flip masks that are in different orientation
    masks[1] = np.flip(masks[1], axis=1)
    masks[1] = np.flip(masks[1], axis=2)

    masks[3] = np.flip(masks[3], axis=1)
    masks[3] = np.flip(masks[3], axis=2)

    # Set MS lesions label to be 1
    binary_masks = []
    for mask in masks:
        binary_mask = mask
        binary_mask[mask > 0] = 1
        binary_masks.append(binary_mask)
    masks = binary_masks

    # Calculate inter-rater metrics
    if settings.calculate_inter_rater_metrics:
        dice_matrix = np.zeros(shape=(settings.experts_num, settings.experts_num))
        agreement_matrix = np.zeros(shape=(settings.experts_num, settings.experts_num))
        for row in range(settings.experts_num):
            for col in range(settings.experts_num):
                dice_matrix[row, col] = settings.metrics_container.dice_coef(masks[row], masks[col])
                agreement_matrix[row, col] = settings.metrics_container.precision(masks[row], masks[col])

        logger.log("\nDice Matrix:")
        logger.log(dice_matrix)
        logger.log("\nAgreement Matrix:")
        logger.log(agreement_matrix)

    # Create dilated masks
    dilated_masks = [ms_utils.dilate_mask(mask, data[1]) for mask in masks]
    masks = masks + dilated_masks

    # Iterate over slices
    for slice_idx in range(data[0].shape[0]):  # axial

        # Create data and mask sample
        data_sample = [modality_volume[slice_idx, :, :] for modality_volume in normalized_data]
        data_sample = np.moveaxis(np.array(data_sample), 0, -1)

        mask_sample = [mask[slice_idx, :, :] for mask in masks]
        mask_sample = np.moveaxis(np.array(mask_sample), 0, -1)

        # Plot examples of masks
        if settings.plot_examples and 100 < slice_idx < 120:

            settings.plots.overlay_plot(data_sample[:, :, 1], overlay_1=mask_sample[:, :, 0],
                                        name="_single_mask{0}_{1}".format(0, slice_idx))

            for mask_idx in range(1, settings.experts_num):
                settings.plots.overlay_plot(data_sample[:, :, 1], overlay_1=mask_sample[:, :, mask_idx],
                                            name="_single_mask{0}_{1}".format(mask_idx, slice_idx))

                settings.plots.overlay_plot(data_sample[:, :, 1], overlay_1=mask_sample[:, :, 0],
                                            overlay_2=mask_sample[:, :, mask_idx],
                                            name="_comparison{0}_{1}".format(mask_idx, slice_idx))

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

