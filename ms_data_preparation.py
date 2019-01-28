import os
import numpy as np
import pandas as pd

from MsDataPreparationSettings import MsDataPreparationSettings as Settings

# Initialize settings and utils
settings = Settings()
utils = settings.utils

# Get paths of data and masks
mask_paths = utils.get_mask_paths()
data_paths = utils.get_data_paths()

# Iterate over cases
ms_info = pd.DataFrame()  # create empty MS info table
for patient_idx in range(len(data_paths)):

    # Load different modalities and experts masks for case
    data = [np.load(path) for path in data_paths[patient_idx]]
    masks = [np.load(path) for path in mask_paths[patient_idx]]

    # Iterate over slices
    for slice_idx in range(len(data[0])):  # axial

        # Get patient and time from data path
        name = os.path.basename(data_paths[patient_idx][0])
        patient = int(name[6:8])
        time = int(name[13:15])

        # Create data and mask sample
        data_sample = [modality_volume[slice_idx, :, :] for modality_volume in data]
        data_sample = np.moveaxis(np.array(data_sample), 0, -1)

        mask_sample = [expert_mask[slice_idx, :, :] for expert_mask in masks]
        mask_sample = np.moveaxis(np.array(mask_sample), 0, -1)

        # Save data and mask sample
        data_name = "data_p{0}_t{1}_s{2}.npy".format(patient, time, slice_idx)
        np.save(os.path.join(settings.data_folder, data_name), data_sample)

        mask_name = "mask_p{0}_t{1}_s{2}.npy".format(patient, time, slice_idx)
        np.save(os.path.join(settings.data_folder, mask_name), mask_sample)

        # Fill info row
        ms_info_row = dict()
        ms_info_row["patient"] = patient
        ms_info_row["time"] = time
        ms_info_row["slice"] = slice_idx
        ms_info_row["expert_1"] = np.sum(mask_sample[:, :, 0])
        ms_info_row["expert_2"] = np.sum(mask_sample[:, :, 1])
        ms_info_row["intersection"] = np.sum(np.logical_and(mask_sample[:, :, 0], mask_sample[:, :, 1]))
        ms_info_row["union"] = np.sum(np.logical_or(mask_sample[:, :, 0], mask_sample[:, :, 1]))

        ms_info = ms_info.append(ms_info_row, ignore_index=True)  # add sample info to table

ms_info.to_json(settings.data_definition_file_path)  # save MS info table
