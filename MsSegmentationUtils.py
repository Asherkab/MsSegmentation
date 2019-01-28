class MsSegmentationUtils(object):

    def __init__(self, settings):
        self.settings = settings

    def get_mask_paths(self):

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

    def get_data_paths(self):

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
