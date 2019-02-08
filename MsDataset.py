import numpy as np
from DatasetUtils.BaseDataset import BaseDataset
from MsMaskTypes import MsMaskTypes as MaskTypes


class MsDataset(BaseDataset):

    def __init__(self, settings):
        super(MsDataset, self).__init__(settings)

        self.expert_1_dice_list = []
        self.expert_1_precision_list = []
        self.expert_1_recall_list = []

        self.expert_2_dice_list = []
        self.expert_2_precision_list = []
        self.expert_2_recall_list = []

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

    def apply_postprocessing(self,
                             test_predictions,
                             test_data,
                             train_predictions,
                             train_data,
                             fold_info):

        # find optimal threshold or use predefined one
        if self.settings.find_opt_thr:

            self.settings.training_mask_type = self.settings.find_thr_mask_type

            find_thr_train_masks = []
            for _, info_row in fold_info["train_info"].iterrows():
                mask = self._get_label(info_row)
                find_thr_train_masks.append(mask)
            find_thr_train_masks = np.array(find_thr_train_masks)

            opt_thr = self.settings.metrics_container.get_opt_thr_dice(y_pred=np.array(train_predictions),
                                                                       y_true=find_thr_train_masks,
                                                                       thrs=self.settings.thrs_to_check)

            find_thr_test_masks = []
            for _, info_row in fold_info["test_info"].iterrows():
                mask = self._get_label(info_row)
                find_thr_test_masks.append(mask)
            find_thr_test_masks = np.array(find_thr_test_masks)
            test_opt_thr = self.settings.metrics_container.get_opt_thr_dice(y_pred=np.array(test_predictions),
                                                                            y_true=find_thr_test_masks,
                                                                            thrs=self.settings.thrs_to_check)

            self.settings.logger.log("\nOptimal threshold that maximizes DICE of:")
            self.settings.logger.log(" - training data = {0}".format(opt_thr))
            self.settings.logger.log(" - test data = {0}".format(test_opt_thr))

        else:
            opt_thr = self.settings.opt_thr

        thresholded_test_predictions = np.greater(test_predictions, opt_thr)  # apply threshold on test data
        thresholded_test_predictions = thresholded_test_predictions.astype(np.byte)

        return thresholded_test_predictions

    def calculate_fold_metrics(self,
                               test_predictions,
                               test_data,
                               test_evaluations,
                               train_predictions,
                               train_data,
                               fold_info):

        expert_1_test_masks = []
        self.settings.training_mask_type = MaskTypes.EXPERT_1
        for _, info_row in fold_info["test_info"].iterrows():
            mask = self._get_label(info_row)
            expert_1_test_masks.append(mask)
        expert_1_test_masks = np.array(expert_1_test_masks)

        dice = self.settings.metrics_container.dice_coef(expert_1_test_masks, test_predictions)
        self.expert_1_dice_list.append(dice)
        precision = self.settings.metrics_container.precision(expert_1_test_masks, test_predictions)
        self.expert_1_precision_list.append(precision)
        recall = self.settings.metrics_container.recall(expert_1_test_masks, test_predictions)
        self.expert_1_recall_list.append(recall)

        self.settings.logger.log("\nMetrics that are measured relative to expert 1 masks:")
        self.settings.logger.log("- dice = " + str(dice))
        self.settings.logger.log("- precision = " + str(precision))
        self.settings.logger.log("- recall = " + str(recall))

        expert_2_test_masks = []
        self.settings.training_mask_type = MaskTypes.EXPERT_2
        for _, info_row in fold_info["test_info"].iterrows():
            mask = self._get_label(info_row)
            expert_2_test_masks.append(mask)
        expert_2_test_masks = np.array(expert_2_test_masks)

        dice = self.settings.metrics_container.dice_coef(expert_2_test_masks, test_predictions)
        self.expert_2_dice_list.append(dice)
        precision = self.settings.metrics_container.precision(expert_2_test_masks, test_predictions)
        self.expert_2_precision_list.append(precision)
        recall = self.settings.metrics_container.recall(expert_2_test_masks, test_predictions)
        self.expert_2_recall_list.append(recall)

        self.settings.logger.log("\nMetrics that are measured relative to expert 2 masks:")
        self.settings.logger.log("- dice = " + str(dice))
        self.settings.logger.log("- precision = " + str(precision))
        self.settings.logger.log("- recall = " + str(recall))

    def save_tested_data(self, test_info):
        pass

    def log_metrics(self):
        self.settings.logger.log("\nMetrics that are measured relative to expert 1 masks:")
        self.settings.logger.log(" - dice mean: " + str(np.mean(self.expert_1_dice_list)))
        self.settings.logger.log(" - dice std: " + str(np.std(self.expert_1_dice_list)))
        self.settings.logger.log(" - recall mean: " + str(np.mean(self.expert_1_recall_list)))
        self.settings.logger.log(" - recall std: " + str(np.std(self.expert_1_recall_list)))
        self.settings.logger.log(" - precision mean: " + str(np.mean(self.expert_1_precision_list)))
        self.settings.logger.log(" - precision std: " + str(np.std(self.expert_1_precision_list)))

        self.settings.logger.log("\nMetrics that are measured relative to expert 2 masks:")
        self.settings.logger.log(" - dice mean: " + str(np.mean(self.expert_2_dice_list)))
        self.settings.logger.log(" - dice std: " + str(np.std(self.expert_2_dice_list)))
        self.settings.logger.log(" - recall mean: " + str(np.mean(self.expert_2_recall_list)))
        self.settings.logger.log(" - recall std: " + str(np.std(self.expert_2_recall_list)))
        self.settings.logger.log(" - precision mean: " + str(np.mean(self.expert_2_precision_list)))
        self.settings.logger.log(" - precision std: " + str(np.std(self.expert_2_precision_list)))
