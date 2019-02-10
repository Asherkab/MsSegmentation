import abc
import numpy as np
from DatasetUtils.BaseDataset import BaseDataset


class MsBaseDataset(BaseDataset):

    def __init__(self, settings):
        super(MsBaseDataset, self).__init__(settings)

        self.dice_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.precision_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.recall_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))

    @abc.abstractmethod
    def _get_path_from_info_row(self, data_dir, info_row):
        pass

    def _get_data(self, info_row):
        data = np.load(info_row["path"]["data"])

        if self.settings.crop:
            data = self.settings.data_utils.crop_center(data)

        return data

    @abc.abstractmethod
    def _get_label(self, info_row):
        pass

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
                               fold,
                               fold_info):

        for mask_idx, mask_type in enumerate(self.settings.calculate_metrics_mask_types):
            test_masks = []
            self.settings.training_mask_type = mask_type
            for _, info_row in fold_info["test_info"].iterrows():
                mask = self._get_label(info_row)
                test_masks.append(mask)
            test_masks = np.array(test_masks)

            dice = self.settings.metrics_container.dice_coef(test_masks, test_predictions)
            self.dice_matrix[fold][mask_idx] = dice
            precision = self.settings.metrics_container.precision(test_masks, test_predictions)
            self.precision_matrix[fold][mask_idx] = precision
            recall = self.settings.metrics_container.recall(test_masks, test_predictions)
            self.recall_matrix[fold][mask_idx] = recall

            self.settings.logger.log("\nMetrics that are measured relative to expert {0} mask:".format(mask_idx))
            self.settings.logger.log("- dice = " + str(dice))
            self.settings.logger.log("- precision = " + str(precision))
            self.settings.logger.log("- recall = " + str(recall))

    def save_tested_data(self, test_info):
        pass

    def log_metrics(self):

        for mask_idx in range(len(self.settings.calculate_metrics_mask_types)):
            self.settings.logger.log("\nMetrics that are measured relative to expert {0} masks:".format(mask_idx))
            self.settings.logger.log(" - dice mean: " + str(np.mean(self.dice_matrix[:, mask_idx])))
            self.settings.logger.log(" - dice std: " + str(np.std(self.dice_matrix[:, mask_idx])))
            self.settings.logger.log(" - recall mean: " + str(np.mean(self.recall_matrix[:, mask_idx])))
            self.settings.logger.log(" - recall std: " + str(np.std(self.recall_matrix[:, mask_idx])))
            self.settings.logger.log(" - precision mean: " + str(np.mean(self.precision_matrix[:, mask_idx])))
            self.settings.logger.log(" - precision std: " + str(np.std(self.precision_matrix[:, mask_idx])))
