import abc
import cv2
import numpy as np
from skimage import morphology
from DatasetUtils.BaseDataset import BaseDataset


class MsBaseDataset(BaseDataset):

    def __init__(self, settings):
        super(MsBaseDataset, self).__init__(settings)

        self.dice_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.precision_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.recall_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.tpr_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))
        self.fpr_matrix = np.zeros(shape=(settings.folds, len(settings.calculate_metrics_mask_types)))

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

    def modify(self, data, label):

        rot_angle = np.random.uniform(-self.settings.rotation, self.settings.rotation)
        rot_matrix = cv2.getRotationMatrix2D((90, 108), rot_angle, 1)

        rotated_data = np.zeros_like(data)
        for idx in range(4):
            rotated_data[:, :, idx] = cv2.warpAffine(data[:, :, idx], rot_matrix, (180, 216))

        rotated_label = np.zeros_like(label)
        rotated_label[:, :, 0] = cv2.warpAffine(label[:, :, 0].astype(np.float32), rot_matrix, (180, 216),
                                                flags=cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS)

        return rotated_data, rotated_label

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

        test_predictions = np.greater(test_predictions, opt_thr)  # apply threshold on test predictions
        test_predictions = morphology.remove_small_objects(test_predictions, self.settings.small_objects)
        test_predictions = test_predictions.astype(np.byte)

        # Conditional dilation postprocessing
        # flair = np.array([data_slice[:, :, 1] for data_slice in test_data[0]])  # get flair test data
        # mask = np.zeros(shape=test_predictions.shape)
        # dilation_iterations = self.settings.dilation_iterations
        # for slice_idx in range(test_predictions.shape[0]):
        #     mask_slice = test_predictions[slice_idx, :, :, 0]
        #     if np.sum(mask_slice) > 0:
        #         mask_constructed = False
        #         while not mask_constructed:
        #             mask_dilated = ndimage.binary_dilation(mask_slice, iterations=dilation_iterations)
        #             mask_candidates = flair[slice_idx, :, :] > self.settings.data_thr
        #             mask_dilated_cur = mask_dilated - mask_slice
        #             mask_dilated_cur = np.logical_and(mask_candidates, mask_dilated_cur)
        #             mask_dilated_cur = mask_dilated_cur.astype(np.byte)
        #
        #             mask_dilated_labeled, labels_num = morphology.label(np.logical_or(mask_slice,
        #                                                                               mask_dilated_cur),
        #                                                                 connectivity=1, return_num=True)
        #             connected_to_original_mask_labeled = np.multiply(mask_dilated_labeled, mask_slice)
        #             for lbl in range(1, labels_num + 1):
        #                 if np.sum(connected_to_original_mask_labeled == lbl) == 0:
        #                     mask_dilated_cur[mask_dilated_labeled == lbl] = 0
        #
        #             ratio = np.sum(mask_dilated_cur) / np.sum(mask_slice)
        #             if ratio < self.settings.dilation_thr and dilation_iterations < 10:
        #                 dilation_iterations = dilation_iterations + 1
        #             else:
        #                 mask[slice_idx, :, :, 0] = mask_slice + mask_dilated_cur
        #
        #                 dilation_iterations = self.settings.dilation_iterations
        #                 mask_constructed = True
        #
        # test_predictions = mask
        # test_predictions = test_predictions.astype(np.byte)

        return test_predictions

    def calculate_fold_metrics(self,
                               test_predictions,
                               test_data,
                               test_evaluations,
                               train_predictions,
                               train_data,
                               fold,
                               fold_info):

        test_predictions = test_predictions.squeeze()
        for mask_idx, mask_type in enumerate(self.settings.calculate_metrics_mask_types):
            self.settings.logger.log("\nMetrics that are measured relative to expert {0} mask:".format(mask_idx))

            test_masks = []
            self.settings.training_mask_type = mask_type
            for _, info_row in fold_info["test_info"].iterrows():
                mask = self._get_label(info_row)
                test_masks.append(mask)
            test_masks = np.array(test_masks)
            test_masks = test_masks.squeeze()

            dice = self.settings.metrics_container.dice_coef(test_masks, test_predictions)
            self.dice_matrix[fold][mask_idx] = dice
            precision = self.settings.metrics_container.precision(test_masks, test_predictions)
            self.precision_matrix[fold][mask_idx] = precision
            recall = self.settings.metrics_container.recall(test_masks, test_predictions)
            self.recall_matrix[fold][mask_idx] = recall
            tpr = self.settings.metrics_container.lesion_tpr(test_masks, test_predictions,
                                                             self.settings.minimum_overlap)
            self.tpr_matrix[fold][mask_idx] = tpr
            fpr = self.settings.metrics_container.lesion_fpr(test_masks, test_predictions,
                                                             self.settings.minimum_overlap)
            self.fpr_matrix[fold][mask_idx] = fpr

            self.settings.logger.log("- dice = " + str(dice))
            self.settings.logger.log("- precision = " + str(precision))
            self.settings.logger.log("- recall = " + str(recall))
            self.settings.logger.log("- tpr = " + str(tpr))
            self.settings.logger.log("- fpr = " + str(fpr))

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
            self.settings.logger.log(" - tpr mean: " + str(np.mean(self.tpr_matrix[:, mask_idx])))
            self.settings.logger.log(" - tpr std: " + str(np.std(self.tpr_matrix[:, mask_idx])))
            self.settings.logger.log(" - fpr mean: " + str(np.mean(self.fpr_matrix[:, mask_idx])))
            self.settings.logger.log(" - fpr std: " + str(np.std(self.fpr_matrix[:, mask_idx])))
