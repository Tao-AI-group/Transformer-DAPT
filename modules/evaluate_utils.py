from collections import defaultdict
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


class Evaluator:
    """
    Evaluation class for a survival model that predicts binary classification outputs.

    This class can:
    - Evaluate binary classification metrics such as accuracy, F1 score, AUROC, and AUPRC.
    - Perform optional bootstrapping to compute confidence intervals for these metrics.

    Args:
        df (pd.DataFrame): A DataFrame containing all relevant data.
        train_index (pd.Index or list/array): The indices used for the training set.
    """
    def __init__(self, df, train_index):
        """
        Initialize the evaluator with a reference DataFrame and the training indices.

        Args:
            df (pd.DataFrame): Full DataFrame containing survival data.
            train_index (pd.Index or list): Indices that define the training subset.
        """
        self.df_train_all = df.loc[train_index]

    def eval_single(self, model, test_set, times=None, val_batch_size=64):
        """
        Evaluate the model on a binary classification task.

        This method prints:
        - Accuracy
        - F1 Score
        - AUROC
        - AUPRC

        Args:
            model: The model that should implement a `.predict_binary(...)` method.
            test_set (tuple): A tuple of (df_test, df_y_test) where:
                - df_test is the set of features/inputs.
                - df_y_test is a DataFrame with 'duration' and 'event' columns.
            times (unused): Potentially for time-dependent metrics (not used here).
            val_batch_size (int, optional): Batch size for model predictions. Defaults to 64.
        """
        model.eval()

        get_target = lambda df: (df['duration'].values, df['event'].values)

        df_test, df_y_test = test_set

        # Compute binary classification scores
        binary_score = model.predict_binary(df_test, batch_size=val_batch_size)

        durations_test, events_test = get_target(df_y_test)

        # Move predictions to CPU if on GPU
        if binary_score.is_cuda:
            binary_score = binary_score.cpu()

        # Calculate and print metrics
        acc = accuracy_score(events_test, binary_score > 0.5)
        f1 = f1_score(events_test, binary_score > 0.5)
        auroc = roc_auc_score(events_test, binary_score)
        auprc = average_precision_score(events_test, binary_score)

        print(f"Accuracy: {acc}")
        print(f"F1 score: {f1}")
        print(f"AUROC: {auroc}")
        print(f"AUPRC: {auprc}")

        # If desired, return a dictionary for downstream processing or bootstrapping
        # return {"accuracy": acc, "f1": f1, "auroc": auroc, "auprc": auprc}

    def eval(self, model, test_set, times=None, confidence=None, val_batch_size=None):
        """
        Master evaluation method which calls `eval_single` for binary tasks.
        Supports optional bootstrapping for confidence intervals.

        Args:
            model: The model to evaluate. Must have `.predict_binary(...)` implemented.
            test_set (tuple): A tuple of (df_test, df_y_test).
            times (list or None): Potentially for time-dependent metrics (unused here).
            confidence (float or None): If provided (e.g., 0.95), bootstrapping is used
                to estimate confidence intervals.
            val_batch_size (int or None): Batch size for evaluation. If None, defaults
                to an internal setting or a constant in `model.predict_binary`.

        Returns:
            None or dict:
                - If `confidence` is None, prints metrics for single-event evaluation.
                - If `confidence` is provided, returns a dictionary containing the average
                  metric and half-width of the confidence interval for each key metric.
        """
        print("***" * 10)
        print("start evaluation")
        print("***" * 10)

        if confidence is None:
            return self.eval_single(model, test_set, times, val_batch_size)
        else:
            # Perform bootstrapping for confidence intervals
            stats_dict = defaultdict(list)
            df_test_original, df_y_test_original = test_set

            for _ in range(10):  # 10 bootstraps; adjust as needed
                df_test_sample = df_test_original.sample(df_test_original.shape[0], replace=True)
                df_y_test_sample = df_y_test_original.loc[df_test_sample.index]

                # Potentially, eval_single could return a dict if you implement that
                res_dict = self.eval_single(model, (df_test_sample, df_y_test_sample), val_batch_size)

                # If eval_single returned a dict, store results. Currently it prints only.
                # E.g., if you changed eval_single to return a dict:
                #
                # if isinstance(res_dict, dict):
                #     for k in res_dict.keys():
                #         stats_dict[k].append(res_dict[k])
                #
                # Otherwise, nothing will be collected here.

            # Since eval_single doesn't return a dict by default, stats_dict will be empty.
            # This block shows how you'd compute confidence intervals if you had data.
            metric_dict = {}
            alpha = confidence
            p1 = ((1 - alpha) / 2) * 100
            p2 = (alpha + ((1.0 - alpha) / 2.0)) * 100

            for k in stats_dict.keys():
                stats = stats_dict[k]
                lower = max(0, np.percentile(stats, p1))
                upper = min(1.0, np.percentile(stats, p2))

                avg_ = (upper + lower) / 2
                half_interval_ = (upper - lower) / 2

                print(f'{alpha} confidence {k} average: {avg_}')
                print(f'{alpha} confidence {k} interval: {half_interval_}')

                metric_dict[k] = [avg_, half_interval_]

            return metric_dict
