import torch
import numpy as np
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

class ModelWithIsotonicCalibration(nn.Module):
    """
    A wrapper class that applies isotonic regression calibration to a PyTorch model's binary predictions.
    This improves the reliability of probability estimates by transforming raw model outputs.
    """
    def __init__(self, model):
        super(ModelWithIsotonicCalibration, self).__init__()
        self.model = model
        self.device = next(model.parameters()).device
        self.isotonic = IsotonicRegression(out_of_bounds='clip')  # Clip ensures predictions stay within [0,1]
        self.fitted = False
    
    def fit_calibration(self, dataloader):
        """
        Fit the isotonic regression model on validation data to calibrate probabilities.
        
        Args:
            dataloader: DataLoader containing validation data
            
        Returns:
            tuple: Raw probabilities and ground truth labels used for fitting
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        
        # Collect predictions and labels without gradient computation
        with torch.no_grad():
            for x, mask, y in dataloader:
                x, mask = x.to(self.device), mask.to(self.device)
                _, _, _, logits = self.model((x, mask))
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                labels = y[:, 1].numpy()
                all_probs.extend(probs)
                all_labels.extend(labels)
        
        # Fit isotonic regression on collected data
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        self.isotonic.fit(all_probs, all_labels)
        self.fitted = True
        return all_probs, all_labels

    def forward(self, input_data):
        """
        Forward pass that applies calibration to model outputs.
        
        Args:
            input_data: Input tensor or tuple of tensors (input_ids, attention_mask)
            
        Returns:
            tuple: Model outputs with calibrated binary logits
        """
        # Handle different input formats
        if isinstance(input_data, tuple):
            input_ids, attention_mask = input_data
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            input_data = (input_ids, attention_mask)
        
        # Get model outputs
        hidden_state, output_logits, _, binary_logits = self.model(input_data)
        
        # Apply calibration if the model has been fitted
        if self.fitted:
            with torch.no_grad():
                # Convert logits to probabilities
                probs = torch.sigmoid(binary_logits).cpu().numpy()
                # Transform probabilities using fitted isotonic regression
                calibrated_probs = torch.tensor(
                    self.isotonic.transform(probs),
                    device=binary_logits.device
                )
                # Convert back to logits
                binary_logits = torch.log(calibrated_probs / (1 - calibrated_probs + 1e-7))
        
        return hidden_state, output_logits, None, binary_logits

    def predict_binary(self, x_input, batch_size=128):
        """
        Make calibrated binary predictions.
        
        Args:
            x_input: Input data
            batch_size: Batch size for processing
            
        Returns:
            torch.Tensor: Calibrated probabilities
        """
        # Get uncalibrated probabilities from the base model
        uncalibrated_probs = self.model.predict_binary(x_input, batch_size)
        
        # Apply calibration if fitted
        if self.fitted:
            return torch.tensor(
                self.isotonic.transform(uncalibrated_probs.cpu().numpy()),
                device=uncalibrated_probs.device
            )
        return uncalibrated_probs


class CalibratedModelEvaluator:
    """
    Evaluates model calibration by measuring and visualizing how well predicted 
    probabilities match the actual frequency of positive outcomes.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
    
    def _create_dataset(self, x, mask, labels):
        """
        Create a PyTorch Dataset from input tensors and labels.
        
        Args:
            x: Input tensor
            mask: Attention mask tensor
            labels: Labels dataframe
            
        Returns:
            TensorDataset: Dataset containing inputs and labels
        """
        return TensorDataset(
            x,
            mask,
            torch.tensor(labels.values, dtype=torch.float32)
        )
    
    def _get_predictions(self, model, dataloader):
        """
        Get model predictions on a dataset.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader containing evaluation data
            
        Returns:
            tuple: Predictions and true labels
        """
        predictions = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for x, mask, y in dataloader:
                x, mask = x.to(self.device), mask.to(self.device)
                _, _, _, logits = model((x, mask))
                pred = torch.sigmoid(logits).cpu().numpy()
                if len(pred.shape) > 1:
                    pred = pred.squeeze()
                predictions.append(pred)
                
                label = y[:, 1].numpy()
                if len(label.shape) > 1:
                    label = label.squeeze()
                labels.append(label)
        
        return np.concatenate(predictions), np.concatenate(labels)

    def _compute_calibration_metrics(self, probs, cal_probs, labels, n_bins=10):
        """
        Compute calibration metrics for uncalibrated and calibrated predictions.
        
        Args:
            probs: Uncalibrated probabilities
            cal_probs: Calibrated probabilities
            labels: True labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            dict: Dictionary of calibration metrics
        """
        def compute_single_metrics(prediction_probs, true_labels):
            # Calculate calibration curve
            prob_true, prob_pred = calibration_curve(
                true_labels, prediction_probs, n_bins=n_bins, strategy='quantile'
            )
            
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(prob_true - prob_pred))
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(prob_true - prob_pred))
            
            return {
                'ece': float(ece),
                'mce': float(mce)
            }
        
        return {
            'uncalibrated': compute_single_metrics(probs, labels),
            'calibrated': compute_single_metrics(cal_probs, labels)
        }

    def _plot_calibration_curves(self, val_uncal_probs, val_cal_probs, val_labels,
                                test_uncal_probs, test_cal_probs, test_labels, n_bins=10):
        """
        Plot calibration curves for validation and test sets, before and after calibration.
        
        Args:
            val_uncal_probs: Validation set uncalibrated probabilities
            val_cal_probs: Validation set calibrated probabilities
            val_labels: Validation set labels
            test_uncal_probs: Test set uncalibrated probabilities
            test_cal_probs: Test set calibrated probabilities
            test_labels: Test set labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            matplotlib.figure.Figure: Figure containing calibration plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Validation set plots
        for ax, probs, title in [
            (axes[0,0], val_uncal_probs, 'Validation Set: Before Calibration'),
            (axes[0,1], val_cal_probs, 'Validation Set: After Calibration')
        ]:
            prob_true, prob_pred = calibration_curve(
                val_labels, probs, n_bins=n_bins, strategy='quantile'
            )
            ax.plot(prob_pred, prob_true, marker='o', label='Model predictions')
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        
        # Test set plots
        for ax, probs, title in [
            (axes[1,0], test_uncal_probs, 'Test Set: Before Calibration'),
            (axes[1,1], test_cal_probs, 'Test Set: After Calibration')
        ]:
            prob_true, prob_pred = calibration_curve(
                test_labels, probs, n_bins=n_bins, strategy='quantile'
            )
            ax.plot(prob_pred, prob_true, marker='o', label='Model predictions')
            ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(title)
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        return fig

    def create_enhanced_calibration_plot(self, uncal_probs, cal_probs, true_labels, n_bins=10):
        """
        Create enhanced calibration plots showing before and after calibration side by side,
        each with a reliability diagram and histogram.
        
        Args:
            uncal_probs: Uncalibrated probabilities
            cal_probs: Calibrated probabilities
            true_labels: True labels
            n_bins: Number of bins for calibration curve
            
        Returns:
            matplotlib.figure.Figure: Figure containing enhanced calibration plots
        """
        # Set plot parameters
        plt.rcParams.update({
            'font.size': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'legend.fontsize': 9
        })

        # Calculate calibration metrics
        metrics_uncal = self._compute_calibration_metrics(uncal_probs, uncal_probs, true_labels)
        metrics_cal = self._compute_calibration_metrics(cal_probs, cal_probs, true_labels)
        
        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[1, 1], 
                            hspace=0.1, wspace=0.25)
        
        # Create the subplots
        ax_upper_left = fig.add_subplot(gs[0, 0])  # Before calibration - upper
        ax_lower_left = fig.add_subplot(gs[1, 0])  # Before calibration - lower
        ax_upper_right = fig.add_subplot(gs[0, 1])  # After calibration - upper
        ax_lower_right = fig.add_subplot(gs[1, 1])  # After calibration - lower
        
        # Before Calibration - Upper Left (Reliability diagram)
        prob_true_uncal, prob_pred_uncal = calibration_curve(
            true_labels, uncal_probs, n_bins=n_bins, strategy='quantile'
        )

        ax_upper_left.plot(prob_pred_uncal, prob_true_uncal, 'o-', color='#E23F44', markersize=5, linewidth=1.5,
                        label=f'Test set, ECE: {metrics_uncal["uncalibrated"]["ece"]:.3f}')
        ax_upper_left.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Ideally calibrated')
        ax_upper_left.set_ylabel('Ratio of positives')
        ax_upper_left.set_xlim([0, 1])
        ax_upper_left.set_ylim([0, 1])
        ax_upper_left.grid(True, alpha=0.3)
        ax_upper_left.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='black')
        ax_upper_left.set_title('Before Calibration', pad=10)
        ax_upper_left.set_xticklabels([])  # Hide x-label for upper plot

        # Before Calibration - Lower Left (Histogram)
        ax_lower_left.hist(uncal_probs, bins=n_bins, density=True, color='#E23F44', edgecolor='black',
                        alpha=0.9, range=(0, 1), label='Test set', rwidth=0.85)
        ax_lower_left.set_xlabel('Mean predicted probability')
        ax_lower_left.set_ylabel('Relative frequency')
        ax_lower_left.grid(True, alpha=0.3)
        ax_lower_left.legend(frameon=True, framealpha=1.0, edgecolor='black')

        # After Calibration - Upper Right (Reliability diagram)
        prob_true_cal, prob_pred_cal = calibration_curve(
            true_labels, cal_probs, n_bins=n_bins, strategy='quantile'
        )
        
        ax_upper_right.plot(prob_pred_cal, prob_true_cal, 'o-', color='#E23F44', markersize=5, linewidth=1.5,
                            label=f'Test set, ECE: {metrics_cal["uncalibrated"]["ece"]:.3f}')
        ax_upper_right.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Ideally calibrated')
        ax_upper_right.set_xlim([0, 1])
        ax_upper_right.set_ylim([0, 1])
        ax_upper_right.grid(True, alpha=0.3)
        ax_upper_right.legend(loc='upper left', frameon=True, framealpha=1.0, edgecolor='black')
        ax_upper_right.set_title('After Calibration', pad=10)
        ax_upper_right.set_xticklabels([])  # Hide x-label for upper plot
        ax_upper_right.set_ylabel('')  # Remove redundant y-axis label
        
        # After Calibration - Lower Right (Histogram)
        ax_lower_right.hist(cal_probs, bins=n_bins, density=True, color='#E23F44', edgecolor='black',
                            alpha=0.9, range=(0, 1), label='Test set', rwidth=0.85)
        ax_lower_right.set_xlabel('Mean predicted probability')
        ax_lower_right.grid(True, alpha=0.3)
        ax_lower_right.legend(frameon=True, framealpha=1.0, edgecolor='black')
        ax_lower_right.set_ylabel('')  # Remove redundant y-axis label
        
        # Add main title
        fig.text(0.37, 0.99, 'Model Calibration Analysis for Predicting Bleeding', 
                fontsize=12, weight='bold', ha='center')
        
        # Adjust subplot layout
        fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
        
        return fig

    def run_calibration_pipeline(self, val_data, val_labels, test_data, test_labels, batch_size=32):
        """
        Run the complete calibration pipeline:
        1. Get validation set predictions
        2. Fit calibration on validation data
        3. Get test set predictions
        4. Compute metrics and generate plots
        
        Args:
            val_data: Validation input data (x, mask)
            val_labels: Validation labels
            test_data: Test input data (x, mask)
            test_labels: Test labels
            batch_size: Batch size for inference
            
        Returns:
            tuple: Calibrated model, metrics dictionary, and calibration plot figure
        """
        # Get validation set predictions before calibration
        val_dataset = self._create_dataset(val_data[0], val_data[1], val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        val_uncal_probs, val_labels_array = self._get_predictions(self.model, val_loader)
        
        # Fit calibration on validation data
        print("Fitting calibration on validation data...")
        calibrated_model = ModelWithIsotonicCalibration(self.model)
        calibrated_model.fit_calibration(val_loader)
        val_cal_probs = calibrated_model.isotonic.transform(val_uncal_probs)
        
        # Get test set predictions
        print("\nEvaluating on test data...")
        test_dataset = self._create_dataset(test_data[0], test_data[1], test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_uncal_probs, test_labels_array = self._get_predictions(self.model, test_loader)
        test_cal_probs = calibrated_model.isotonic.transform(test_uncal_probs)
        
        # Compute metrics
        val_metrics = self._compute_calibration_metrics(val_uncal_probs, val_cal_probs, val_labels_array)
        test_metrics = self._compute_calibration_metrics(test_uncal_probs, test_cal_probs, test_labels_array)
        
        # Generate plots
        fig = self._plot_calibration_curves(
            val_uncal_probs, val_cal_probs, val_labels_array,
            test_uncal_probs, test_cal_probs, test_labels_array
        )
        
        # Print metrics
        print("\nValidation Set Metrics:")
        print("Before calibration:")
        print(f"ECE: {val_metrics['uncalibrated']['ece']:.3f}")
        print(f"MCE: {val_metrics['uncalibrated']['mce']:.3f}")
        print("After calibration:")
        print(f"ECE: {val_metrics['calibrated']['ece']:.3f}")
        print(f"MCE: {val_metrics['calibrated']['mce']:.3f}")
        
        print("\nTest Set Metrics:")
        print("Before calibration:")
        print(f"ECE: {test_metrics['uncalibrated']['ece']:.3f}")
        print(f"MCE: {test_metrics['uncalibrated']['mce']:.3f}")
        print("After calibration:")
        print(f"ECE: {test_metrics['calibrated']['ece']:.3f}")
        print(f"MCE: {test_metrics['calibrated']['mce']:.3f}")
        
        return calibrated_model, {
            'validation': val_metrics,
            'test': test_metrics
        }, fig