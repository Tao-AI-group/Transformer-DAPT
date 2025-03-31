import os
import pickle
import numpy as np
import random
from modules.model_comparison import ModelComparison

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.

    Args:
        seed: Integer value to use for seeding the random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)

def load_results(results_dir: str, filename: str):
    """
    Load pickled results from the specified directory and filename.

    Args:
        results_dir: Path to the directory containing results files.
        filename: Name of the pickle file to load.

    Returns:
        The object loaded from the specified pickle file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    filepath = os.path.join(results_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def main() -> None:
    """
    Main function to compare model performance using the ModelComparison utility class.

    Steps:
        1. Set a fixed random seed for reproducibility.
        2. Load survival prediction results for different models.
        3. Perform bootstrap statistical comparisons.
        4. Print the comparison results.
    """
    # Set random seed
    set_seed(42)

    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'model_comparison_results')
    
    # Load pickled results
    TDAPT_results = load_results(results_dir, 'TDAPT_results.pkl')
    deepsurv_results = load_results(results_dir, 'deepsurv_results.pkl')
    deephit_results = load_results(results_dir, 'deephit_results.pkl')
    
    # Initialize the model comparison object
    model_comparison = ModelComparison(n_bootstrap=1000)
    
    # Time points (in days) at which we want to compare performance
    evaluation_times = [30, 60, 90, 180, 270, 365]
    
    # Compare Transformer_DAPT vs DeepSurv
    print("\nComparing Transformer_DAPT vs DeepSurv...")
    TDAPT_vs_deepsurv = model_comparison.bootstrap_model_comparison(
        TDAPT_results['predictions'],
        deepsurv_results['predictions'],
        TDAPT_results['durations'],
        TDAPT_results['events'],
        evaluation_times,
        model1_name="Transformer_DAPT",
        model2_name="DeepSurv"
    )
    
    # Compare Transformer_DAPT vs DeepHit
    print("\nComparing Transformer_DAPT vs DeepHit...")
    TDAPT_vs_deephit = model_comparison.bootstrap_model_comparison(
        TDAPT_results['predictions'],
        deephit_results['predictions'],
        TDAPT_results['durations'],
        TDAPT_results['events'],
        evaluation_times,
        model1_name="Transformer_DAPT",
        model2_name="DeepHit"
    )
    
    # Print results
    print("\n" + "="*50)
    print("Statistical Comparison of Models")
    print("="*50)
    model_comparison.print_comparison_results(TDAPT_vs_deepsurv)
    print("\n" + "="*50)
    model_comparison.print_comparison_results(TDAPT_vs_deephit)

if __name__ == "__main__":
    main()
