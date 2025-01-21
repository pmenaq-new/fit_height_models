# Tree Height Predictor

## Project Overview

This project includes a Python module designed to fit nonlinear models to forestry data, specifically focusing on predicting tree heights. The models account for local variations by adjusting predictions per plot, allowing for detailed analysis and visualization of the effects of local conditions on tree height.

## Features

- **Model Fitting**: Fit nonlinear models to tree height data, incorporating both non-local (general) and local (plot-specific) effects.
- **Prediction Visualization**: Visualize actual data points alongside model predictions, with separate plots for non-local and local predictions to assess model performance.
- **Plot-Specific Adjustments**: Each plot's predictions are adjusted using specific local data, enhancing the model's accuracy and relevance to local conditions.

## Installation

To set up this project, you'll need a Python environment. It is recommended to use Python 3.8 or above. You can install the necessary dependencies with pip:

```bash
pip install numpy pandas matplotlib scipy
```

Ensure you have the following files in your project directory:
- `tree_height_predictor.py`: Contains the model fitting and plotting logic.
- `global_parameter.json`: Configuration file for model parameters.

## Usage

To use the model fitting and plotting features, follow these steps:

1. **Prepare Your Data**: Ensure your data is formatted correctly, typically in a pandas DataFrame with columns for tree heights, plot IDs, and other relevant metrics.
2. **Configure Models**: Adjust settings in `global_parameter.json` as needed for your specific models.
3. **Run the Predictor**: Use the `NonlinearFixedModelEstimator` class from `tree_height_predictor.py` to fit models and plot predictions.

Example usage:

```python
from tree_height_predictor import NonlinearFixedModelEstimator

# Assuming you have prepared DataFrame 'df' with the necessary data
estimator = NonlinearFixedModelEstimator(df['x'], df['y'], df['plot_id'])
estimator.fit()
estimator.plot_data_by_plotid()
```

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
