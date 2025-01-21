import json
from scipy.stats import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numba as nb
import matplotlib.pyplot as plt


global_parameters =  json.load(open('global_parameter.json'))
selected_models = global_parameters['height_models']

class NonlinearFixedModelEstimator:
    """
    This class is designed to fit nonlinear models to datasets that may include
    fixed effects. Various models are used to estimate optimal parameters and assess
    model fit using metrics such as MSE, R2, AIC, and BIC.

    Methods:
        fit_all_models: Fits all defined models to the dataset.
        construct_Z: Constructs the Z design matrix for random effects.
        fit_nonlinear_fixed_model: Fits a nonlinear model with fixed effects.
        calculate_statistics: Calculates model statistics to evaluate performance.
    """
    def __init__(self, x, y, plotid, scale=1.0, n_attempts = 3):

        self.x = np.array(x)
        self.y = np.array(y)
        self.plotid = np.array(plotid)
        self.n_attempts = n_attempts
        
        # Data for model estimation
        self.mask = np.logical_not(np.isnan(self.y))
        self.X = self.x[self.mask]
        self.Y = self.y[self.mask]
        self.Plotid = self.plotid[self.mask]
        self.nPlots = len(np.unique(self.Plotid))
        
        # Design matrix for fixed effects                
        self.Z = self.construct_Z(self.Plotid)
        self.initial_scale = np.array([scale])
        
        # We only select the models indicated in global_parameters.
        self.models = {k: v for k, v in self.define_models().items() if int(k.split('_')[-1]) in selected_models}
        
        # Initialize all models as empty and add all fitted models when executing fit
        self.result_all_models = None
        self.best_model = None
        self.fit()
        self.get_best_model()
    
    def define_models(self):
        """
        Define all the models to adjust. The effects labeled as 'random_effects' are actually
        local effects since the method of random effects was not used but fixed effects
        by plot.
        """
        def model_1nr(x, fixed_effects):
            b0 = fixed_effects
            return 1.3 + b0 * x**3

        def model_1(x, fixed_effects, Z, random_effects):
            b0 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + b0 * np.power(x, 0.3)

        def model_2nr(x, fixed_effects):
            b0 = fixed_effects
            return 1.3 + b0 * np.exp(-0.11 * x)

        def model_2(x, fixed_effects, Z, random_effects):
            b0 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + b0 * np.exp(-0.11 * x)

        def model_3nr(x, fixed_effects):
            b0, b1 = fixed_effects
            return 1.3 + np.exp(b0 + b1 / np.sqrt(x))

        def model_3(x, fixed_effects, Z, random_effects):
            b0, b1 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + np.exp(b0 + b1 / np.sqrt(x))

        def model_4nr(x, fixed_effects):
            b0, b1 = fixed_effects
            return 1.3 + b0 * x + b1 * x**2

        def model_4(x, fixed_effects, Z, random_effects):
            b0, b1 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + b0 * x + b1 * x**2

        def model_5nr(x, fixed_effects):
            b0, b1 = fixed_effects
            return 1.3 + (b0 * x) / (b1 + x)

        def model_5(x, fixed_effects, Z, random_effects):
            b0, b1 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + (b0 * x) / (b1 + x)

        def model_6nr(x, fixed_effects):
            b0, b1 = fixed_effects
            return 1.3 + (b0 * x) / (1 + x)**b1

        def model_6(x, fixed_effects, Z, random_effects):
            b0, b1 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + (b0 * x) / (1 + x)**b1

        def model_7nr(x, fixed_effects):
            b0, b1 = fixed_effects
            return 1.3 + x**2 / (b0 * x + b1)**2

        def model_7(x, fixed_effects, Z, random_effects):
            b0, b1 = np.asarray((fixed_effects + Z.dot(random_effects.T)).T)
            return 1.3 + x**2 / (b0 * x + b1)**2
        
        return {
            'model_1': {'model': model_1, 'basemodel': model_1nr, 'npar': 1},
            'model_2': {'model': model_2, 'basemodel': model_2nr, 'npar': 1},
            'model_3': {'model': model_3, 'basemodel': model_3nr, 'npar': 2},
            'model_4': {'model': model_4, 'basemodel': model_4nr, 'npar': 2},
            'model_5': {'model': model_5, 'basemodel': model_5nr, 'npar': 2},
            'model_6': {'model': model_6, 'basemodel': model_6nr, 'npar': 2},
            'model_7': {'model': model_7, 'basemodel': model_7nr, 'npar': 2}
        }

    def fit_nonlinear_model(self, nonlinear_model, n_fixpar, scale):
        np.random.seed(1234)
        initial_guess = np.concatenate((np.random.rand(n_fixpar), scale))
        
        def negative_log_likelihood(params):
            fixed_effects = params[:-1]
            scale = np.abs(params[-1])

            predicted = nonlinear_model(self.X, fixed_effects)
            residual = self.Y - predicted
            return -np.sum(norm.logpdf(residual, loc=0, scale=scale))

        result = minimize(negative_log_likelihood, initial_guess, method='Nelder-Mead')
        init_params = result.x
        init_cost = result.fun

        for _ in range(self.n_attempts):
            result_new = minimize(negative_log_likelihood, init_params, method='Nelder-Mead')
            if result_new.fun < init_cost:
                init_params = result_new.x
                init_cost = result_new.fun

        estimated_fixed_effects = result_new.x[:-1]
        scale = result_new.x[-1]

        return {'fixef': estimated_fixed_effects, 'scale': scale}

    def fit_nonlinear_fixed_effect_model(self, nonlinear_model, n_fixpar, initial_guess_fixed, initial_scale):
        
        initial_scale = initial_scale
        initial_guess_effects = np.zeros(self.nPlots * n_fixpar)
        initial_guess_indices = np.arange(n_fixpar, n_fixpar + (n_fixpar * self.nPlots)).reshape(n_fixpar, self.nPlots)
        initial_guess = np.concatenate((initial_guess_fixed, initial_guess_effects, initial_scale))
        
        def negative_log_likelihood(params):
            fixed_params = params[:n_fixpar]
            fixed_effects = params[initial_guess_indices]
            scales = np.abs(params[-1])
            
            predicted = nonlinear_model(self.X, fixed_params, self.Z, fixed_effects)
            residual = self.Y - predicted
            log_likelihood = norm.logpdf(residual, loc=0, scale=scales).sum()
            
            return -log_likelihood
        
        result = minimize(negative_log_likelihood, initial_guess, method='Nelder-Mead')
        
        estimated_fixed_parms = result.x[:n_fixpar]
        estimated_fixed_effects = result.x[n_fixpar:n_fixpar + (n_fixpar * self.nPlots)].reshape((n_fixpar, self.nPlots))
        estimated_scales = np.abs(result.x[-1])
        
        return {'fixef': estimated_fixed_parms, 'local_effect': estimated_fixed_effects, 'scale': estimated_scales}

    def fit(self):
        """
        Fits all defined models to the provided data and stores predictions for each model type.
        """
        results = []
        self.predictions = {}  # Dictionary to store predictions by model and type

        for model_name, model_info in self.models.items():
            fit_result = self.fit_nonlinear_model(model_info['basemodel'], model_info['npar'], self.initial_scale)
            fit_fixeff = self.fit_nonlinear_fixed_effect_model(model_info['model'], model_info['npar'], fit_result['fixef'], self.initial_scale)

            y_pred = model_info['basemodel'](self.X, fit_result['fixef'])
            y_pred_local = model_info['model'](self.X, fit_fixeff['fixef'], self.Z, fit_fixeff['local_effect'])

            # Store predictions
            self.predictions[model_name + '_non_local'] = y_pred
            self.predictions[model_name + '_local'] = y_pred_local

            mse, r2, aic, bic = self.calculate_statistics(self.Y, y_pred, model_info['npar'] + 1)
            msel, r2l, aicl, bicl = self.calculate_statistics(self.Y, y_pred_local, (model_info['npar'] * self.nPlots) + 1)

            results.append({
                'model_name': model_name,
                'model_type': 'non_local',
                'parameters': fit_result['fixef'],
                'scale': fit_result['scale'],
                'locals': np.zeros((model_info['npar'], self.nPlots)),  # Assuming no local effects here
                'mse': mse,
                'r2': r2,
                'aic': aic,
                'bic': bic,
                'y_pred': y_pred
            })

            results.append({
                'model_name': model_name,
                'model_type': 'local',
                'parameters': fit_fixeff['fixef'],
                'scale': fit_fixeff['scale'],
                'locals': fit_fixeff['local_effect'],
                'mse': msel,
                'r2': r2l,
                'aic': aicl,
                'bic': bicl,
                'y_pred': y_pred_local
            })

        self.result_all_models = pd.DataFrame(results)

    def get_best_model(self):
        """
        Finds and returns the best model based on a combined ranking of mse, r2, aic, and bic.
        """
        all_models = self.result_all_models

        # Crear una copia para trabajar con los rangos
        ranks = all_models[['r2', 'mse', 'aic', 'bic']].copy()
        ranks['r2'] = ranks['r2'].rank(ascending=False, method='min')
        ranks['mse'] = ranks['mse'].rank(ascending=True, method='min')
        ranks['aic'] = ranks['aic'].rank(ascending=True, method='min')
        ranks['bic'] = ranks['bic'].rank(ascending=True, method='min')

        # Calcular un puntaje total sumando los rangos y normalizar
        all_models['total_rank'] = ranks.sum(axis=1) / 4.0

        # Encontrar el modelo con el menor puntaje total (mejor modelo)
        best_index = all_models['total_rank'].idxmin()
        best_model = all_models.loc[best_index]

        # Almacenar el mejor modelo para uso futuro dentro de la clase
        self.best_model = best_model
        return best_model

    def predict(self):
        """
        Performs predictions using the best model fitted previously.
        Ensures that all computed local effects are consistent with the new plot IDs.
        If a plot ID does not have a computed local effect, zero is used.

        Args:
            X_new (list or np.array): New values of independent variables for which predictions are desired.
            plotid_new (list or np.array): Plot identifiers corresponding to the new values of X.

        Returns:
            np.array: Predictions of the model for the provided new values.
        """
        if not hasattr(self, 'best_model'):
            raise ValueError("No best model found. Please fit the model before predicting.")

        X_new = self.x
        plotid_new = self.plotid

        # Construct the Z matrix for the new data
        Z_new = self.construct_Z(plotid_new)

        # Extract parameters from the best model
        fixed_effects = self.best_model['parameters']
        local_effects_matrix = self.best_model['locals']
        best_model_func = self.models[self.best_model['model_name']]['model']

        # Assign local effects based on existing plot IDs
        unique_plots = np.unique(self.Plotid)  # plot IDs used in model fitting
        local_effects = np.zeros((len(fixed_effects), len(plotid_new)))  # initialize with zeros

        for i, plot_id in enumerate(plotid_new):
            if plot_id in unique_plots:
                plot_index = np.where(unique_plots == plot_id)[0][0]
                local_effects[:, i] = local_effects_matrix[:, plot_index]
            else:
                # It is not necessary to explicitly assign zeros since local_effects is initialized with zeros
                continue
            
        predictions = best_model_func(X_new, fixed_effects, Z_new, local_effects_matrix)    

        return predictions

    def plot_data_by_plotid(self):
        """
        Plots data and model predictions from the result_all_models DataFrame,
        with separate lines for each plot's local predictions.
        """
        unique_plots = np.unique(self.Plotid)  # This should be available or correctly passed
        model_names = self.result_all_models['model_name'].unique()
        num_models = len(model_names)
        fig, axs = plt.subplots(num_models, 2, figsize=(15, 5 * num_models), squeeze=False)

        for i, model_name in enumerate(model_names):
            model_data = self.result_all_models[self.result_all_models['model_name'] == model_name]

            for j, model_type in enumerate(['non_local', 'local']):
                ax = axs[i][j]
                ax.scatter(self.x, self.y, color='gray', alpha=0.5, label='Actual Data')

                if model_type == 'non_local':
                    non_local_data = model_data[model_data['model_type'] == 'non_local']
                    for index, row in non_local_data.iterrows():
                        y_pred = row['y_pred']
                        ax.plot(self.X, y_pred, linestyle='--', label=f'{model_name} Non-Local Predictions')

                elif model_type == 'local':
                    local_data = model_data[model_data['model_type'] == 'local']
                    for index, row in local_data.iterrows():
                        for k, plot_id in enumerate(unique_plots):
                            plot_mask = self.Plotid == plot_id
                            x_plot = self.x[plot_mask]
                            y_pred_local = row['y_pred'][plot_mask] 
                            ax.plot(x_plot, y_pred_local, linestyle='-.', label=f'{model_name} Local Predictions for Plot {plot_id}')

                ax.set_title(f'{model_name} - {model_type.capitalize()} Predictions')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.legend()

        plt.tight_layout()
        plt.show()

          
    @staticmethod
    def prepare_results_dataframe(self, best_result):
        model_name, params, mse, r2, aic, bic, y_pred = best_result
        df = pd.DataFrame({
            'plotid': self.plotid,
            'x': self.x,
            'y': self.y,
            'yhat': y_pred,
            'model': model_name,
            'parameters': str(params)  
        })
        return df

    @staticmethod
    def calculate_statistics(y_true, y_pred, n_params):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        log_likelihood = -0.5 * n * np.log(2 * np.pi * np.var(y_true - y_pred)) - (n / 2)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n) * n_params - 2 * log_likelihood
        return mse, r2, aic, bic

    @staticmethod
    def construct_Z(plotid):
        n_samples = len(plotid)
        n_plots = len(np.unique(plotid))
        Z = np.zeros((n_samples, n_plots))
        for i, plot in enumerate(np.unique(plotid)):
            Z[plotid == plot, i] = 1
        return csr_matrix(Z)
