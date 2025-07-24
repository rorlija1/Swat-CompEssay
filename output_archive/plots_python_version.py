#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:07:40 2025

@author: sranka
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import random
from pathlib import Path
import warnings

def read_normalization_params(filename="normalization_params.txt", output_name='_'):
    """Function to read normalization parameters"""
    std_path = os.path.join(f"output_{output_name}", filename)
    required_files = [std_path]
    
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Missing required file: {file}")
    
    target_mean = 0.0
    target_std = 1.0
    
    if os.path.isfile(std_path):
        with open(std_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    if parts[0] == "target_mean":
                        target_mean = float(parts[1])
                    elif parts[0] == "target_std":
                        target_std = float(parts[1])
        print(f"Loaded normalization params: mean={target_mean}, std={target_std}")
    else:
        print("Warning: normalization_params.txt not found. Using default values.")
    
    return target_mean, target_std

def read_rmse_values(filename):
    """Function to extract RMSE values from a file"""
    rmse_values = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("RMSE is "):
                value = float(line.split()[-1])  # Extract numeric value
                rmse_values.append(value)
    return rmse_values

def read_actual_predicted(filename, rmse_values, particle_plot=False, task="", savefig="", units="standardized"):
    """
    Function to extract actual and predicted values from a file
    also extracts actual and predicted for one particle across all epochs
    (if particle_plot = True)
    """
    actual_values = []
    predicted_values = []
    actual_values_parity = []
    predicted_values_parity = []
    n_epochs = len(rmse_values)
    
    if particle_plot:
        actual_values_single = []
        predicted_values_single = []
        # count total number of particles
        n = 0
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:  # Ensure correct format
                    n += 1
                elif len(parts) > 0 and parts[0] == "RMSE":
                    break
        
        # pick random particle and accumulate vals
        choice = random.randint(1, n)
        
        counter = 1  # count particle number
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:  # Ensure correct format
                    if counter == choice:  # accumulate for one particle
                        try:
                            actual = float(parts[0])  # First column: actual values
                            predicted = float(parts[1])  # Second column: predicted values
                            actual_values_single.append(actual)
                            predicted_values_single.append(predicted)
                        except ValueError:
                            print(f"Skipping invalid line in {filename}: {line}")
                    counter += 1
                elif len(parts) > 0 and parts[0] == "RMSE":
                    counter = 1  # reset for new epoch
        
        # plot particle actual and predicted
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, n_epochs + 1)
        ylabel_text = "Target (normalized)" if units == "standardized" else "Target (original units)"
        
        ax.plot(epochs, predicted_values_single, linewidth=2, color='darkcyan', label='Prediction')
        ax.scatter(epochs, predicted_values_single, color='black', s=25)
        ax.plot(epochs, actual_values_single, linewidth=2, color='cyan', linestyle='--', label='Target')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel_text)
        ax.set_title(f'Validation predictions; particle #{choice} ({units})')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"1Particle_{task}_{units}_{savefig}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    epoch = 1  # counter
    best_rmse = min(rmse_values)
    best_epoch = rmse_values.index(best_rmse) + 1
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if epoch == best_epoch:
                # save output from best epoch to use for parity plot
                if len(parts) == 2:  # Ensure correct format
                    try:
                        actual = float(parts[0])  # First column: actual values
                        predicted = float(parts[1])  # Second column: predicted values
                        actual_values_parity.append(actual)
                        predicted_values_parity.append(predicted)
                    except ValueError:
                        print(f"Skipping invalid line in {filename}: {line}")
            
            if len(parts) == 2:  # Ensure correct format
                try:
                    actual = float(parts[0])  # First column: actual values
                    predicted = float(parts[1])  # Second column: predicted values
                    actual_values.append(actual)
                    predicted_values.append(predicted)
                except ValueError:
                    print(f"Skipping invalid line in {filename}: {line}")
            elif len(parts) > 0 and parts[0] == "RMSE":
                epoch += 1
    
    return actual_values, predicted_values, actual_values_parity, predicted_values_parity

def read_lr(filename="Learning_rates", output_name='_'):
    """Function to read learning rates"""
    lr_path = os.path.join(f"output_{output_name}", filename)
    required_files = [lr_path]
    
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Missing required file: {file}")
    
    lrs = []
    with open(lr_path, 'r') as file:
        for line in file:
            value = float(line.split()[-1])  # Extract numeric value
            lrs.append(value)
    return lrs

def safe_plot(ax, x, y, scatter_only=False, origin="", label="", **kwargs):
    """Safe plotting function with dimension checking"""
    # Extract specific kwargs
    line_kwargs = kwargs.get('line_kwargs', {})
    scatter_kwargs = kwargs.get('scatter_kwargs', {})
    
    if len(x) != len(y):
        error_msg = f"""
        🚫 Dimension mismatch in plotting at: {origin}
        → x has length {len(x)}, y has length {len(y)}
        → shape(x) = {np.array(x).shape}, shape(y) = {np.array(y).shape}
        """
        raise ValueError(error_msg)
    
    if not scatter_only:
        ax.plot(x, y, label=label, **line_kwargs)
    else:
        ax.scatter(x, y, **scatter_kwargs)

def compute_r2(y_true, y_pred):
    """Compute R-squared"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot

def make_plots(task, savefig, display_only=False, output_name='_'):
    """Main function to create all plots"""
    
    # Output directory
    MLfiles = f"output_{output_name}"
    outdir = os.path.join(MLfiles, f"plots_{task}_{savefig}")
    os.makedirs(outdir, exist_ok=True)
    
    def handle_output(fig, name):
        if display_only:
            plt.show()
        else:
            fig.savefig(os.path.join(outdir, name), dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    # File checks
    val_file_std = os.path.join(MLfiles, "Actual_and_predicted_values_val_standardized")
    train_file_std = os.path.join(MLfiles, "Actual_and_predicted_values_train_standardized")
    val_file_orig = os.path.join(MLfiles, "Actual_and_predicted_values_val_original")
    train_file_orig = os.path.join(MLfiles, "Actual_and_predicted_values_train_original")
    required_files = [val_file_std, train_file_std, val_file_orig, train_file_orig]
    
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Missing required file: {file}")
    
    target_mean, target_std = read_normalization_params(output_name=output_name)
    rmse_values_val = read_rmse_values(val_file_std)
    rmse_values_train = read_rmse_values(train_file_std)
    best_rmse = min(rmse_values_val)
    best_epoch = rmse_values_val.index(best_rmse) + 1
    epochs = list(range(1, len(rmse_values_val) + 1))
    
    actual_val_std, predicted_val_std, actual_val_parity_std, predicted_val_parity_std = read_actual_predicted(
        val_file_std, rmse_values_val, particle_plot=True, task=task, savefig=savefig, units="standardized")
    actual_train_std, predicted_train_std, actual_train_parity_std, predicted_train_parity_std = read_actual_predicted(
        train_file_std, rmse_values_val, task=task, savefig=savefig, units="standardized")
    
    rmse_values_val_orig = read_rmse_values(val_file_orig)
    rmse_values_train_orig = read_rmse_values(train_file_orig)
    actual_val_orig, predicted_val_orig, actual_val_parity_orig, predicted_val_parity_orig = read_actual_predicted(
        val_file_orig, rmse_values_val_orig, particle_plot=True, task=task, savefig=savefig, units="original")
    actual_train_orig, predicted_train_orig, actual_train_parity_orig, predicted_train_parity_orig = read_actual_predicted(
        train_file_orig, rmse_values_train_orig, task=task, savefig=savefig, units="original")
    
    # === RMSE Plot ===
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        safe_plot(ax, epochs, rmse_values_val,
                 origin="RMSE (Validation)",
                 label="Validation",
                 line_kwargs={'linewidth': 2, 'color': 'darkcyan'})
        ax.scatter(epochs, rmse_values_val, color='black', s=25)
        
        safe_plot(ax, epochs, rmse_values_train,
                 origin="RMSE (Training)",
                 label="Training",
                 line_kwargs={'linewidth': 2, 'color': 'cyan'})
        ax.scatter(epochs, rmse_values_train, color='black', s=25)
        
        ax.axhline(y=best_rmse, color='mediumseagreen', linestyle='--', 
                  label=f'Best: {best_rmse:.6f} (epoch {best_epoch})')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE (normalized)')
        ax.set_title(f'RMSE ({task})')
        ax.legend(loc='upper right')
        
        handle_output(fig, "RMSE.png")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to plot RMSE: {e}")
    
    # === Log-log RMSE ===
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.loglog(epochs, rmse_values_val, linewidth=2, color='red', alpha=0.5, label='Validation')
        ax.scatter(epochs, rmse_values_val, color='red', s=25)
        
        ax.loglog(epochs, rmse_values_train, linewidth=2, color='blue', alpha=0.5, label='Training')
        ax.scatter(epochs, rmse_values_train, color='blue', s=25)
        
        ax.axhline(y=best_rmse, color='mediumseagreen', linestyle='--')
        
        ax.set_xlabel('log(Epoch)')
        ax.set_ylabel('log(RMSE)')
        ax.set_title(f'RMSE log-log ({task})')
        ax.legend(loc='upper right')
        
        handle_output(fig, "LogRMSE.png")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to plot Log-log RMSE: {e}")
    
    # === Parity Plots: Standardized ===
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Validation plot
        ax1.scatter(actual_val_parity_std, predicted_val_parity_std, 
                   alpha=0.2, color='darkcyan', edgecolors='darkcyan', s=100)
        
        # Reference line y = x
        min_val = min(min(actual_val_parity_std), min(predicted_val_parity_std))
        max_val = max(max(actual_val_parity_std), max(predicted_val_parity_std))
        ax1.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.8)
        
        # R² and RMSE
        r2 = compute_r2(actual_val_parity_std, predicted_val_parity_std)
        rmse = np.sqrt(np.mean((np.array(actual_val_parity_std) - np.array(predicted_val_parity_std))**2))
        ax1.text(0.02, 0.98, f'R² = {r2:.4f}, RMSE = {rmse:.4f}',
                transform=ax1.transAxes, fontsize=16, verticalalignment='top')
        
        ax1.set_xlabel('Actual (standardized)')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{task} Val (Standardized)')
        
        # Training plot
        ax2.scatter(actual_train_parity_std, predicted_train_parity_std,
                   alpha=0.2, color='darkcyan', edgecolors='darkcyan', s=100)
        
        # Reference line y = x
        min_train = min(min(actual_train_parity_std), min(predicted_train_parity_std))
        max_train = max(max(actual_train_parity_std), max(predicted_train_parity_std))
        ax2.plot([min_train, max_train], [min_train, max_train], 'k:', alpha=0.8)
        
        # R² and RMSE
        r2 = compute_r2(actual_train_parity_std, predicted_train_parity_std)
        rmse = np.sqrt(np.mean((np.array(actual_train_parity_std) - np.array(predicted_train_parity_std))**2))
        ax2.text(0.02, 0.98, f'R² = {r2:.4f}, RMSE = {rmse:.4f}',
                transform=ax2.transAxes, fontsize=16, verticalalignment='top')
        
        ax2.set_xlabel('Actual (standardized)')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'{task} Train (Standardized)')
        
        plt.tight_layout()
        handle_output(fig, "Parity_standardized.png")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to plot standardized parity plots: {e}")
    
    # === Parity Plots: Original Units ===
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Validation plot
        ax1.scatter(actual_val_parity_orig, predicted_val_parity_orig,
                   alpha=0.2, color='darkcyan', edgecolors='darkcyan', s=100)
        
        # Reference line y = x
        min_val_orig = min(min(actual_val_parity_orig), min(predicted_val_parity_orig))
        max_val_orig = max(max(actual_val_parity_orig), max(predicted_val_parity_orig))
        ax1.plot([min_val_orig, max_val_orig], [min_val_orig, max_val_orig], 'k:', alpha=0.8)
        
        # R² and RMSE
        r2 = compute_r2(actual_val_parity_orig, predicted_val_parity_orig)
        rmse = np.sqrt(np.mean((np.array(actual_val_parity_orig) - np.array(predicted_val_parity_orig))**2))
        ax1.text(0.02, 0.98, f'R² = {r2:.4f}, RMSE = {rmse:.4f}',
                transform=ax1.transAxes, fontsize=16, verticalalignment='top')
        
        ax1.set_xlabel('Actual (original units)')
        ax1.set_ylabel('Predicted')
        ax1.set_title(f'{task} Val (Original Units)')
        
        # Training plot
        ax2.scatter(actual_train_parity_orig, predicted_train_parity_orig,
                   alpha=0.2, color='darkcyan', edgecolors='darkcyan', s=100)
        
        # Reference line y = x
        min_train_orig = min(min(actual_train_parity_orig), min(predicted_train_parity_orig))
        max_train_orig = max(max(actual_train_parity_orig), max(predicted_train_parity_orig))
        ax2.plot([min_train_orig, max_train_orig], [min_train_orig, max_train_orig], 'k:', alpha=0.8)
        
        # R² and RMSE
        r2 = compute_r2(actual_train_parity_orig, predicted_train_parity_orig)
        rmse = np.sqrt(np.mean((np.array(actual_train_parity_orig) - np.array(predicted_train_parity_orig))**2))
        ax2.text(0.02, 0.98, f'R² = {r2:.4f}, RMSE = {rmse:.4f}',
                transform=ax2.transAxes, fontsize=16, verticalalignment='top')
        
        ax2.set_xlabel('Actual (original units)')
        ax2.set_ylabel('Predicted')
        ax2.set_title(f'{task} Train (Original Units)')
        
        plt.tight_layout()
        handle_output(fig, "Parity_original.png")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to plot original unit parity plots: {e}")
    
    # === Learning Rate Plot ===
    try:
        learning_rates = read_lr(output_name=output_name)
        fig, ax = plt.subplots(figsize=(12, 6))
        safe_plot(ax, epochs, learning_rates,
                 origin="Learning Rate",
                 line_kwargs={'linewidth': 2, 'color': 'darkcyan'})
        ax.scatter(epochs, learning_rates, color='black', s=36)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate ({task}, {savefig})')
        
        handle_output(fig, "LearningRate.png")
    except Exception as e:
        warnings.warn(f"⚠️ Failed to plot learning rate: {e}")
    
    print(f"✅ Finished. Plots stored in: {outdir}")

# Example usage
if __name__ == "__main__":
    make_plots("Displacement", "MD10000", output_name="10000_run1")