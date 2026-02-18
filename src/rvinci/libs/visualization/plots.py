import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm, labels, output_path, normalize=False, title='Confusion Matrix'):
    """
    Plot the confusion matrix using seaborn and matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 10)) # Slightly larger for readability
        
        plot_cm = cm
        fmt = "d"
        
        if normalize:
            # Normalize by row (Ground Truth)
            # Avoid division by zero
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            plot_cm = cm.astype('float') / (row_sums + 1e-9)
            # plot_cm = np.nan_to_num(plot_cm) # Should not be needed with epsilon
            fmt = ".2f"
            
        sns.heatmap(plot_cm, annot=True, fmt=fmt, xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path)
        logger.info(f"Confusion matrix plot saved to {output_path}")
        plt.close()
        
    except ImportError:
        logger.warning("matplotlib or seaborn not found. detailed plot skipped.")
    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")
