import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Set consistent style
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['text.color'] = '#e6edf3'
plt.rcParams['axes.labelcolor'] = '#e6edf3'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'
plt.rcParams['grid.color'] = '#21262d'
plt.rcParams['axes.titlecolor'] = '#e6edf3'

FRAUD_COLOR = '#ff6b6b'
LEGIT_COLOR = '#4ecdc4'
ACCENT_COLOR = '#a78bfa'

def plot_class_distribution(y, save_path='outputs/class_distribution.png'):
    """Plot fraud vs legitimate transaction counts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Class Distribution', fontsize=16, fontweight='bold', color='#e6edf3')
    
    counts = y.value_counts()
    labels = ['Legitimate', 'Fraud']
    colors = [LEGIT_COLOR, FRAUD_COLOR]
    
    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, width=0.5, edgecolor='none')
    axes[0].set_title('Transaction Counts')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{count:,}', ha='center', va='bottom', fontsize=12)
    
    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct='%1.2f%%', startangle=90,
        textprops={'color': '#e6edf3'}
    )
    for autotext in autotexts:
        autotext.set_color('#0d1117')
        autotext.set_fontweight('bold')
    axes[1].set_title('Class Proportions')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_amount_distribution(df, save_path='outputs/amount_distribution.png'):
    """Compare transaction amounts for fraud vs legitimate."""
    df = df.copy()
    if 'Amount_log' not in df.columns:
        if 'Amount' not in df.columns:
            raise KeyError("plot_amount_distribution requires either 'Amount_log' or 'Amount'.")
        df['Amount_log'] = np.log1p(df['Amount'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Transaction Amount Analysis', fontsize=16, fontweight='bold')
    
    fraud = df[df['Class'] == 1]['Amount_log']
    legit = df[df['Class'] == 0]['Amount_log']
    
    axes[0].hist(legit, bins=60, alpha=0.7, color=LEGIT_COLOR, label='Legitimate', density=True)
    axes[0].hist(fraud, bins=60, alpha=0.7, color=FRAUD_COLOR, label='Fraud', density=True)
    axes[0].set_title('Log Amount Distribution')
    axes[0].set_xlabel('Log(Amount + 1)')
    axes[0].legend()
    
    # Boxplot
    data_to_plot = [
        df[df['Class'] == 0]['Amount_log'].values,
        df[df['Class'] == 1]['Amount_log'].values
    ]
    bp = axes[1].boxplot(data_to_plot, patch_artist=True,
                         medianprops={'color': '#ffffff', 'linewidth': 2})
    bp['boxes'][0].set_facecolor(LEGIT_COLOR)
    bp['boxes'][1].set_facecolor(FRAUD_COLOR)
    axes[1].set_xticklabels(['Legitimate', 'Fraud'])
    axes[1].set_title('Amount Box Plot')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_confusion_matrix(cm, save_path='outputs/confusion_matrix.png'):
    """Plot styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = np.array([
        [f'True Negative\n{cm[0,0]:,}', f'False Positive\n{cm[0,1]:,}'],
        [f'False Negative\n{cm[1,0]:,}', f'True Positive\n{cm[1,1]:,}']
    ])
    
    im = ax.imshow(cm, interpolation='nearest', cmap='RdYlGn')
    plt.colorbar(im, ax=ax)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Legit', 'Predicted Fraud'], fontsize=12)
    ax.set_yticklabels(['Actual Legit', 'Actual Fraud'], fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] < thresh else '#0d1117'
            ax.text(j, i, labels[i, j], ha='center', va='center',
                   fontsize=13, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_roc_curve(y_test, y_prob_rf, y_prob_xgb, save_path='outputs/roc_curve.png'):
    """Plot ROC curves for both models."""
    from sklearn.metrics import roc_auc_score
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for name, y_prob, color in [
        ('Random Forest', y_prob_rf, LEGIT_COLOR),
        ('XGBoost', y_prob_xgb, ACCENT_COLOR)
    ]:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f'{name} (AUC = {auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Baseline')
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Model Comparison', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_feature_importance(model, feature_names, top_n=15,
                             save_path='outputs/feature_importance.png'):
    """Plot top N most important features."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = [ACCENT_COLOR if i < 5 else LEGIT_COLOR for i in range(top_n)]
    bars = ax.barh(
        range(top_n),
        importance[indices][::-1],
        color=colors[::-1],
        edgecolor='none'
    )
    
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=11)
    ax.set_xlabel('Feature Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_hourly_fraud(df, save_path='outputs/hourly_fraud.png'):
    """Plot fraud transactions by hour of day."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hourly = df.groupby(['Hour', 'Class']).size().unstack(fill_value=0)
    if 1 in hourly.columns:
        fraud_rate = (hourly[1] / (hourly[0] + hourly[1]) * 100)
        ax.bar(fraud_rate.index, fraud_rate.values, color=FRAUD_COLOR, alpha=0.8, edgecolor='none')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax.set_title('Fraud Rate by Hour of Day', fontsize=16, fontweight='bold')
        ax.set_xticks(range(0, 24))
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
