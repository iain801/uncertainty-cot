import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
import re

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def parse_filename(filename):
    """Extract threshold, window type, and window size from filename"""
    # Pattern: benchmark_THRESHOLD_TYPE+SIZE_TIMESTAMP.parquet
    pattern = r'benchmark_([0-9.]+)_(rolling|warmup)(\d+)_.*\.parquet'
    match = re.match(pattern, filename)
    if match:
        threshold = float(match.group(1))
        window_type = match.group(2)
        window_size = int(match.group(3))
        return threshold, window_type, window_size
    return None, None, None

def load_all_results(results_dir):
    """Load all parquet files from the results directory"""
    results = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.parquet'):
            filepath = os.path.join(results_dir, filename)
            threshold, window_type, window_size = parse_filename(filename)
            
            if threshold is not None:
                df = pd.read_parquet(filepath)
                
                # Determine control type
                control_type = "None"
                if threshold == 0.0:
                    control_type = "Base Thinking"
                elif threshold == 1.0:
                    control_type = "No Thinking"
                
                # Calculate summary statistics
                summary = {
                    'filename': filename,
                    'threshold': threshold,
                    'window_type': window_type,
                    'window_size': window_size,
                    'control_type': control_type,
                    'total_samples': len(df),
                    'correct_samples': df['is_correct'].sum(),
                    'accuracy': df['is_correct'].mean() * 100,
                    'avg_thinking_tokens': df['thinking_tokens'].mean(),
                    'avg_response_tokens': df['response_tokens'].mean(),
                    'avg_total_tokens': df['total_tokens'].mean(),
                    'avg_generation_time': df['generation_time_seconds'].mean(),
                    'avg_tokens_per_second': df['tokens_per_second'].mean(),
                    'std_thinking_tokens': df['thinking_tokens'].std(),
                    'std_generation_time': df['generation_time_seconds'].std()
                }
                results.append(summary)
    
    return pd.DataFrame(results)

def create_thinking_tokens_distribution(results_df, output_dir):
    """Create distribution plots for thinking tokens by configuration"""
    
    # We need to load the raw data again to get individual sample distributions
    raw_data = []
    
    for _, config in results_df.iterrows():
        filepath = os.path.join("/home/ixw/uncertainty-cot/results/run_1", config['filename'])
        df = pd.read_parquet(filepath)
        
        for _, sample in df.iterrows():
            raw_data.append({
                'threshold': config['threshold'],
                'window_size': config['window_size'],
                'window_type': config['window_type'],
                'control_type': config['control_type'],
                'thinking_tokens': sample['thinking_tokens'],
                'config_label': f"T:{config['threshold']:.2f}, W:{config['window_size']}" if config['control_type'] == 'None' 
                               else config['control_type']
            })
    
    raw_df = pd.DataFrame(raw_data)
    
    # Calculate reasonable y-axis limits (5th to 95th percentile to exclude outliers)
    y_min = raw_df['thinking_tokens'].quantile(0.05)
    y_max = raw_df['thinking_tokens'].quantile(0.95)
    
    # Add some padding for violin plots
    y_range = y_max - y_min
    y_min = max(0, y_min - 0.1 * y_range)
    y_max = y_max + 0.1 * y_range
    
    # Separate controls and rolling data
    controls_df = raw_df[raw_df['control_type'] != 'None']
    rolling_df = raw_df[raw_df['control_type'] == 'None']
    
    # Create violin plots with larger figure size
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Plot 1: Violin plots by window size for rolling data
    ax1 = axes[0, 0]
    window_sizes = sorted(rolling_df['window_size'].unique())
    violin_data = []
    labels = []
    colors = []
    positions = []
    pos_counter = 1
    
    color_map = {2: 'lightblue', 5: 'lightgreen', 10: 'lightcoral'}
    
    for window_size in window_sizes:
        window_data = rolling_df[rolling_df['window_size'] == window_size]
        thresholds = sorted(window_data['threshold'].unique())
        
        for threshold in thresholds:
            threshold_data = window_data[window_data['threshold'] == threshold]
            # Filter data to remove outliers for cleaner violin plots
            filtered_data = threshold_data[(threshold_data['thinking_tokens'] >= y_min) & 
                                         (threshold_data['thinking_tokens'] <= y_max)]['thinking_tokens'].values
            if len(filtered_data) > 0:  # Only add if we have data
                violin_data.append(filtered_data)
                labels.append(f"W{window_size}\nT{threshold:.2f}")
                colors.append(color_map[window_size])
                positions.append(pos_counter)
                pos_counter += 1
    
    if violin_data:
        violin_parts = ax1.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(violin_parts['bodies'], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, rotation=45, fontsize=9)
    ax1.set_title('Thinking Tokens Distribution by Window Size & Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Thinking Tokens', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)
    
    # Plot 2: Controls vs Rolling averages
    ax2 = axes[0, 1]
    control_violin_data = []
    control_labels = []
    control_colors = []
    control_positions = []
    pos_counter = 1
    
    # Add No Thinking control first (leftmost)
    no_thinking_data = controls_df[controls_df['control_type'] == 'No Thinking']
    if not no_thinking_data.empty:
        filtered_no_thinking = no_thinking_data[(no_thinking_data['thinking_tokens'] >= y_min) & 
                                               (no_thinking_data['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(filtered_no_thinking) > 0:
            control_violin_data.append(filtered_no_thinking)
            control_labels.append('No Thinking')
            control_colors.append('orange')
            control_positions.append(pos_counter)
            pos_counter += 1
    
    # Add rolling window data by window size (middle)
    for window_size in window_sizes:
        window_subset = rolling_df[rolling_df['window_size'] == window_size]
        filtered_window = window_subset[(window_subset['thinking_tokens'] >= y_min) & 
                                      (window_subset['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(filtered_window) > 0:
            control_violin_data.append(filtered_window)
            control_labels.append(f"Rolling W{window_size}")
            control_colors.append(color_map[window_size])
            control_positions.append(pos_counter)
            pos_counter += 1
    
    # Add Base Thinking control last (rightmost)
    base_thinking_data = controls_df[controls_df['control_type'] == 'Base Thinking']
    if not base_thinking_data.empty:
        filtered_base_thinking = base_thinking_data[(base_thinking_data['thinking_tokens'] >= y_min) & 
                                                   (base_thinking_data['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(filtered_base_thinking) > 0:
            control_violin_data.append(filtered_base_thinking)
            control_labels.append('Base Thinking')
            control_colors.append('red')
            control_positions.append(pos_counter)
            pos_counter += 1
    
    if control_violin_data:
        violin_parts2 = ax2.violinplot(control_violin_data, positions=control_positions, 
                                      showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(violin_parts2['bodies'], control_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax2.set_xticks(control_positions)
    ax2.set_xticklabels(control_labels, rotation=45, fontsize=10)
    ax2.set_title('Thinking Tokens: Controls vs Rolling Windows', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Thinking Tokens', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_min, y_max)
    
    # Plot 3: Violin plots for detailed distribution shapes by window size only
    ax3 = axes[1, 0]
    
    # Create violin plot for rolling data by window size
    rolling_plot_data = []
    rolling_plot_labels = []
    rolling_positions = []
    
    for i, window_size in enumerate(window_sizes, 1):
        window_subset = rolling_df[rolling_df['window_size'] == window_size]
        # Filter data to remove outliers for cleaner violin plots
        filtered_data = window_subset[(window_subset['thinking_tokens'] >= y_min) & 
                                     (window_subset['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(filtered_data) > 0:
            rolling_plot_data.append(filtered_data)
            rolling_plot_labels.append(f"Window {window_size}")
            rolling_positions.append(i)
    
    if rolling_plot_data:
        violin_parts3 = ax3.violinplot(rolling_plot_data, positions=rolling_positions, 
                                      showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(violin_parts3['bodies'], [color_map[ws] for ws in window_sizes if ws in color_map])):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax3.set_xticks(rolling_positions)
    ax3.set_xticklabels(rolling_plot_labels)
    ax3.set_title('Thinking Tokens Distribution Shape by Window Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Thinking Tokens', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(y_min, y_max)
    
    # Plot 4: Threshold effect including Base Thinking control
    ax4 = axes[1, 1]
    
    # Show how threshold affects distribution, including base thinking control
    window_size = 10  # Focus on window size 10 as an example
    threshold_violin_data = []
    threshold_labels = []
    threshold_positions = []
    threshold_colors = []
    
    # First add Base Thinking control (threshold 0.0)
    base_thinking_data = controls_df[controls_df['control_type'] == 'Base Thinking']
    if not base_thinking_data.empty:
        base_filtered = base_thinking_data[(base_thinking_data['thinking_tokens'] >= y_min) & 
                                         (base_thinking_data['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(base_filtered) > 0:
            threshold_violin_data.append(base_filtered)
            threshold_labels.append("Base\n(T:0.0)")
            threshold_positions.append(1)
            threshold_colors.append('red')
    
    # Then add rolling window data for window size 10
    window_10_data = rolling_df[rolling_df['window_size'] == window_size]
    thresholds = sorted(window_10_data['threshold'].unique())
    
    pos_counter = 2  # Start after base thinking
    for threshold in thresholds:
        threshold_subset = window_10_data[window_10_data['threshold'] == threshold]
        filtered_threshold = threshold_subset[(threshold_subset['thinking_tokens'] >= y_min) & 
                                            (threshold_subset['thinking_tokens'] <= y_max)]['thinking_tokens'].values
        if len(filtered_threshold) > 0:
            threshold_violin_data.append(filtered_threshold)
            threshold_labels.append(f"T:{threshold:.2f}")
            threshold_positions.append(pos_counter)
            threshold_colors.append('lightcoral')
            pos_counter += 1
    
    if threshold_violin_data:
        violin_parts4 = ax4.violinplot(threshold_violin_data, positions=threshold_positions, 
                                      showmeans=True, showmedians=True)
        
        for i, (pc, color) in enumerate(zip(violin_parts4['bodies'], threshold_colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax4.set_xticks(threshold_positions)
    ax4.set_xticklabels(threshold_labels, rotation=45, fontsize=9)
    ax4.set_title(f'Thinking Tokens Distribution: Base vs Rolling Thresholds (W{window_size})', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Thinking Tokens', fontsize=11)
    ax4.set_xlabel('Configuration', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thinking_tokens_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(results_df, output_dir='visualizations'):
    """Create various visualizations from the results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate data types
    rolling_df = results_df[results_df['window_type'] == 'rolling'].copy()
    control_df = results_df[results_df['threshold'].isin([0.0, 1.0])].copy()
    
    # Define marker styles for window sizes
    markers = {2: 'o', 5: 's', 10: '^'}
    
    # 1. Accuracy vs Threshold for different window sizes (with controls)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot rolling window data
    for window_size in sorted(rolling_df['window_size'].unique()):
        data = rolling_df[rolling_df['window_size'] == window_size].sort_values('threshold')
        ax.plot(data['threshold'], data['accuracy'], marker=markers.get(window_size, 'o'), 
                label=f'Rolling Window {window_size}', linewidth=2, markersize=8)
    
    # Add control reference lines only
    for _, control in control_df.iterrows():
        if control['threshold'] == 0.0:
            ax.axhline(y=control['accuracy'], color='red', linestyle='--', alpha=0.7, 
                      label=f"Base Thinking Control ({control['accuracy']:.1f}%)")
        elif control['threshold'] == 1.0:
            ax.axhline(y=control['accuracy'], color='orange', linestyle='--', alpha=0.7,
                      label=f"No Thinking Control ({control['accuracy']:.1f}%)")
    
    ax.set_xlabel('Entropy Threshold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Entropy Threshold by Window Size\n(with Control References)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Thinking Tokens vs Threshold (with controls)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for window_size in sorted(rolling_df['window_size'].unique()):
        data = rolling_df[rolling_df['window_size'] == window_size].sort_values('threshold')
        ax.plot(data['threshold'], data['avg_thinking_tokens'], marker=markers.get(window_size, 's'), 
                label=f'Rolling Window {window_size}', linewidth=2, markersize=8)
    
    # Add control reference lines only
    for _, control in control_df.iterrows():
        if control['threshold'] == 0.0:
            ax.axhline(y=control['avg_thinking_tokens'], color='red', linestyle='--', alpha=0.7,
                      label=f"Base Thinking ({control['avg_thinking_tokens']:.0f} tokens)")
        elif control['threshold'] == 1.0:
            ax.axhline(y=control['avg_thinking_tokens'], color='orange', linestyle='--', alpha=0.7,
                      label=f"No Thinking ({control['avg_thinking_tokens']:.0f} tokens)")
    
    ax.set_xlabel('Entropy Threshold', fontsize=12)
    ax.set_ylabel('Average Thinking Tokens', fontsize=12)
    ax.set_title('Average Thinking Tokens vs Entropy Threshold\n(with Control References)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thinking_tokens_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Generation Time vs Threshold (with controls)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for window_size in sorted(rolling_df['window_size'].unique()):
        data = rolling_df[rolling_df['window_size'] == window_size].sort_values('threshold')
        ax.plot(data['threshold'], data['avg_generation_time'], marker=markers.get(window_size, '^'), 
                label=f'Rolling Window {window_size}', linewidth=2, markersize=8)
    
    # Add control reference lines only
    for _, control in control_df.iterrows():
        if control['threshold'] == 0.0:
            ax.axhline(y=control['avg_generation_time'], color='red', linestyle='--', alpha=0.7,
                      label=f"Base Thinking ({control['avg_generation_time']:.1f}s)")
        elif control['threshold'] == 1.0:
            ax.axhline(y=control['avg_generation_time'], color='orange', linestyle='--', alpha=0.7,
                      label=f"No Thinking ({control['avg_generation_time']:.1f}s)")
    
    ax.set_xlabel('Entropy Threshold', fontsize=12)
    ax.set_ylabel('Average Generation Time (seconds)', fontsize=12)
    ax.set_title('Average Generation Time vs Entropy Threshold\n(with Control References)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generation_time_vs_threshold.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Heatmap of Accuracy (rolling only)
    pivot_accuracy = rolling_df.pivot(index='threshold', columns='window_size', values='accuracy')
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_accuracy, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Accuracy (%)'})
    ax.set_xlabel('Window Size', fontsize=12)
    ax.set_ylabel('Entropy Threshold', fontsize=12)
    ax.set_title('Accuracy Heatmap: Threshold vs Window Size\n(Rolling Windows Only)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=300)
    plt.close()
    
    # 5. Efficiency Plot: Accuracy vs Generation Time (with controls)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot rolling data with different markers for window sizes
    for window_size in sorted(rolling_df['window_size'].unique()):
        data = rolling_df[rolling_df['window_size'] == window_size]
        scatter = ax.scatter(data['avg_generation_time'], data['accuracy'], 
                           c=data['threshold'], marker=markers.get(window_size, 'o'),
                           s=100, alpha=0.7, cmap='viridis', 
                           label=f'Window {window_size}', edgecolors='black', linewidth=0.5)
    
    # Add control points
    for _, control in control_df.iterrows():
        if control['threshold'] == 0.0:
            ax.scatter([control['avg_generation_time']], [control['accuracy']], 
                      color='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                      label=f"Base Thinking Control", zorder=5)
        elif control['threshold'] == 1.0:
            ax.scatter([control['avg_generation_time']], [control['accuracy']], 
                      color='orange', s=200, marker='*', edgecolors='black', linewidth=2,
                      label=f"No Thinking Control", zorder=5)
    
    # Create colorbar for threshold
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=12)
    
    ax.set_xlabel('Average Generation Time (seconds)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Efficiency Plot: Accuracy vs Generation Time\n(Marker Shape = Window Size, Color = Threshold)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Tokens vs Accuracy Trade-off (with controls)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot rolling data with different markers for window sizes
    for window_size in sorted(rolling_df['window_size'].unique()):
        data = rolling_df[rolling_df['window_size'] == window_size]
        scatter = ax.scatter(data['avg_thinking_tokens'], data['accuracy'], 
                           c=data['threshold'], marker=markers.get(window_size, 'o'),
                           s=100, alpha=0.7, cmap='plasma', 
                           label=f'Window {window_size}', edgecolors='black', linewidth=0.5)
    
    # Add control points
    for _, control in control_df.iterrows():
        if control['threshold'] == 0.0:
            ax.scatter([control['avg_thinking_tokens']], [control['accuracy']], 
                      color='red', s=200, marker='*', edgecolors='black', linewidth=2, 
                      label=f"Base Thinking Control", zorder=5)
        elif control['threshold'] == 1.0:
            ax.scatter([control['avg_thinking_tokens']], [control['accuracy']], 
                      color='orange', s=200, marker='*', edgecolors='black', linewidth=2,
                      label=f"No Thinking Control", zorder=5)
    
    # Create colorbar for threshold
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold', fontsize=12)
    
    ax.set_xlabel('Average Thinking Tokens', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Token Efficiency: Accuracy vs Thinking Tokens\n(Marker Shape = Window Size, Color = Threshold)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tokens_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Thinking Tokens Distribution Analysis
    create_thinking_tokens_distribution(results_df, output_dir)
    
    # 8. Compare control vs rolling performance
    if not control_df.empty:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        control_acc = [control_df[control_df['threshold'] == 0.0]['accuracy'].iloc[0] if len(control_df[control_df['threshold'] == 0.0]) > 0 else 0,
                      control_df[control_df['threshold'] == 1.0]['accuracy'].iloc[0] if len(control_df[control_df['threshold'] == 1.0]) > 0 else 0]
        rolling_acc = rolling_df['accuracy'].mean()
        
        bars1 = ax1.bar(['Base Thinking\n(0.0)', 'No Thinking\n(1.0)', 'Rolling Avg'], 
                       control_acc + [rolling_acc], 
                       color=['red', 'orange', 'blue'], alpha=0.7)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Accuracy Comparison', fontsize=14)
        for bar, val in zip(bars1, control_acc + [rolling_acc]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Generation time comparison
        control_time = [control_df[control_df['threshold'] == 0.0]['avg_generation_time'].iloc[0] if len(control_df[control_df['threshold'] == 0.0]) > 0 else 0,
                       control_df[control_df['threshold'] == 1.0]['avg_generation_time'].iloc[0] if len(control_df[control_df['threshold'] == 1.0]) > 0 else 0]
        rolling_time = rolling_df['avg_generation_time'].mean()
        
        bars2 = ax2.bar(['Base Thinking\n(0.0)', 'No Thinking\n(1.0)', 'Rolling Avg'], 
                       control_time + [rolling_time], 
                       color=['red', 'orange', 'blue'], alpha=0.7)
        ax2.set_ylabel('Generation Time (seconds)', fontsize=12)
        ax2.set_title('Generation Time Comparison', fontsize=14)
        for bar, val in zip(bars2, control_time + [rolling_time]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Thinking tokens comparison
        control_tokens = [control_df[control_df['threshold'] == 0.0]['avg_thinking_tokens'].iloc[0] if len(control_df[control_df['threshold'] == 0.0]) > 0 else 0,
                         control_df[control_df['threshold'] == 1.0]['avg_thinking_tokens'].iloc[0] if len(control_df[control_df['threshold'] == 1.0]) > 0 else 0]
        rolling_tokens = rolling_df['avg_thinking_tokens'].mean()
        
        bars3 = ax3.bar(['Base Thinking\n(0.0)', 'No Thinking\n(1.0)', 'Rolling Avg'], 
                       control_tokens + [rolling_tokens], 
                       color=['red', 'orange', 'blue'], alpha=0.7)
        ax3.set_ylabel('Thinking Tokens', fontsize=12)
        ax3.set_title('Thinking Tokens Comparison', fontsize=14)
        for bar, val in zip(bars3, control_tokens + [rolling_tokens]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency scatter
        ax4.scatter([control_time[0]], [control_acc[0]], color='red', s=200, marker='*', 
                   edgecolors='black', linewidth=2, label='Base Thinking')
        if control_time[1] > 0:  # Check if no thinking control exists
            ax4.scatter([control_time[1]], [control_acc[1]], color='orange', s=200, marker='*', 
                       edgecolors='black', linewidth=2, label='No Thinking')
        ax4.scatter(rolling_df['avg_generation_time'], rolling_df['accuracy'], 
                   c='blue', alpha=0.6, s=60, label='Rolling Windows')
        ax4.set_xlabel('Generation Time (seconds)', fontsize=12)
        ax4.set_ylabel('Accuracy (%)', fontsize=12)
        ax4.set_title('Efficiency: Accuracy vs Time', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'control_vs_rolling.png'), dpi=300, bbox_inches='tight')
        plt.close()

def print_summary_table(results_df):
    """Print a formatted summary table"""
    # Sort by accuracy descending
    sorted_df = results_df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*140)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*140)
    print(f"{'Type':<8} {'Control':<12} {'Threshold':<10} {'Window':<8} {'Accuracy':<10} {'Thinking Tokens':<16} {'Gen Time (s)':<14} {'Tokens/s':<10}")
    print("-"*140)
    
    for _, row in sorted_df.iterrows():
        print(f"{row['window_type']:<8} {row['control_type']:<12} {row['threshold']:<10.2f} {row['window_size']:<8} "
              f"{row['accuracy']:<10.2f} {row['avg_thinking_tokens']:<16.2f} "
              f"{row['avg_generation_time']:<14.2f} {row['avg_tokens_per_second']:<10.2f}")
    
    print("="*140)
    
    # Print best configurations
    print("\nBEST CONFIGURATIONS:")
    print("-"*50)
    
    # Best accuracy
    best_accuracy = sorted_df.iloc[0]
    print(f"Highest Accuracy: {best_accuracy['accuracy']:.2f}% "
          f"(Threshold: {best_accuracy['threshold']}, Window: {best_accuracy['window_size']}, "
          f"Type: {best_accuracy['window_type']}, Control: {best_accuracy['control_type']})")
    
    # Fastest generation
    fastest = results_df.loc[results_df['avg_generation_time'].idxmin()]
    print(f"Fastest Generation: {fastest['avg_generation_time']:.2f}s "
          f"(Threshold: {fastest['threshold']}, Window: {fastest['window_size']}, Accuracy: {fastest['accuracy']:.2f}%)")
    
    # Most efficient (high accuracy with low time) - exclude controls for this calculation
    rolling_only = results_df[results_df['control_type'] == 'None']
    if not rolling_only.empty:
        rolling_only.loc[:, 'efficiency_score'] = rolling_only['accuracy'] / rolling_only['avg_generation_time']
        most_efficient = rolling_only.loc[rolling_only['efficiency_score'].idxmax()]
        print(f"Most Efficient (Rolling): {most_efficient['accuracy']:.2f}% in {most_efficient['avg_generation_time']:.2f}s "
              f"(Threshold: {most_efficient['threshold']}, Window: {most_efficient['window_size']})")
    
    # Control performance
    controls = results_df[results_df['control_type'] != 'None']
    if not controls.empty:
        print("\nCONTROL PERFORMANCE:")
        print("-"*30)
        for _, control in controls.iterrows():
            print(f"{control['control_type']}: {control['accuracy']:.2f}% accuracy, "
                  f"{control['avg_thinking_tokens']:.0f} thinking tokens, "
                  f"{control['avg_generation_time']:.2f}s generation time")

def main():
    # Directory containing the results
    results_dir = "/home/ixw/uncertainty-cot/results/run_1"
    
    print("Loading results from:", results_dir)
    results_df = load_all_results(results_dir)
    
    if results_df.empty:
        print("No results found!")
        return
    
    print(f"Loaded {len(results_df)} benchmark results")
    
    # Print summary table
    print_summary_table(results_df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(results_df)
    print("Visualizations saved to 'visualizations' directory")
    
    # Save summary to CSV for further analysis
    results_df.to_csv('benchmark_summary.csv', index=False)
    print("\nSummary data saved to 'benchmark_summary.csv'")

if __name__ == "__main__":
    main() 