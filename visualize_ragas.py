import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_results(file_path="ragas_evaluation_results.json"):
    """Load evaluation results from JSON"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualizations(data):
    """Create visualization charts for RAGAS metrics"""
    metrics = data['metrics']
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RAGAS Evaluation Results - Indian Legal Assistant', fontsize=16, fontweight='bold')
    
    # 1. Overall Metrics Bar Chart
    ax1 = axes[0, 0]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax1.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Overall RAGAS Metrics', fontsize=14, fontweight='bold')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Fair (0.5)')
    ax1.legend()
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Metric Comparison Radar Chart
    ax2 = axes[0, 1]
    ax2.axis('off')
    
    # Create radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    angles = [n / len(categories) * 2 * 3.14159 for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]
    
    ax_radar = plt.subplot(2, 2, 2, projection='polar')
    ax_radar.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax_radar.fill(angles, values, alpha=0.25, color='#3498db')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, size=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Metric Distribution', fontsize=14, fontweight='bold', pad=20)
    ax_radar.grid(True)
    
    # 3. Performance Summary Table
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create performance interpretation
    performance_data = []
    for metric, value in metrics.items():
        if value >= 0.7:
            status = '✓ Good'
            color = '#d4edda'
        elif value >= 0.5:
            status = '⚠ Fair'
            color = '#fff3cd'
        else:
            status = '✗ Needs Improvement'
            color = '#f8d7da'
        
        performance_data.append([metric.replace('_', ' ').title(), f'{value:.4f}', status])
    
    table = ax3.table(cellText=performance_data,
                     colLabels=['Metric', 'Score', 'Status'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.4, 0.2, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color code the status column
    for i in range(1, len(performance_data) + 1):
        value = float(performance_data[i-1][1])
        if value >= 0.7:
            color = '#d4edda'
        elif value >= 0.5:
            color = '#fff3cd'
        else:
            color = '#f8d7da'
        table[(i, 2)].set_facecolor(color)
    
    # Header styling
    for i in range(3):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    # 4. Metric Insights
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insights_text = "Key Insights:\n\n"
    
    # Generate insights
    avg_score = sum(metrics.values()) / len(metrics)
    insights_text += f"• Average Score: {avg_score:.3f}\n\n"
    
    best_metric = max(metrics.items(), key=lambda x: x[1])
    insights_text += f"• Best Metric: {best_metric[0].replace('_', ' ').title()}\n  ({best_metric[1]:.3f})\n\n"
    
    worst_metric = min(metrics.items(), key=lambda x: x[1])
    insights_text += f"• Needs Focus: {worst_metric[0].replace('_', ' ').title()}\n  ({worst_metric[1]:.3f})\n\n"
    
    # Recommendations
    insights_text += "Recommendations:\n"
    if metrics['faithfulness'] < 0.7:
        insights_text += "• Improve answer accuracy\n"
    if metrics['answer_relevancy'] < 0.7:
        insights_text += "• Enhance answer relevance\n"
    if metrics['context_precision'] < 0.7:
        insights_text += "• Refine retrieval precision\n"
    if metrics['context_recall'] < 0.7:
        insights_text += "• Improve context coverage\n"
    
    if avg_score >= 0.7:
        insights_text += "• Overall: Excellent performance! ✓"
    
    ax4.text(0.1, 0.9, insights_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ragas_visualization_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {filename}")
    
    plt.show()

def generate_report(data):
    """Generate detailed text report"""
    print("\n" + "=" * 70)
    print("DETAILED RAGAS EVALUATION REPORT")
    print("=" * 70)
    
    metrics = data['metrics']
    results = data['results']
    
    print(f"\nTotal Questions Evaluated: {len(results)}")
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-" * 70)
    print("METRIC SCORES")
    print("-" * 70)
    
    for metric, value in metrics.items():
        status = "✓" if value >= 0.7 else "⚠" if value >= 0.5 else "✗"
        print(f"{status} {metric.replace('_', ' ').title():.<50} {value:.4f}")
    
    avg_score = sum(metrics.values()) / len(metrics)
    print(f"\n{'Average Score':.<50} {avg_score:.4f}")
    
    print("\n" + "-" * 70)
    print("SAMPLE RESULTS (First 3)")
    print("-" * 70)
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n[{i}] Question: {result['question']}")
        print(f"    Ground Truth: {result['ground_truth']}")
        print(f"    RAG Answer: {result['answer'][:150]}...")
        print(f"    Contexts Retrieved: {len(result['contexts'])}")
    
    print("\n" + "=" * 70)

def main():
    print("Loading RAGAS evaluation results...")
    
    try:
        data = load_results()
        print("✓ Results loaded successfully\n")
        
        # Generate visualizations
        print("Creating visualizations...")
        create_visualizations(data)
        
        # Generate text report
        generate_report(data)
        
    except FileNotFoundError:
        print("✗ Error: ragas_evaluation_results.json not found")
        print("  Please run evaluate_ragas.py first")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    main()
