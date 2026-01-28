"""
RAGAS Results Visualization Script
Generates comprehensive graphs and charts from RAGAS evaluation results
"""

import json
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class RAGASVisualizer:
    """Generate visualizations from RAGAS evaluation results"""
    
    def __init__(self, results_file: str = None):
        """
        Initialize visualizer
        
        Args:
            results_file: Path to RAGAS results JSON. If None, uses latest.
        """
        if results_file is None:
            # Find latest results file
            results_files = glob.glob("evaluation_results/ragas_evaluation_*.json")
            if not results_files:
                raise FileNotFoundError("No RAGAS evaluation results found!")
            results_file = max(results_files, key=os.path.getctime)
            print(f"📂 Using latest results: {results_file}")
        
        self.results_file = results_file
        self.viz_dir = "evaluation_results/visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Load results
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.timestamp = self.data['timestamp']
        print(f"✅ Loaded results from {self.timestamp}")
        print(f"📊 Sample size: {self.data['sample_size']}")
    
    def create_overall_metrics_chart(self):
        """Create bar chart of overall metrics"""
        print("\n📊 Creating overall metrics chart...")
        
        metrics = self.data['overall_metrics']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        scores = list(metrics.values())
        
        # Create color map
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metric_names)))
        
        bars = ax.bar(metric_names, scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('RAGAS Evaluation Metrics - Overall Performance', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.viz_dir, f"overall_metrics_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_distribution_histograms(self):
        """Create distribution histograms for each metric"""
        print("\n📊 Creating metric distribution histograms...")
        
        # Extract individual scores
        detailed_results = self.data['detailed_results']
        
        metrics_data = {
            'faithfulness': [],
            'answer_relevancy': [],
            'context_precision': [],
            'context_recall': [],
            'answer_correctness': []
        }
        
        for result in detailed_results:
            for metric_name in metrics_data.keys():
                score = result['metrics'].get(metric_name)
                if score is not None:
                    metrics_data[metric_name].append(score)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, scores) in enumerate(metrics_data.items()):
            if len(scores) == 0:
                continue
            
            ax = axes[idx]
            
            # Create histogram
            ax.hist(scores, bins=20, color=plt.cm.viridis(idx/len(metrics_data)), 
                   edgecolor='black', alpha=0.7)
            
            # Add mean line
            mean_score = np.mean(scores)
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_score:.3f}')
            
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'{metric_name.replace("_", " ").title()}', 
                        fontweight='bold', fontsize=11)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle('Distribution of RAGAS Metrics Across Samples', 
                    fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output_path = os.path.join(self.viz_dir, f"metrics_distribution_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap between metrics"""
        print("\n📊 Creating correlation heatmap...")
        
        # Extract metrics into DataFrame
        detailed_results = self.data['detailed_results']
        
        metrics_list = []
        for result in detailed_results:
            metrics_list.append(result['metrics'])
        
        df = pd.DataFrame(metrics_list)
        
        # Calculate correlation
        correlation = df.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Correlation Between RAGAS Metrics', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.viz_dir, f"correlation_heatmap_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_box_plots(self):
        """Create box plots for metric comparison"""
        print("\n📊 Creating box plots...")
        
        # Extract metrics into DataFrame
        detailed_results = self.data['detailed_results']
        
        metrics_list = []
        for result in detailed_results:
            metrics_list.append(result['metrics'])
        
        df = pd.DataFrame(metrics_list)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df.boxplot(ax=ax, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df.columns)))
        for patch, color in zip(ax.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('RAGAS Metrics - Box Plot Comparison', 
                    fontweight='bold', fontsize=14, pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(self.viz_dir, f"metrics_boxplot_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_detailed_table(self):
        """Create detailed statistics table"""
        print("\n📊 Creating detailed statistics table...")
        
        # Extract metrics into DataFrame
        detailed_results = self.data['detailed_results']
        
        metrics_list = []
        for result in detailed_results:
            metrics_list.append(result['metrics'])
        
        df = pd.DataFrame(metrics_list)
        
        # Calculate statistics
        stats = df.describe().T
        stats['median'] = df.median()
        stats = stats[['mean', 'median', 'std', 'min', 'max']]
        
        # Create table visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Format the data
        table_data = []
        for metric in stats.index:
            row = [metric.replace('_', ' ').title()]
            row.extend([f"{val:.4f}" for val in stats.loc[metric]])
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(6):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(6):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('RAGAS Metrics - Detailed Statistics', 
                 fontweight='bold', fontsize=14, pad=20)
        
        output_path = os.path.join(self.viz_dir, f"statistics_table_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def create_performance_radar(self):
        """Create radar chart for overall performance"""
        print("\n📊 Creating performance radar chart...")
        
        metrics = self.data['overall_metrics']
        
        # Prepare data
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw the plot
        ax.plot(angles, values, 'o-', linewidth=2, color='#2E86AB', label='Scores')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        
        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', '\n').title() for cat in categories], 
                          fontsize=10, fontweight='bold')
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.title('RAGAS Metrics - Performance Radar', 
                 fontweight='bold', fontsize=14, pad=30)
        
        output_path = os.path.join(self.viz_dir, f"performance_radar_{self.timestamp}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_path}")
        plt.close()
    
    def generate_html_report(self):
        """Generate HTML report with all visualizations"""
        print("\n📄 Generating HTML report...")
        
        # Get all visualization files
        viz_files = glob.glob(os.path.join(self.viz_dir, f"*_{self.timestamp}.png"))
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAGAS Evaluation Report - {self.timestamp}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2E86AB;
                    text-align: center;
                    border-bottom: 3px solid #2E86AB;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #333;
                    margin-top: 30px;
                }}
                .metric-summary {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .metric-item {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .metric-name {{
                    font-weight: bold;
                    color: #555;
                }}
                .metric-value {{
                    color: #2E86AB;
                    font-weight: bold;
                    font-size: 1.1em;
                }}
                .visualization {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    text-align: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #777;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <h1>🏛️ Legal Chatbot - RAGAS Evaluation Report</h1>
            
            <div class="metric-summary">
                <h2>📊 Overall Metrics Summary</h2>
                <p><strong>Evaluation Date:</strong> {self.timestamp}</p>
                <p><strong>Sample Size:</strong> {self.data['sample_size']} questions</p>
                <hr>
        """
        
        # Add metrics
        for metric, score in self.data['overall_metrics'].items():
            html_content += f"""
                <div class="metric-item">
                    <span class="metric-name">{metric.replace('_', ' ').title()}</span>
                    <span class="metric-value">{score:.4f}</span>
                </div>
            """
        
        html_content += """
            </div>
            
            <h2>📈 Visualizations</h2>
        """
        
        # Add visualizations
        for viz_file in sorted(viz_files):
            viz_name = os.path.basename(viz_file).replace(f"_{self.timestamp}.png", "").replace("_", " ").title()
            rel_path = os.path.relpath(viz_file, self.viz_dir)
            
            html_content += f"""
            <div class="visualization">
                <h3>{viz_name}</h3>
                <img src="{rel_path}" alt="{viz_name}">
            </div>
            """
        
        html_content += f"""
            <div class="footer">
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Legal Advisor Chatbot - RAGAS Evaluation System</p>
            </div>
        </body>
        </html>
        """
        
        # Save HTML
        html_path = os.path.join(self.viz_dir, f"evaluation_report_{self.timestamp}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report saved: {html_path}")
        return html_path
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*60)
        print("🎨 GENERATING RAGAS VISUALIZATIONS")
        print("="*60)
        
        self.create_overall_metrics_chart()
        self.create_distribution_histograms()
        self.create_correlation_heatmap()
        self.create_box_plots()
        self.create_detailed_table()
        self.create_performance_radar()
        html_path = self.generate_html_report()
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📂 Visualizations saved in: {self.viz_dir}")
        print(f"📄 Open HTML report: {html_path}")
        print("\n")


def main(results_file: str = None):
    """Main execution function"""
    visualizer = RAGASVisualizer(results_file)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations from RAGAS results")
    parser.add_argument("--results-file", type=str, default=None, 
                       help="Path to RAGAS results JSON (uses latest if not specified)")
    
    args = parser.parse_args()
    
    main(results_file=args.results_file)
