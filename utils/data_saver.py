"""
Data saving and analysis utilities for the trust simulation.
"""
import json
import os
import csv
from typing import Dict, Any, List
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class DataSaver:
    """
    Utility class for saving simulation results and generating analysis.
    """
    
    def __init__(self, output_config: Dict[str, Any]):
        """
        Initialize data saver.
        
        Args:
            output_config: Output configuration
        """
        self.config = output_config
        self.results_dir = output_config.get("results_dir", "./results")
        self.save_interactions = output_config.get("save_interactions", True)
        self.save_agent_states = output_config.get("save_agent_states", True)
        self.format = output_config.get("format", "json")
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.results_dir, f"session_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
    
    async def save_results(self, results: Dict[str, Any]) -> None:
        """
        Save simulation results to files.
        
        Args:
            results: Simulation results dictionary
        """
        try:
            # Save main results as JSON
            results_file = os.path.join(self.session_dir, "simulation_results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save individual experiment results
            for experiment_name, experiment_data in results.items():
                if experiment_name in ["trust_game", "dictator_game", "mixed"]:
                    await self._save_experiment_results(experiment_name, experiment_data)
            
            print(f"Results saved to: {self.session_dir}")
        except Exception as e:
            print(f"Error saving results: {e}")
            # 尝试保存简化版本的结果
            try:
                simple_results_file = os.path.join(self.session_dir, "simulation_results_simple.json")
                with open(simple_results_file, 'w') as f:
                    json.dump({"error": str(e), "partial_results": str(results)}, f, indent=2)
                print(f"Partial results saved to: {simple_results_file}")
            except Exception as e2:
                print(f"Failed to save even simplified results: {e2}")
    
    async def _save_experiment_results(self, experiment_name: str, data: Dict[str, Any]) -> None:
        """
        Save individual experiment results.
        
        Args:
            experiment_name: Name of the experiment
            data: Experiment data
        """
        try:
            experiment_dir = os.path.join(self.session_dir, experiment_name)
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Save raw data
            data_file = os.path.join(experiment_dir, "raw_data.json")
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Save summary statistics
            if "summary_stats" in data:
                stats_file = os.path.join(experiment_dir, "summary_stats.json")
                with open(stats_file, 'w') as f:
                    json.dump(data["summary_stats"], f, indent=2)
            
            # Save rankings
            if "rankings" in data:
                rankings_file = os.path.join(experiment_dir, "rankings.json")
                with open(rankings_file, 'w') as f:
                    json.dump(data["rankings"], f, indent=2)
            
            # Save detailed results as CSV
            if "results" in data:
                await self._save_results_csv(experiment_name, data["results"], experiment_dir)
            
            # Save agent states if enabled
            if self.save_agent_states and "agent_states" in data:
                await self._save_agent_states(data["agent_states"], experiment_dir)
        except Exception as e:
            print(f"Error saving experiment results for {experiment_name}: {e}")
    
    async def _save_results_csv(self, experiment_name: str, results: List[Any], output_dir: str) -> None:
        """
        Save results as CSV file for analysis.
        
        Args:
            experiment_name: Name of the experiment
            results: List of result objects
            output_dir: Output directory
        """
        try:
            if not results:
                return
            
            # Convert results to list of dictionaries
            csv_data = []
            for result in results:
                if hasattr(result, '__dict__'):
                    csv_data.append(result.__dict__)
                elif isinstance(result, dict):
                    csv_data.append(result)
                else:
                    # Try to extract attributes
                    csv_data.append({
                        "data": str(result),
                        "type": type(result).__name__
                    })
            
            if csv_data:
                csv_file = os.path.join(output_dir, "results.csv")
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error saving CSV results for {experiment_name}: {e}")
    
    async def _save_agent_states(self, agent_states: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save agent states for analysis.
        
        Args:
            agent_states: List of agent state dictionaries
            output_dir: Output directory
        """
        try:
            states_file = os.path.join(output_dir, "agent_states.json")
            with open(states_file, 'w') as f:
                json.dump(agent_states, f, indent=2, default=str)
            
            # Create agent summary CSV
            summary_data = []
            for state in agent_states:
                summary = {
                    "name": state.get("name", ""),
                    "age": state.get("age", 0),
                    "education": state.get("education", ""),
                    "final_endowment": state.get("endowment", 0),
                    "total_beliefs": len(state.get("beliefs", [])),
                    "total_desires": len(state.get("desires", [])),
                    "total_intentions": len(state.get("intentions", [])),
                    "games_played": len(state.get("game_history", []))
                }
                
                # Add personality traits
                traits = state.get("personality_traits", {})
                for trait, value in traits.items():
                    summary[f"trait_{trait}"] = value
                
                # Add trust scores
                trust_scores = state.get("trust_scores", {})
                if trust_scores:
                    avg_trust = np.mean(list(trust_scores.values())) if trust_scores else 0
                    summary["avg_trust_score"] = avg_trust
                
                summary_data.append(summary)
            
            if summary_data:
                summary_file = os.path.join(output_dir, "agent_summary.csv")
                df = pd.DataFrame(summary_data)
                df.to_csv(summary_file, index=False)
        except Exception as e:
            print(f"Error saving agent states: {e}")
    
    async def save_analysis(self, analysis: Dict[str, Any]) -> None:
        """
        Save comprehensive analysis.
        
        Args:
            analysis: Analysis results
        """
        try:
            analysis_file = os.path.join(self.session_dir, "comprehensive_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Generate visualizations
            await self._generate_visualizations(analysis)
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    async def _generate_visualizations(self, analysis: Dict[str, Any]) -> None:
        """
        Generate visualization plots.
        
        Args:
            analysis: Analysis data
        """
        try:
            viz_dir = os.path.join(self.session_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Personality distribution plot
            if "personality_analysis" in analysis and "groups" in analysis["personality_analysis"]:
                await self._plot_personality_distribution(
                    analysis["personality_analysis"]["groups"], 
                    viz_dir
                )
            
            # Cross-experiment correlation plot
            if "cross_experiment_analysis" in analysis:
                await self._plot_cross_experiment_correlations(
                    analysis["cross_experiment_analysis"], 
                    viz_dir
                )
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    async def _plot_personality_distribution(self, personality_groups: Dict[str, List[str]], output_dir: str) -> None:
        """
        Plot personality trait distribution.
        
        Args:
            personality_groups: Groups by primary personality trait
            output_dir: Output directory
        """
        try:
            traits = list(personality_groups.keys())
            counts = [len(personality_groups[trait]) for trait in traits]
            
            plt.figure(figsize=(10, 6))
            plt.bar(traits, counts)
            plt.title('Distribution of Primary Personality Traits')
            plt.xlabel('Personality Trait')
            plt.ylabel('Number of Agents')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, "personality_distribution.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting personality distribution: {e}")
    
    async def _plot_cross_experiment_correlations(self, correlation_data: Dict[str, Any], output_dir: str) -> None:
        """
        Plot cross-experiment correlations.
        
        Args:
            correlation_data: Correlation data
            output_dir: Output directory
        """
        try:
            if "trust_generosity_correlation" not in correlation_data:
                return
            
            correlations = correlation_data["trust_generosity_correlation"]
            
            if correlations:
                agents = [c["agent"] for c in correlations]
                trust_profits = [c["trust_profit"] for c in correlations]
                generosity_scores = [c["generosity"] for c in correlations]
                
                plt.figure(figsize=(10, 6))
                plt.scatter(generosity_scores, trust_profits, alpha=0.7)
                plt.xlabel('Generosity Score (Dictator Game)')
                plt.ylabel('Trust Profit (Trust Game)')
                plt.title('Correlation between Generosity and Trust Profit')
                
                # Add trend line
                if len(generosity_scores) > 1:
                    z = np.polyfit(generosity_scores, trust_profits, 1)
                    p = np.poly1d(z)
                    plt.plot(generosity_scores, p(generosity_scores), "r--", alpha=0.8)
                
                # Add agent labels
                for i, agent in enumerate(agents):
                    plt.annotate(agent, (generosity_scores[i], trust_profits[i]), 
                               fontsize=8, alpha=0.7)
                
                plt.tight_layout()
                
                plot_file = os.path.join(output_dir, "trust_generosity_correlation.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error plotting cross-experiment correlations: {e}")
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report.
        
        Returns:
            Summary report as string
        """
        try:
            report_lines = [
                "Trust Behavior Simulation Summary Report",
                "=" * 50,
                f"Session Directory: {self.session_dir}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            # Add information about saved files
            report_lines.extend([
                "Saved Files:",
                f"- Main results: simulation_results.json",
                f"- Analysis: comprehensive_analysis.json",
                ""
            ])
            
            if os.path.exists(self.session_dir):
                for root, dirs, files in os.walk(self.session_dir):
                    level = root.replace(self.session_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    report_lines.append(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        report_lines.append(f"{subindent}{file}")
            
            return "\n".join(report_lines)
        except Exception as e:
            return f"Error generating summary report: {e}"