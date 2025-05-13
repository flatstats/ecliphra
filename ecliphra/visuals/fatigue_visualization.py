"""
Fixed visualization utilities for fatigue dynamics in the Ecliphra system.

This module provides tools to visualize:
1. Fatigue level over time
2. Goal completion and fatigue recovery events
3. Fatigue components (G, ΔS, Rₘ)
4. Impact of fatigue on goal priorities
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_fatigue_dynamics(prefrontal_module, output_dir="fatigue_visualizations"):
    """
    Create visualizations of fatigue dynamics from a prefrontal module.
    
    Args:
        prefrontal_module: A PrefrontalModule instance with fatigue tracking
        output_dir: Directory to save visualizations
    """
    if not hasattr(prefrontal_module, 'fatigue_controller'):
        print("No fatigue controller found in prefrontal module. Visualization skipped.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get fatigue history data
    fatigue_history = prefrontal_module.fatigue_controller.fatigue_history
    if not fatigue_history:
        print("No fatigue history found. Run the model first.")
        return
    
    # Extract goal completion history if available
    goal_completions = []
    completion_steps = []
    
    for i, goal in enumerate(prefrontal_module.fatigue_controller.goal_completion_history):
        completion_steps.append(i)
        goal_completions.append(goal.get("restoration", 0))
    
    # 1. Plot fatigue level over time
    plt.figure(figsize=(12, 6))
    steps = range(len(fatigue_history))
    plt.plot(steps, fatigue_history, 'b-', linewidth=2, label='Fatigue Level')
    
    # Highlight goal completions if available
    if completion_steps:
        # Find fatigue values at completion steps
        completion_fatigue = [fatigue_history[min(s, len(fatigue_history)-1)] for s in completion_steps]
        plt.scatter(completion_steps, completion_fatigue, color='green', s=100, marker='o', 
                   label='Goal Completion')
        
        # Add annotations for recovery amounts
        for x, y, recovery in zip(completion_steps, completion_fatigue, goal_completions):
            if x < len(fatigue_history):
                plt.annotate(f"-{recovery:.2f}", (x, y), 
                            xytext=(10, -20), textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add thresholds
    plt.axhline(y=10.0, color='orange', linestyle='--', label='Rebalancing Threshold')
    plt.axhline(y=20.0, color='red', linestyle='--', label='High Fatigue Threshold')
    
    plt.xlabel('Step')
    plt.ylabel('Fatigue Level')
    plt.title('Fatigue Level Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'fatigue_over_time.png'))
    plt.close()
    
    # 2. Plot fatigue components if debug info is available
    if hasattr(prefrontal_module.fatigue_controller, 'debug_info'):
        debug_info = prefrontal_module.fatigue_controller.debug_info
        
        # Check if we have component information
        if 'goal_intensity' in debug_info and 'shift_magnitude' in debug_info and 'residual_meaning' in debug_info:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart of components
            components = ['G (Goal Intensity)', 'ΔS (Shift Magnitude)', 'Rₘ (Residual Meaning)']
            values = [
                debug_info['goal_intensity'],
                debug_info['shift_magnitude'],
                debug_info['residual_meaning']
            ]
            
            colors = ['blue', 'orange', 'green']
            if debug_info['residual_meaning'] < 0:
                # Use green for negative residual meaning (recovery)
                colors[2] = 'green'
            else:
                # Use red for positive residual meaning (additional fatigue)
                colors[2] = 'red'
            
            plt.bar(components, values, color=colors)
            
            # Add horizontal line at y=0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add fatigue formula
            equation = r"$\Psi = (G \times \Delta S) + R_m$"
            plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction', 
                      fontsize=16, ha='center', va='top')
            
            # Calculate total
            result = (debug_info['goal_intensity'] * debug_info['shift_magnitude']) + debug_info['residual_meaning']
            plt.annotate(f"Result: {result:.3f}", xy=(0.5, 0.9), xycoords='axes fraction', 
                      fontsize=12, ha='center', va='top')
            
            plt.ylabel('Value')
            plt.title('Fatigue Formula Components')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fatigue_components.png'))
            plt.close()
    
    # 3. Plot the effect of fatigue on goals
    if hasattr(prefrontal_module, 'current_goals') and prefrontal_module.current_goals:
        plt.figure(figsize=(12, 6))
        
        # Extract goal data
        goal_types = []
        priorities = []
        activations = []
        progress = []
        
        for goal in prefrontal_module.current_goals:
            goal_types.append(goal.get('type', 'generic'))
            priorities.append(goal.get('priority', 0.5))
            activations.append(goal.get('activation', 1.0))
            progress.append(goal.get('progress', 0.0))
        
        # Number of goals
        n_goals = len(goal_types)
        ind = np.arange(n_goals)
        width = 0.2
        
        # Create grouped bar chart
        plt.bar(ind, priorities, width, label='Priority', color='blue')
        plt.bar(ind + width, activations, width, label='Activation', color='green')
        plt.bar(ind + width*2, progress, width, label='Progress', color='orange')
        
        # Add fatigue level indicator
        fatigue_level = prefrontal_module.fatigue_level if hasattr(prefrontal_module, 'fatigue_level') else 0
        plt.axhline(y=fatigue_level/100, color='red', linestyle='--', 
                   label=f'Fatigue Level ({fatigue_level:.1f}/100)')
        
        plt.xlabel('Goal')
        plt.ylabel('Value')
        plt.title('Current Goals State (Modified by Fatigue)')
        plt.xticks(ind + width, goal_types)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goals_fatigue_impact.png'))
        plt.close()

    # 4. Create energy efficiency visualization
    if hasattr(prefrontal_module, 'decision_history') and len(prefrontal_module.decision_history) > 5:
        plt.figure(figsize=(12, 6))
        
        # Extract energy data
        energy_data = []
        for i, decision in enumerate(prefrontal_module.decision_history[-20:]):
            if 'modified_distribution' in decision:
                energy_data.append(decision['modified_distribution'].get('total', 0.0))
            elif 'energy_distribution' in decision:
                energy_data.append(decision['energy_distribution'].get('total', 0.0))
            else:
                energy_data.append(0.0)
        
        # Plot energy use
        x = range(len(energy_data))
        plt.plot(x, energy_data, 'g-', linewidth=2, label='Energy Used')
        
        # Plot fatigue (scaled to fit on same graph)
        recent_fatigue = fatigue_history[-min(len(fatigue_history), len(energy_data)):]
        if recent_fatigue:
            max_fatigue = max(recent_fatigue) or 1.0
            max_energy = max(energy_data) or 1.0
            scaled_fatigue = [f * (max_energy / max_fatigue) for f in recent_fatigue]
            plt.plot(x[-len(scaled_fatigue):], scaled_fatigue, 'r-', linewidth=2, label='Fatigue (Scaled)')
        
        # Calculate efficiency trend
        if energy_data and len(energy_data) > 1 and recent_fatigue:
            efficiency_trend = []
            for i in range(min(len(energy_data), len(recent_fatigue))):
                if energy_data[i] > 0:
                    eff = recent_fatigue[i] / energy_data[i]
                    efficiency_trend.append(eff)
                else:
                    efficiency_trend.append(0)
            
            if efficiency_trend:
                plt.plot(x[-len(efficiency_trend):], efficiency_trend, 'b--', linewidth=1.5, label='Fatigue/Energy Ratio')
        
        plt.xlabel('Recent Steps')
        plt.ylabel('Value')
        plt.title('Energy Consumption vs. Fatigue')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'energy_vs_fatigue.png'))
        plt.close()

def create_fatigue_debug_dashboard(prefrontal_module, output_dir="fatigue_debug"):
    """
    Create a diagnostic dashboard for debugging fatigue dynamics.
    
    Args:
        prefrontal_module: A PrefrontalModule instance with fatigue tracking
        output_dir: Directory to save visualizations
    """
    if not hasattr(prefrontal_module, 'fatigue_controller'):
        print("No fatigue controller found in prefrontal module. Debug dashboard skipped.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get controller status
    fatigue_controller_status = prefrontal_module.fatigue_controller.get_status()
    
    # Create a text file with detailed status
    with open(os.path.join(output_dir, 'fatigue_debug.txt'), 'w') as f:
        f.write("FATIGUE CONTROLLER DEBUG INFORMATION\n")
        f.write("====================================\n\n")
        
        f.write(f"Current Fatigue Level: {fatigue_controller_status['fatigue_level']:.2f}\n")
        f.write(f"Completed Goals: {fatigue_controller_status['completed_goals']}\n\n")
        
        f.write("Recent Fatigue History:\n")
        for i, level in enumerate(fatigue_controller_status['fatigue_history']):
            f.write(f"  Step {i}: {level:.2f}\n")
        
        f.write("\nDebug Info:\n")
        for key, value in fatigue_controller_status.get('debug_info', {}).items():
            f.write(f"  {key}: {value}\n")
    
    # Create a bar chart showing fatigue components
    debug_info = fatigue_controller_status.get('debug_info', {})
    if 'goal_intensity' in debug_info and 'shift_magnitude' in debug_info and 'residual_meaning' in debug_info:
        plt.figure(figsize=(10, 6))
        
        # Create bar chart of components
        components = ['G × ΔS', 'Rₘ', 'Total']
        
        product = debug_info['goal_intensity'] * debug_info['shift_magnitude']
        residual = debug_info['residual_meaning']
        total = product + residual
        
        values = [product, residual, total]
        
        colors = ['blue', 'green' if residual < 0 else 'red', 'purple']
        
        plt.bar(components, values, color=colors)
        
        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.ylabel('Value')
        plt.title('Fatigue Calculation Components')
        plt.savefig(os.path.join(output_dir, 'fatigue_calculation.png'))
        plt.close()
    
    # Create a visual representation of goal completions
    goal_completions = prefrontal_module.fatigue_controller.goal_completion_history
    if goal_completions:
        plt.figure(figsize=(12, 6))
        
        goal_types = [g.get('goal_type', 'unknown') for g in goal_completions]
        priorities = [g.get('priority', 0.5) for g in goal_completions]
        restorations = [g.get('restoration', 0.0) for g in goal_completions]
        
        # Plot as scatter with size based on restoration
        plt.scatter(range(len(goal_completions)), priorities, 
                   s=[r*500 for r in restorations], 
                   alpha=0.7, c=range(len(goal_completions)), cmap='viridis')
        
        plt.xlabel('Completion Index')
        plt.ylabel('Goal Priority')
        plt.title('Goal Completions and Their Restorative Effect')
        
        # Add text labels
        for i, (gtype, rest) in enumerate(zip(goal_types, restorations)):
            plt.annotate(f"{gtype}: -{rest:.2f}", (i, priorities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'goal_completions.png'))
        plt.close()
        
    # Create a plot showing fatigue recovery impact
    fatigue_history = fatigue_controller_status.get('fatigue_history', [])
    if len(fatigue_history) > 1:
        plt.figure(figsize=(12, 6))
        
        # Calculate fatigue changes
        fatigue_changes = [j-i for i, j in zip(fatigue_history[:-1], fatigue_history[1:])]
        
        # Plot changes as a bar chart
        plt.bar(range(len(fatigue_changes)), fatigue_changes, 
               color=['green' if x < 0 else 'red' for x in fatigue_changes])
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.xlabel('Step')
        plt.ylabel('Fatigue Change')
        plt.title('Fatigue Changes Over Time (Negative = Recovery)')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fatigue_changes.png'))
        plt.close()
    
    # Create a heatmap showing fatigue level vs. goal progress
    if hasattr(prefrontal_module, 'fatigue_level') and hasattr(prefrontal_module, 'current_goals') and prefrontal_module.current_goals:
        # Extract current goals
        goal_types = [g.get('type', 'unknown') for g in prefrontal_module.current_goals]
        progress_values = [g.get('progress', 0.0) for g in prefrontal_module.current_goals]
        
        # Get fatigue level
        fatigue_level = prefrontal_module.fatigue_level
        
        # Create comparison visualization
        plt.figure(figsize=(10, 6))
        
        y_pos = np.arange(len(goal_types))
        progress_bars = plt.barh(y_pos, progress_values, color='blue', alpha=0.7)
        
        # Add a vertical line for fatigue (scaled to 0-1)
        scaled_fatigue = min(1.0, fatigue_level / 30.0)  # Scale to 0-1 range assuming 30 is max
        plt.axvline(x=scaled_fatigue, color='red', linestyle='--', 
                   label=f'Fatigue ({fatigue_level:.1f})')
        
        plt.yticks(y_pos, goal_types)
        plt.xlabel('Value')
        plt.xlim(0, 1.0)
        plt.title('Goal Progress vs. Fatigue Level')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'progress_vs_fatigue.png'))
        plt.close()
    
    # Create a visualization of current fatigue components
    if 'debug_info' in fatigue_controller_status:
        # Create a radar/spider chart of fatigue factors
        factors = ['Goal Intensity', 'Shift Magnitude', 'Fatigue Increment', 'Effort Efficiency']
        
        # Extract values or use defaults
        goal_intensity = debug_info.get('goal_intensity', 0.0) 
        shift_magnitude = debug_info.get('shift_magnitude', 0.0)
        fatigue_increment = debug_info.get('fatigue_increment', 0.0)
        
        # Calculate effort efficiency (inverse of fatigue per unit energy)
        # Higher is better - represents how effectively energy is used
        effort_efficiency = 0.5  # Default
        if 'goal_intensity' in debug_info and debug_info['goal_intensity'] > 0:
            if fatigue_increment > 0:
                effort_efficiency = 1.0 - min(1.0, fatigue_increment / debug_info['goal_intensity'])
            else:
                effort_efficiency = 1.0  # Perfect efficiency (fatigue decreasing)
        
        values = [goal_intensity, shift_magnitude, max(0, fatigue_increment), effort_efficiency]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(factors), endpoint=False).tolist()
        values += values[:1]  # Close the loop
        angles += angles[:1]  # Close the loop
        factors += factors[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), factors[:-1])
        
        ax.set_ylim(0, 1)
        ax.grid(True)
        
        plt.title('Fatigue Component Analysis')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fatigue_radar.png'))
        plt.close()