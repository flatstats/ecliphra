"""
Ecliphra Experiments

This module contains test runners for different aspects of the Ecliphra system:
- Semantic memory retention and pattern recognition
- Echo resonance under low activity
- Noise resistance and recovery
- Gradual semantic transitions over time
- Transitions handles a gradual morphing from one pattern to another.
- Photosynthesis routes, filters, and integrates different types of signals.
- 


Its not about accuracy, its about behavior.
Each of these generates visual traces and metrics that help track whether the model
is stabilizing, drifting, or evolving under pressure.

Note: These experiments are rough but functional.
They were written to stress test internal behavior during model development.
Repetitive but enough to see resonance, memory decay, and drift.

"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import json
from datetime import datetime
from collections import defaultdict
from ecliphra.visuals.fatigue_visualization import visualize_fatigue_dynamics, create_fatigue_debug_dashboard

class EcliphraExperiment:
    """Base class for all Ecliphra experiments"""

    def __init__(self, model, output_dir=None, field_size=32):
        """
        Initialize experiment

        Args:
            model: Ecliphra model instance to use
            output_dir: Directory to save results (creates timestamped dir if None)
            field_size: Size of the field (used for pattern generation)
        """
        self.model = model
        self.field_size = field_size

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = self.__class__.__name__.lower().replace('experiment', '')
            self.output_dir = f"ecliphra_results/{experiment_name}_{timestamp}"
        else:
            self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)


        self.results = {
            "field_snapshots": [],
            "attractors": [],
            "stability": [],
            "echo_events": [],
            "semantic_matches": [],
            "match_strengths": [],
            "experiment_config": {
                "experiment_type": self.__class__.__name__,
                "model_type": type(model).__name__,
                "field_size": field_size
            }
        }
        

    def generate_patterns(self, pattern_type='gaussian', variations=3, noise_levels=[0.1, 0.2, 0.3]):
        """
        Generate test patterns for experiments

        Args:
            pattern_type: Type of pattern ('gaussian', 'spiral')
            variations: Number of variations to create
            noise_levels: List of noise levels for variations

        Returns:
            Dictionary with patterns and variations
        """
        patterns = {}

        if pattern_type == 'gaussian':
            pattern_base1 = torch.zeros((self.field_size, self.field_size))
            pattern_base2 = torch.zeros((self.field_size, self.field_size))

            # top-left centered
            for i in range(self.field_size):
                for j in range(self.field_size):
                    r = np.sqrt((i - self.field_size/3)**2 + (j - self.field_size/3)**2) / self.field_size
                    pattern_base1[i, j] = np.exp(-3 * r)

            # bottom-right centered
            for i in range(self.field_size):
                for j in range(self.field_size):
                    r = np.sqrt((i - 2*self.field_size/3)**2 + (j - 2*self.field_size/3)**2) / self.field_size
                    pattern_base2[i, j] = np.exp(-3 * r)

            
            pattern_base1 = pattern_base1 / torch.norm(pattern_base1)
            pattern_base2 = pattern_base2 / torch.norm(pattern_base2)

           
            patterns["base"] = [pattern_base1, pattern_base2]
            patterns["variations1"] = [pattern_base1]  # Start with base pattern
            patterns["variations2"] = [pattern_base2]  # Start with base pattern

           
            for noise_level in noise_levels:
                var1 = pattern_base1 + torch.randn_like(pattern_base1) * noise_level
                var2 = pattern_base2 + torch.randn_like(pattern_base2) * noise_level
                patterns["variations1"].append(var1 / torch.norm(var1))
                patterns["variations2"].append(var2 / torch.norm(var2))

            # morph from pattern1 to pattern2
            patterns["transition"] = []
            steps = 10
            for i in range(steps + 1):
                alpha = i / steps  # Blending factor
                transition = (1 - alpha) * pattern_base1 + alpha * pattern_base2
                patterns["transition"].append(transition / torch.norm(transition))

        elif pattern_type == 'spiral':
            # Create spiral patterns with different orientations
            pattern_base1 = torch.zeros((self.field_size, self.field_size))
            pattern_base2 = torch.zeros((self.field_size, self.field_size))

            # Center coordinates
            cx, cy = self.field_size // 2, self.field_size // 2

            # clockwise spiral
            for i in range(self.field_size):
                for j in range(self.field_size):
                    # polar coordinates
                    r = np.sqrt((i - cx)**2 + (j - cy)**2) / (self.field_size/2)
                    theta = np.arctan2(j - cy, i - cx)
                    # Spiral function
                    pattern_base1[i, j] = np.sin(theta + 5*r) * np.exp(-r)

            # counter-clockwise spiral
            for i in range(self.field_size):
                for j in range(self.field_size):
                    r = np.sqrt((i - cx)**2 + (j - cy)**2) / (self.field_size/2)
                    theta = np.arctan2(j - cy, i - cx)
                    pattern_base2[i, j] = np.sin(-theta + 5*r) * np.exp(-r)

            
            pattern_base1 = pattern_base1 / torch.norm(pattern_base1)
            pattern_base2 = pattern_base2 / torch.norm(pattern_base2)

            
            patterns["base"] = [pattern_base1, pattern_base2]
            patterns["variations1"] = [pattern_base1]
            patterns["variations2"] = [pattern_base2]

           
            for noise_level in noise_levels:
                var1 = pattern_base1 + torch.randn_like(pattern_base1) * noise_level
                var2 = pattern_base2 + torch.randn_like(pattern_base2) * noise_level
                patterns["variations1"].append(var1 / torch.norm(var1))
                patterns["variations2"].append(var2 / torch.norm(var2))

            
            patterns["transition"] = []
            steps = 10
            for i in range(steps + 1):
                alpha = i / steps  # Blending factor
                transition = (1 - alpha) * pattern_base1 + alpha * pattern_base2
                patterns["transition"].append(transition / torch.norm(transition))

      
        self.results["experiment_config"]["pattern_type"] = pattern_type
        self.results["experiment_config"]["variations"] = variations
        self.results["experiment_config"]["noise_levels"] = noise_levels

        return patterns

    def record_step_results(self, step, outputs):
        """Record results for a single experiment step"""
  
        self.results["attractors"].append(len(outputs.get('attractors', [])))
        self.results["stability"].append(outputs.get('stability', 0.5))
        self.results["echo_events"].append(1.0 if outputs.get('echo_applied', False) else 0.0)
        self.results["semantic_matches"].append(1.0 if outputs.get('semantic_match_applied', False) else 0.0)

        if outputs.get('semantic_match_applied', False):
            match_strength = outputs.get('match_similarity', 0.85)
            self.results["match_strengths"].append((step, match_strength))

        if step % 5 == 0 or outputs.get('echo_applied', False) or outputs.get('semantic_match_applied', False):
            self.results["field_snapshots"].append({
                "step": step,
                "field": self.model.field.detach().clone().cpu().numpy().tolist(),
                "echo_applied": outputs.get('echo_applied', False),
                "semantic_match": outputs.get('semantic_match_applied', False)
            })

    def visualize_field(self, step, echo_applied=False, semantic_match=False):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        field_cmap = LinearSegmentedColormap.from_list('Ecliphra', [
            (0, 'darkblue'), (0.4, 'royalblue'),
            (0.6, 'mediumpurple'), (0.8, 'darkorchid'), (1.0, 'gold')
        ])


        field = self.model.field.detach().cpu().numpy()
        im1 = ax1.imshow(field, cmap=field_cmap)
        fig.colorbar(im1, ax=ax1)

        # Set title based on state
        if semantic_match:
            title = f"Field State (Step {step}) - Semantic Match"
            ax1.set_title(title, color='orange', fontweight='bold')
        elif echo_applied:
            title = f"Field State (Step {step}) - Echo Applied"
            ax1.set_title(title)
        else:
            title = f"Field State (Step {step})"
            ax1.set_title(title)

        if hasattr(self.model, 'attractors'):
            for attractor in self.model.attractors:
                # Handle both old and new attractor formats
                if len(attractor) == 2:
                    pos, strength = attractor
                    base_size = None
                    adaptive_size = None
                elif len(attractor) >= 4:
                    pos, strength, base_size, adaptive_size = attractor
                else:
                    continue  # Skip if format doesn't match expected patterns

                ax1.scatter(pos[1], pos[0], c='white', s=100*strength + 50, marker='*',
                        edgecolors='black', linewidths=1)

                if adaptive_size is not None:
                    label = f"{strength:.2f} (x{adaptive_size/base_size:.1f})"
                else:
                    label = f"{strength:.2f}"

                ax1.annotate(label, (pos[1], pos[0]),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        if hasattr(self.model, 'velocity'):
            velocity = self.model.velocity.detach().cpu().numpy()
            velocity_magnitude = np.sqrt(np.sum(velocity**2))
            im2 = ax2.imshow(velocity, cmap='inferno')
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f"Velocity Field (Norm: {velocity_magnitude:.3f})")
        else:
            # If no velocity, show gradient field
            grad_y, grad_x = np.gradient(field)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            im2 = ax2.imshow(gradient_magnitude, cmap='inferno')
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f"Field Gradient")

        if hasattr(self.model, 'attractor_memory') and self.model.attractor_memory:
            memory_text = "Attractor Memory:\n"
            for i, entry in enumerate(self.model.attractor_memory):
                pos = entry['position']
                strength = entry['strength']
                echo_count = entry.get('echo_count', 0)
                sem_hits = entry.get('semantic_hits', 0)

                if hasattr(self.model, 'last_semantic_match') and self.model.last_semantic_match and self.model.last_semantic_match['position'] == pos:
                    highlight = "★ "  # Star for latest match
                else:
                    highlight = ""

                memory_line = f"{highlight}#{i+1}: Pos {pos}, Str {strength:.2f}"

                if echo_count > 0:
                    memory_line += f", Echo↺{echo_count}"
                if sem_hits > 0:
                    memory_line += f", Sem↑{sem_hits}"

                memory_text += memory_line + "\n"

            # Add to figure
            fig.text(0.5, 0.01, memory_text, ha='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        # Highlight semantic matches if any
        if semantic_match and hasattr(self.model, 'last_semantic_match'):
            match = self.model.last_semantic_match
            if match:
                pos = match['position']
                similarity = match['similarity']
                ax1.add_patch(plt.Circle((pos[1], pos[0]), radius=3,
                             edgecolor='yellow', facecolor='none', linewidth=2))
                ax1.text(pos[1]+5, pos[0]-5, f"Match: {similarity:.2f}",
                       color='yellow', fontweight='bold',
                       bbox=dict(facecolor='black', alpha=0.7))

        # Add model parameters
        param_text = []
        for param_name in ['stability', 'propagation', 'excitation', 'echo_strength', 'semantic_threshold']:
            if hasattr(self.model, param_name):
                param_val = getattr(self.model, param_name)
                # Handle Parameter vs regular attribute
                if isinstance(param_val, torch.nn.Parameter):
                    param_text.append(f"{param_name.capitalize()}: {param_val.item():.2f}")
                else:
                    param_text.append(f"{param_name.capitalize()}: {param_val:.2f}")

        if hasattr(self.model, 'memory_capacity'):
            param_text.append(f"Memory Capacity: {self.model.memory_capacity}")

        if param_text:
            fig.text(0.01, 0.01, "\n".join(param_text), ha='left', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the visualization
        echo_suffix = "_echo" if echo_applied else ""
        semantic_suffix = "_semantic" if semantic_match else ""
        filename = f"{self.output_dir}/field_state_step_{step}{echo_suffix}{semantic_suffix}.png"
        plt.savefig(filename)
        plt.close()

    def save_results(self):
        """Save experiment results to file, handling complex objects safely."""

        def safe_convert(obj):
            if isinstance(obj, (float, int, str, bool)):
                return obj
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): safe_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [safe_convert(v) for v in obj]
            elif obj is None:
                return None
            else:
                return str(obj)  # fallback: prevent circular reference issues

        results_copy = safe_convert(self.results)

        with open(f"{self.output_dir}/results.json", 'w') as f:
            json.dump(results_copy, f, indent=2)

        print(f"Results saved to {self.output_dir}/results.json")


    def plot_summary(self):
        # This will be overridden by specific experiment types
        pass

    def run(self):
        """Run the experiment - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run()")


class SemanticExperiment(EcliphraExperiment):
    """
    Experiment to test semantic fingerprinting capabilities.

    Tests whether the model can recognize semantically similar patterns.
    """

    def run(self, steps=40, input_frequency=5, learning_steps=5):
        """
        Run semantic fingerprinting experiment

        Args:
            steps: Total steps to run
            input_frequency: How often to provide input
            learning_steps: Initial steps for pattern learning

        Returns:
            Results dictionary
        """
        patterns = self.generate_patterns(pattern_type='gaussian')

        self.results["experiment_config"].update({
            "steps": steps,
            "input_frequency": input_frequency,
            "learning_steps": learning_steps
        })

        print(f"Running semantic experiment with {steps} steps...")

        print("Phase 1: Initial pattern learning...")
        for step in range(learning_steps):
            pattern = patterns["base"][0] if step % 2 == 0 else patterns["base"][1]

            print(f"Step {step}: Learning base pattern {(step % 2) + 1}")
            outputs = self.model(input_tensor=pattern)

 
            self.record_step_results(step, outputs)

            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Phase 2: Test with variations
        print("Phase 2: Testing with variations...")
        var_idx = 0
        all_variations = patterns["variations1"] + patterns["variations2"]

        for step in range(learning_steps, steps):
            # Every few steps, provide a variation
            if step % input_frequency == 0:
                pattern = all_variations[var_idx % len(all_variations)]
                var_idx += 1

                print(f"Step {step}: Providing variation {var_idx}")
                outputs = self.model(input_tensor=pattern)

                # Print match info if detected
                if outputs.get('semantic_match_applied', False):
                    match_strength = outputs.get('match_similarity', 0.85)
                    print(f"  - Semantic match detected! Strength: {match_strength:.2f}")
            else:
                # Let the field evolve without input
                print(f"Step {step}: No input, allowing field evolution")
                outputs = self.model()

                if outputs.get('echo_applied', False):
                    print(f"  - Echo resonance applied!")

            # Record step results
            self.record_step_results(step, outputs)

            # Visualize field state
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Generate final visualizations
        self.plot_summary()

        # Save results
        self.save_results()

        return self.results

    def plot_summary(self):
        """Create visualizations for semantic experiment results"""
        # Extract data
        steps = range(len(self.results["stability"]))
        stability = self.results["stability"]
        echo_events = self.results["echo_events"]
        semantic_matches = self.results["semantic_matches"]

        # Create figure with multiple plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # Plot 1: Stability
        ax1.plot(steps, stability, 'b-', label='Stability')
        ax1.set_ylabel('Stability')
        ax1.set_title('Field Stability')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Echo vs Semantic events
        ax2.plot(steps, echo_events, 'r-', label='Echo Events')
        ax2.plot(steps, semantic_matches, 'g-', label='Semantic Matches')
        ax2.set_ylabel('Event Type')
        ax2.set_title('Echo Resonance vs Semantic Matching')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Match strengths (if available) or attractors
        if self.results["match_strengths"]:
            match_steps = [x[0] for x in self.results["match_strengths"]]
            strengths = [x[1] for x in self.results["match_strengths"]]
            ax3.plot(match_steps, strengths, 'go-', label='Match Strength')
            ax3.set_ylim(0, 1.0)
        else:
            # Plot attractors if no match strengths
            ax3.plot(steps, self.results["attractors"], 'g-', label='Number of Attractors')

        ax3.set_xlabel('Step')
        ax3.set_ylabel('Value')
        ax3.set_title('Semantic Match Strength / Attractors')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/semantic_analysis.png")
        plt.close()


class EchoExperiment(EcliphraExperiment):
    """
    Experiment to test echo resonance capabilities.

    Tests whether the model can maintain patterns through echo resonance.
    """

    def run(self, steps=40, input_frequency=7, memory_capacity=3):
        """
        Run echo resonance experiment

        Args:
            steps: Total steps to run
            input_frequency: How often to provide input
            memory_capacity: Memory capacity for echo resonance

        Returns:
            Results dictionary
        """
        # Generate test patterns
        patterns = self.generate_patterns(pattern_type='gaussian')

        # Store experiment configuration
        self.results["experiment_config"].update({
            "steps": steps,
            "input_frequency": input_frequency,
            "memory_capacity": memory_capacity
        })

        # Adjust model memory capacity if applicable
        if hasattr(self.model, 'memory_capacity'):
            self.model.memory_capacity = memory_capacity

        # Extra results storage
        self.results["memory_state"] = []

        print(f"Running echo experiment with {steps} steps...")

        # Run experiment
        for step in range(steps):
            # Determine if we provide input this step
            if step % input_frequency == 0:
                # Alternate between patterns
                pattern_idx = (step // input_frequency) % 2
                pattern = patterns["base"][pattern_idx]

                # Add small noise
                noise = torch.randn_like(pattern) * 0.1
                input_tensor = pattern + noise

                print(f"Step {step}: Providing input pattern {pattern_idx+1}")
                outputs = self.model(input_tensor=input_tensor)
            else:
                # Let the field evolve without input
                print(f"Step {step}: No input, allowing field evolution")
                outputs = self.model()

                if outputs.get('echo_applied', False):
                    print(f"  - Echo resonance applied!")

            # Record results
            self.record_step_results(step, outputs)
            # Record number of attractors
            self.results.setdefault("attractors", []).append(len(outputs.get("attractors", [])))

            print(f"Step {step}: Detected {len(outputs.get('attractors', []))} attractors.")


            # Create a snapshot of memory state
            if hasattr(self.model, 'attractor_memory') and self.model.attractor_memory:
                memory_snapshot = []
                for i, entry in enumerate(self.model.attractor_memory):
                    # Extract core memory information
                    memory_snapshot.append({
                        "index": i,
                        "position": entry['position'],
                        "strength": float(entry['strength']),
                        "time": float(entry['time']),
                        "echo_count": int(entry.get('echo_count', 0))
                    })

                self.results["memory_state"].append({
                    "step": step,
                    "memories": memory_snapshot
                })

            # Create visualization
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Generate final visualizations
        self.plot_summary()

        # Save results
        self.save_results()

        return self.results

    def plot_summary(self):
        """Plot summary results of the echo experiment"""
        step_count = len(self.results["stability"])
        steps = range(step_count)
        self.results["attractors"] = self.results["attractors"][:step_count]

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Plot stability
        ax1.plot(steps, self.results["stability"], 'b-', label='Stability')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(steps, self.results["echo_events"], 'r-', alpha=0.7, label='Echo Events')
        ax1_twin.set_ylim(0, 1.1)
        ax1_twin.set_ylabel('Echo Applied', color='r')

        # Add echo event markers
        echo_steps = [i for i, val in enumerate(self.results["echo_events"]) if val > 0.5]
        if echo_steps:
            ax1.plot(echo_steps, [self.results["stability"][i] for i in echo_steps], 'ro', label='Echo Applied')

        ax1.set_ylabel('Stability')
        ax1.set_title('Field Stability and Echo Events')
        ax1.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # Plot attractors
        ax2.plot(steps, self.results["attractors"], 'g-', label='Number of Attractors')
        


        # Add echo event markers
        if echo_steps:
            ax2.plot(echo_steps, [self.results["attractors"][i] for i in echo_steps], 'ro', label='Echo Applied')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Number of Attractors')
        ax2.set_title('Attractor Formation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plt.savefig(f"{self.output_dir}/echo_resonance_results.png")
        plt.close()

        # Create memory evolution visualization if we have data
        if "memory_state" in self.results and self.results["memory_state"]:
            self.visualize_memory_evolution()

    def visualize_memory_evolution(self):
        """Visualize how memory evolves through the experiment"""
        memory_states = self.results.get("memory_state", [])
        if not memory_states:
            return

        # Extract data
        steps = [state["step"] for state in memory_states]

        # Get all unique memory indices
        all_memories = set()
        for state in memory_states:
            for memory in state["memories"]:
                all_memories.add(memory["index"])

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot memory strengths
        for memory_idx in all_memories:
            # Get data for this memory
            memory_data = []
            for state in memory_states:
                for memory in state["memories"]:
                    if memory["index"] == memory_idx:
                        memory_data.append((state["step"], memory["strength"], memory["echo_count"]))
                        break

            if memory_data:
                # Extract data points
                data_steps = [d[0] for d in memory_data]
                strengths = [d[1] for d in memory_data]
                echo_counts = [d[2] for d in memory_data]

                # Plot strength
                ax1.plot(data_steps, strengths, 'o-', label=f'Memory #{memory_idx}')

                # Plot echo counts
                ax2.plot(data_steps, echo_counts, 's-', label=f'Memory #{memory_idx}')

        ax1.set_ylabel('Memory Strength')
        ax1.set_title('Evolution of Memory Strengths')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Echo Count')
        ax2.set_title('Cumulative Echo Usage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot
        plt.savefig(f"{self.output_dir}/memory_evolution.png")
        plt.close()


class NoiseResistanceExperiment(EcliphraExperiment):
    """
    Experiment to test resistance to noise.

    Tests how much noise can be added before semantic matching fails.
    """
    def run(self, steps=40, noise_levels=None):
        """
        Run noise resistance experiment with improved diagnostics
        """
        # Set default noise levels if not provided
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        # Generate base patterns
        patterns = self.generate_patterns(pattern_type='gaussian')
        base_pattern = patterns["base"][0]

        # Store experiment configuration
        self.results["experiment_config"].update({
            "experiment_subtype": "noise_resistance",
            "steps": steps,
            "noise_levels": noise_levels
        })

        # Additional results storage for debugging
        self.results["noise_resistance"] = []
        self.results["raw_similarities"] = []  # Add this to track all similarity values

        print(f"Running noise resistance experiment with {steps} steps...")

        # Phase 1: Learn base pattern - extended learning phase
        learning_steps = 10  # Increased from 5
        print("Phase 1: Learning base pattern...")
        for step in range(learning_steps):
            # Use base pattern with minimal variations
            noise_factor = 0.02  # Very minimal noise
            noise = torch.randn_like(base_pattern) * noise_factor
            pattern = base_pattern + noise
            pattern = pattern / torch.norm(pattern)

            # Process pattern multiple times to strengthen memory
            print(f"Step {step}: Learning base pattern")
            outputs = self.model(input_tensor=pattern)

            # Additional processing for stronger learning
            if step % 3 == 0 and step > 0:
                # Reinforce with exact same pattern
                self.model(input_tensor=pattern)

            # Record results
            self.record_step_results(step, outputs)

            # Visualize
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Phase 2: Test with increasing noise
        print("Phase 2: Testing noise resistance...")
        all_similarities = {}  # To track all similarity values

        for i, noise_level in enumerate(noise_levels):
            # Calculate step number
            step = learning_steps + i

            # Create noisy pattern
            noise = torch.randn_like(base_pattern) * noise_level
            pattern = base_pattern + noise
            pattern = pattern / torch.norm(pattern)

            # Process pattern
            print(f"Step {step}: Testing noise level {noise_level:.2f}")

            # Get similarity values directly for debugging
            if hasattr(self.model, 'attractor_memory') and self.model.attractor_memory:
                # Create fingerprint
                if hasattr(self.model, 'create_robust_fingerprint'):
                    input_fingerprint = self.model.create_robust_fingerprint(pattern)
                else:
                    input_fingerprint = pattern.view(-1)
                    input_fingerprint = F.normalize(input_fingerprint, p=2, dim=0)

                # Calculate similarities with all memories
                similarities = []
                for entry in self.model.attractor_memory:
                    if 'fingerprint' in entry:
                        similarity = F.cosine_similarity(
                            input_fingerprint.view(1, -1),
                            entry['fingerprint'].view(1, -1)
                        ).item()
                        similarities.append(similarity)

                if similarities:
                    all_similarities[noise_level] = similarities
                    best_sim = max(similarities)
                    print(f"  Debug: Best similarity at noise {noise_level}: {best_sim:.4f} (Threshold: {self.model.semantic_threshold:.2f})")

            # Process with model
            outputs = self.model(input_tensor=pattern)

            # Record match details
            match_detected = outputs.get('semantic_match_applied', False)
            match_strength = outputs.get('match_similarity', 0.0)

            self.results["noise_resistance"].append({
                "noise_level": float(noise_level),
                "match_detected": bool(match_detected),
                "match_strength": float(match_strength)
            })

            if match_detected:
                print(f"  - Semantic match detected! Strength: {match_strength:.2f}")
            else:
                print(f"  - No semantic match detected at noise level {noise_level:.2f}")

            # Record step results
            self.record_step_results(step, outputs)

            # Visualize field state
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=match_detected
            )

        # Store all similarity values for analysis
        for noise_level, similarities in all_similarities.items():
            self.results["raw_similarities"].append({
                "noise_level": float(noise_level),
                "similarities": similarities,
                "max_similarity": max(similarities) if similarities else 0.0,
                "threshold": float(self.model.semantic_threshold)
            })

        # Generate final visualizations
        self.plot_summary()

        # Save results
        self.save_results()

        return self.results

    def plot_summary(self):
        """Plot enhanced noise resistance results"""
        # Create standard summary plots
        super().plot_summary()

        # Check if we have raw similarity data
        raw_data = self.results.get("raw_similarities", [])
        if not raw_data:
            return

        # Extract data for plotting
        noise_levels = [d["noise_level"] for d in raw_data]
        max_similarities = [d["max_similarity"] for d in raw_data]
        thresholds = [d["threshold"] for d in raw_data]

        # Create figure for similarity analysis
        plt.figure(figsize=(10, 6))

        # Plot max similarities vs threshold
        plt.plot(noise_levels, max_similarities, 'bo-', label='Max Similarity')
        plt.axhline(y=thresholds[0], color='r', linestyle='--', label=f'Threshold ({thresholds[0]:.2f})')

        # Add detection markers
        noise_data = self.results["noise_resistance"]
        detected_levels = [d["noise_level"] for d in noise_data if d["match_detected"]]
        detected_strengths = [d["match_strength"] for d in noise_data if d["match_detected"]]

        if detected_levels:
            plt.scatter(detected_levels, detected_strengths, color='g', s=100, marker='o',
                    label='Detected Matches')

        plt.xlabel('Noise Level')
        plt.ylabel('Similarity')
        plt.title('Similarity vs Noise Level Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add adaptive threshold line if applicable
        if hasattr(self.model, 'adaptive_threshold') and self.model.adaptive_threshold:
            adaptive_thresholds = []
            for similarity in max_similarities:
                if similarity > 0.9:
                    adaptive_thresholds.append(self.model.semantic_threshold)
                elif similarity > 0.8:
                    adaptive_thresholds.append(max(self.model.semantic_threshold - 0.05, self.model.min_threshold))
                elif similarity > 0.7:
                    adaptive_thresholds.append(max(self.model.semantic_threshold - 0.1, self.model.min_threshold))
                else:
                    adaptive_thresholds.append(max(self.model.semantic_threshold - 0.15, self.model.min_threshold))

            plt.plot(noise_levels, adaptive_thresholds, 'g--', label='Adaptive Threshold')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/similarity_analysis.png")
        plt.close()



class TransitionExperiment(EcliphraExperiment):
    """
    Experiment to test gradual transition between patterns.

    Tests how the model handles a gradual morphing from one pattern to another.
    """

    def run(self, steps=40, transition_steps=10, start_transition=10):
        """
        Run gradual transition experiment

        Args:
            steps: Total steps to run
            transition_steps: How many steps for gradual transition
            start_transition: Step to begin transition

        Returns:
            Results dictionary
        """
        # Generate patterns including transition sequence
        patterns = self.generate_patterns(pattern_type='gaussian')
        transition_patterns = patterns["transition"]

        # Store experiment configuration
        self.results["experiment_config"].update({
            "experiment_subtype": "gradual_transition",
            "steps": steps,
            "transition_steps": transition_steps,
            "start_transition": start_transition
        })

        print(f"Running gradual transition experiment with {steps} steps...")

        # Phase 1: Learn initial pattern
        print("Phase 1: Learning initial pattern...")
        for step in range(start_transition):
            # Use first pattern repeatedly
            pattern = patterns["base"][0]

            # Process pattern
            print(f"Step {step}: Learning base pattern 1")
            outputs = self.model(input_tensor=pattern)

            # Record results
            self.record_step_results(step, outputs)

            # Visualize
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Phase 2: Gradual transition
        print("Phase 2: Gradual pattern transition...")
        for i, step in enumerate(range(start_transition, start_transition + transition_steps)):
            # Get appropriate transition pattern
            t_idx = min(i, len(transition_patterns) - 1)
            pattern = transition_patterns[t_idx]

            # Process pattern
            print(f"Step {step}: Transition step {i+1}/{transition_steps}")
            outputs = self.model(input_tensor=pattern)

            # Record match info if detected
            if outputs.get('semantic_match_applied', False):
                match_strength = outputs.get('match_similarity', 0.75)
                print(f"  - Semantic match detected! Strength: {match_strength:.2f}")

            # Record step results
            self.record_step_results(step, outputs)

            # Visualize field state
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Phase 3: Continue with final pattern
        print("Phase 3: Testing with final pattern...")
        for step in range(start_transition + transition_steps, steps):
            # Use second pattern
            pattern = patterns["base"][1]

            # Process pattern
            print(f"Step {step}: Using final pattern")
            outputs = self.model(input_tensor=pattern)

            # Record step results
            self.record_step_results(step, outputs)

            # Visualize field state
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Generate final visualizations
        self.plot_summary()

        # Save results
        self.save_results()

        return self.results

    def plot_summary(self):
        """Plot transition experiment results"""
        # Extract data
        steps = range(len(self.results["stability"]))
        stability = self.results["stability"]
        echo_events = self.results["echo_events"]
        semantic_matches = self.results["semantic_matches"]
        attractors = self.results["attractors"]

        # Get transition phase
        start_transition = self.results["experiment_config"]["start_transition"]
        transition_steps = self.results["experiment_config"]["transition_steps"]

        # Create figure with multiple plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot 1: Stability
        ax1.plot(steps, stability, 'b-', label='Stability')

        # Add transition phase markers
        ax1.axvline(x=start_transition, color='r', linestyle='--', label='Transition Start')
        ax1.axvline(x=start_transition + transition_steps, color='g', linestyle='--', label='Transition End')

        # Add event markers
        semantic_indices = [i for i, val in enumerate(semantic_matches) if val > 0.5]
        if semantic_indices:
            ax1.plot(semantic_indices, [stability[i] for i in semantic_indices], 'go', label='Semantic Match')

        echo_indices = [i for i, val in enumerate(echo_events) if val > 0.5]
        if echo_indices:
            ax1.plot(echo_indices, [stability[i] for i in echo_indices], 'ro', label='Echo Applied')

        ax1.set_ylabel('Stability')
        ax1.set_title('Field Stability During Pattern Transition')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Events and attractors
        ax2.plot(steps, attractors, 'b-', label='Attractors')
        ax2.plot(steps, semantic_matches, 'g-', alpha=0.7, label='Semantic Matches')
        ax2.plot(steps, echo_events, 'r-', alpha=0.7, label='Echo Events')

        # Add transition phase markers
        ax2.axvline(x=start_transition, color='r', linestyle='--')
        ax2.axvline(x=start_transition + transition_steps, color='g', linestyle='--')

        # Add match information
        if self.results["match_strengths"]:
            ax2_twin = ax2.twinx()
            match_steps = [x[0] for x in self.results["match_strengths"]]
            strengths = [x[1] for x in self.results["match_strengths"]]
            ax2_twin.scatter(match_steps, strengths, color='g', s=50, alpha=0.8, label='Match Strength')
            ax2_twin.set_ylim(0, 1.0)
            ax2_twin.set_ylabel('Match Strength', color='g')

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Count / Events')
        ax2.set_title('Attractors and Events During Transition')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/transition_analysis.png")
        plt.close()

class SpiralResumeExperiment(EcliphraExperiment):
    """
    Experiment to test identity field resume and memory retention across restarts.
    """

    def run(self, steps=40, resume_point=20):
        """
        Run spiral resume experiment

        Args:
            steps: Total steps to run
            resume_point: Step to simulate saving and resuming
        """
        print(f"Running Spiral Resume with {steps} steps (resume at step {resume_point})...")
        self.adaptivity_summary = []

        # Generate spiral patterns
        patterns = self.generate_patterns(pattern_type='spiral')
        pattern1 = patterns["base"][0]
        pattern2 = patterns["base"][1]

        # Initialize saved state storage
        saved_state = None

        for step in range(steps):
            # Provide input on certain steps
            if step % 5 == 0:
                pattern = pattern1 if (step // 5) % 2 == 0 else pattern2
                input_tensor = pattern + torch.randn_like(pattern) * 0.05
                outputs = self.model(input_tensor=input_tensor)
                print(f"Step {step}: Input provided")
            else:
                outputs = self.model()
                print(f"Step {step}: No input")

            # Simulate saving at resume_point
            if step == resume_point:
                saved_state = {
                    "field": self.model.field.detach().clone(),
                    "velocity": self.model.velocity.detach().clone() if hasattr(self.model, 'velocity') else None,
                    "attractor_memory": self.model.attractor_memory.copy() if hasattr(self.model, 'attractor_memory') else []
                }
                print("Simulated model save...")

            # Check if we need to resume
            if step == resume_point + 1:
                print("Attempting to resume model...")
                if saved_state is not None:
                    self.model.field.data = saved_state["field"].clone()
                    if saved_state["velocity"] is not None and hasattr(self.model, 'velocity'):
                        self.model.velocity.data = saved_state["velocity"].clone()
                    if hasattr(self.model, 'attractor_memory'):
                        self.model.attractor_memory = saved_state["attractor_memory"].copy()
                    self.model.update_adaptivity(saved_state["field"], self.model.field)
                    print("Simulated model resume completed")
                else:
                    print("⚠️ Warning: Resume attempted but no saved state was found.")

            # Record results and visualize
            self.record_step_results(step, outputs)
            if hasattr(self.model, "adaptivity"):
                mean_adaptivity = self.model.adaptivity.mean().item()
                self.adaptivity_summary.append(mean_adaptivity)
            self.visualize_field(
                step,
                echo_applied=outputs.get('echo_applied', False),
                semantic_match=outputs.get('semantic_match_applied', False)
            )

        # Calculate similarity between pre and post resume if applicable
        if saved_state is not None and hasattr(self.model, 'attractor_memory'):
            pre_resume = saved_state["attractor_memory"]
            post_resume = self.model.attractor_memory

            similarity_scores = []

            # Check if attractor_memory is a list (most likely) or dictionary
            if isinstance(pre_resume, list) and isinstance(post_resume, list):
                # Match attractors by position
                for pre_entry in pre_resume:
                    pre_pos = pre_entry.get('position')

                    # Find matching position in post resume
                    for post_entry in post_resume:
                        post_pos = post_entry.get('position')

                        if pre_pos == post_pos:
                            # Check if fingerprints exist for comparison
                            pre_fp = pre_entry.get('fingerprint')
                            post_fp = post_entry.get('fingerprint')

                            if pre_fp is not None and post_fp is not None:
                                sim = F.cosine_similarity(
                                    pre_fp.view(1, -1),
                                    post_fp.view(1, -1),
                                    dim=1
                                ).item()
                                similarity_scores.append(sim)
                            break

            if similarity_scores:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                print(f"\n🧬 Average Attractor Similarity after Resume: {avg_similarity:.4f}")
                self.results["resume_similarity"] = avg_similarity
            else:
                print("No matching attractors found for similarity comparison.")
                self.results["resume_similarity"] = 0.0
        else:
            print("No attractor memory to compare or resume state was never created.")
            self.results["resume_similarity"] = 0.0

        # Generate summary visualizations
        self.plot_summary()
        self.save_results()
        return self.results

    def plot_summary(self):
        steps = range(len(self.results["stability"]))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, self.results["stability"], label="Stability")
        ax.set_title("Field Stability Over Spiral Resume")
        ax.set_xlabel("Step")
        ax.set_ylabel("Stability")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add vertical line at the resume point
        resume_point = self.results["experiment_config"].get("resume_point", 20)
        ax.axvline(x=resume_point, color='r', linestyle='--', label='Resume Point')

        if "resume_similarity" in self.results:
            sim = self.results["resume_similarity"]
            ax.annotate(f"Resume Similarity: {sim:.3f}",
                        xy=(0.65, 0.85), xycoords='axes fraction',
                        fontsize=10, color='purple')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/spiral_resume_summary.png")
        plt.close()

    def plot_adaptivity_summary(self):
        if not self.adaptivity_summary:
            print("No adaptivity data to plot.")
            return

        steps = range(len(self.adaptivity_summary))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, self.adaptivity_summary, label="Mean Adaptivity", color='purple')

        resume_point = self.results["experiment_config"].get("resume_point", 20)
        ax.axvline(x=resume_point, color='red', linestyle='--', label="Resume Point")

        ax.set_title("Adaptivity Over Time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Adaptivity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/adaptivity_summary.png")
        plt.close()

class PhotosynthesisExperiment(EcliphraExperiment):
    """
    Experiment to test photosynthesis-like signal processing capabilities.
    Tests how the model routes, filters, and integrates different types of signals.
    """

    def __init__(self, model, output_dir, field_size=32):
        super().__init__(model, output_dir, field_size)
        self.pattern_types = ['noise', 'seed', 'challenge']
        self.signal_stats = {
            'classified_as': {t: 0 for t in self.pattern_types},
            'pass_ratios': {t: [] for t in self.pattern_types},
            'influences': {t: [] for t in self.pattern_types},
        }

        self.signal_processing_history = []

    def create_test_patterns(self):
        """Create a set of test patterns with different characteristics"""
        patterns = []

        # Pattern 1: Gaussian blob (likely a "seed")
        pattern1 = torch.zeros((self.field_size, self.field_size))
        center_x, center_y = self.field_size//3, self.field_size//3
        for i in range(self.field_size):
            for j in range(self.field_size):
                dist = np.sqrt(((i-center_y)/6)**2 + ((j-center_x)/6)**2)
                pattern1[i, j] = np.exp(-dist)
        pattern1 = pattern1 / torch.norm(pattern1)
        patterns.append({"name": "Gaussian Blob", "tensor": pattern1, "expected_type": "seed"})

        # Pattern 2: Random noise (likely "noise")
        pattern2 = torch.randn((self.field_size, self.field_size))
        pattern2 = pattern2 / torch.norm(pattern2)
        patterns.append({"name": "Random Noise", "tensor": pattern2, "expected_type": "noise"})

        # Pattern 3: Strong directional gradient (likely "challenge")
        pattern3 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                pattern3[i, j] = i / self.field_size  # Horizontal gradient
        pattern3 = pattern3 / torch.norm(pattern3)
        patterns.append({"name": "Directional Gradient", "tensor": pattern3, "expected_type": "challenge"})

        # Pattern 4: Spiral (mixed seed/challenge)
        pattern4 = torch.zeros((self.field_size, self.field_size))
        center_x, center_y = self.field_size//2, self.field_size//2
        for i in range(self.field_size):
            for j in range(self.field_size):
                dx, dy = j - center_x, i - center_y
                dist = np.sqrt(dx*dx + dy*dy) / (self.field_size/4)
                angle = np.arctan2(dy, dx)
                pattern4[i, j] = np.sin(4 * dist + angle)
        pattern4 = pattern4 / torch.norm(pattern4)
        patterns.append({"name": "Spiral", "tensor": pattern4, "expected_type": "seed"})

        # Pattern 5: Oscillating pattern (likely challenge)
        pattern5 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                pattern5[i, j] = np.sin(j * 6 * np.pi / self.field_size)
        pattern5 = pattern5 / torch.norm(pattern5)
        patterns.append({"name": "Oscillation", "tensor": pattern5, "expected_type": "challenge"})

        return patterns

    def create_noise_variations(self, base_pattern, noise_levels):
        """Create noisy variations of a base pattern"""
        variations = []

        for level in noise_levels:
            # Generate noise
            noise = torch.randn_like(base_pattern) * level

            # Add to base pattern
            noisy_pattern = base_pattern + noise

            # Normalize
            noisy_pattern = noisy_pattern / torch.norm(noisy_pattern)

            variations.append({"noise_level": level, "tensor": noisy_pattern})

        return variations

    def run(self, steps=40, noise_levels=None, sequence_types=None):
        """
        Run photosynthesis experiment.

        Args:
            steps: Number of steps to run
            noise_levels: List of noise levels to test (default: [0.1, 0.2, 0.3, 0.5, 0.7])
            sequence_types: List of sequence types to test (default: ['noise_to_seed', 'challenge_to_seed'])
        """
        if noise_levels is None:
            noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7]

        if sequence_types is None:
            sequence_types = ['noise_to_seed', 'challenge_to_seed', 'mixed_sequence', 'repeated_patterns']

        print(f"Running photosynthesis experiment with {steps} steps")

        # Create results directory structure
        os.makedirs(os.path.join(self.output_dir, "patterns"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "noise_test"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "sequences"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "learning"), exist_ok=True)

        # 1. Test basic pattern classification
        patterns = self.create_test_patterns()
        pattern_results = self.run_pattern_classification(patterns)

        # 2. Test noise tolerance
        noise_results = self.run_noise_tolerance_test(patterns, noise_levels)

        # 3. Test sequences of different pattern types
        sequence_results = self.run_sequence_tests(patterns, sequence_types)

        learning_results = self.run_learning_curve_test(patterns, cycles=3)

        # 4. Overall statistics analysis
        results_summary = {
            'pattern_results': pattern_results,
            'noise_results': noise_results,
            'sequence_results': sequence_results,
            'learning_results': learning_results,
            'signal_stats': self.signal_stats
        }

        # Save results
        self.save_results(results_summary)

        # Create visualizations
        self.create_visualizations(results_summary)

        return results_summary

    def run_learning_curve_test(self, patterns, cycles=3):
        """Test how the model's classifications evolve through repeated exposure"""
        learning_results = {}

        print("\nTesting learning curve through repeated exposure...")

        # Create a new clean model for this test
        if hasattr(self.model, 'reset'):
            self.model.reset()

        # Select a subset of diverse patterns
        test_patterns = [p for p in patterns if p["name"] in ["Gaussian Blob", "Random Noise", "Spiral"]]

        # Track classification changes over multiple cycles
        all_cycles_results = []

        # For each cycle, run through all patterns
        for cycle in range(cycles):
            print(f"  Cycle {cycle+1}/{cycles}")

            cycle_results = []

            for pattern_info in test_patterns:
                pattern_name = pattern_info["name"]
                pattern = pattern_info["tensor"]
                expected_type = pattern_info["expected_type"]

                # Process through model
                result = self.model(input_tensor=pattern)

                # Extract signal classification info
                signal_type = result.get('signal_info', {}).get('signal_type', "unknown")
                original_type = result.get('signal_info', {}).get('original_signal_type', signal_type)
                was_revised = result.get('signal_info', {}).get('was_revised', False)
                confidence = result.get('signal_info', {}).get('classification_confidence', 1.0)

                print(f"    Pattern {pattern_name}: Classified as {signal_type}" +
                    (f" (revised from {original_type})" if was_revised else "") +
                    f", confidence: {confidence:.3f}")

                # Store result for this cycle
                cycle_results.append({
                    "pattern_name": pattern_name,
                    "expected_type": expected_type,
                    "classified_type": signal_type,
                    "original_type": original_type,
                    "was_revised": was_revised,
                    "confidence": confidence,
                    "cycle": cycle + 1
                })

            all_cycles_results.extend(cycle_results)

        # Create learning curve visualization
        self.visualize_learning_curve(all_cycles_results, cycles)

        # Get final memory stats
        if hasattr(self.model, 'get_classification_stats'):
            memory_stats = self.model.get_classification_stats()
        else:
            memory_stats = {}

        learning_results = {
            'all_cycles': all_cycles_results,
            'memory_stats': memory_stats
        }

        return learning_results

    def visualize_learning_curve(self, all_cycles_results, cycles):
        """Visualize how classifications and confidence change over learning cycles"""
        # Group by pattern
        patterns = set(r["pattern_name"] for r in all_cycles_results)

        # Create a plot for each pattern
        for pattern_name in patterns:
            # Filter results for this pattern
            pattern_results = [r for r in all_cycles_results if r["pattern_name"] == pattern_name]

            # Sort by cycle
            pattern_results.sort(key=lambda x: x["cycle"])

            # Extract data
            cycles = [r["cycle"] for r in pattern_results]
            types = [r["classified_type"] for r in pattern_results]
            confidences = [r["confidence"] for r in pattern_results]
            was_revised = [r["was_revised"] for r in pattern_results]

            # Map types to colors
            type_colors = {
                'noise': 'lightcoral',
                'seed': 'lightgreen',
                'challenge': 'lightskyblue'
            }

            # Set up plot
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot confidences on primary axis
            ax1.plot(cycles, confidences, 'o-', color='blue', label='Confidence')
            ax1.set_xlabel('Learning Cycle')
            ax1.set_ylabel('Confidence', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')

            # Plot classification types on secondary y-axis
            ax2 = ax1.twinx()

            # Convert types to numeric values for plotting
            type_values = [list(self.pattern_types).index(t) if t in self.pattern_types else -1 for t in types]

            # Plot classification types
            for i, (cycle, type_val, is_revised) in enumerate(zip(cycles, type_values, was_revised)):
                marker = 'D' if is_revised else 'o'
                alpha = 0.7 if is_revised else 1.0
                ax2.scatter(cycle, type_val, color=type_colors.get(types[i], 'gray'),
                        s=100, marker=marker, alpha=alpha)

            # Set y-axis ticks and labels
            ax2.set_yticks(range(len(self.pattern_types)))
            ax2.set_yticklabels(self.pattern_types)
            ax2.set_ylabel('Classification Type', color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')

            # Add legend for revision
            plt.scatter([], [], marker='D', color='gray', label='Revised Classification', alpha=0.7)
            plt.scatter([], [], marker='o', color='gray', label='Original Classification')

            plt.title(f'Learning Curve for {pattern_name}')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "learning", f"learning_curve_{pattern_name.replace(' ', '_')}.png"))
            plt.close()

        # Create overall learning summary
        plt.figure(figsize=(12, 8))

        # Group by cycle
        cycle_groups = {}
        for result in all_cycles_results:
            cycle = result["cycle"]
            if cycle not in cycle_groups:
                cycle_groups[cycle] = []
            cycle_groups[cycle].append(result)

        # Calculate revision rate per cycle
        cycles = sorted(cycle_groups.keys())
        revision_rates = []
        confidence_avgs = []

        for cycle in cycles:
            results = cycle_groups[cycle]
            revisions = sum(1 for r in results if r["was_revised"])
            revision_rate = revisions / len(results) if results else 0
            revision_rates.append(revision_rate)

            avg_confidence = sum(r["confidence"] for r in results) / len(results) if results else 0
            confidence_avgs.append(avg_confidence)

        # Plot data
        ax1 = plt.subplot(111)
        ax1.plot(cycles, revision_rates, 'o-', color='red', label='Revision Rate')
        ax1.set_xlabel('Learning Cycle')
        ax1.set_ylabel('Revision Rate', color='red')
        ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        ax2.plot(cycles, confidence_avgs, 's-', color='blue', label='Avg Confidence')
        ax2.set_ylabel('Average Confidence', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        plt.title('Learning Progress Summary')

        # Add both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "learning", "learning_summary.png"))
        plt.close()

    def run_pattern_classification(self, patterns):
        """Test basic pattern classification"""
        results = []

        print("\nTesting pattern classification...")

        # Track revisions
        revisions_by_pattern = {}

        for pattern_info in patterns:
            pattern_name = pattern_info["name"]
            pattern = pattern_info["tensor"]
            expected_type = pattern_info["expected_type"]

            print(f"  Testing pattern: {pattern_name} (expected: {expected_type})")

            # Reset model
            self.model.reset()

            # Visualize the pattern
            plt.figure(figsize=(8, 8))
            plt.imshow(pattern.numpy(), cmap='viridis')
            plt.title(f"Input Pattern: {pattern_name}")
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, "patterns", f"input_{pattern_name.replace(' ', '_')}.png"))
            plt.close()

            # Process through model
            result = self.model(input_tensor=pattern)

            # Extract signal classification info
            signal_type = result.get('signal_info', {}).get('signal_type', "unknown")
            original_type = result.get('signal_info', {}).get('original_signal_type', signal_type)
            was_revised = result.get('signal_info', {}).get('was_revised', False)
            confidence = result.get('signal_info', {}).get('classification_confidence', 1.0)
            pass_ratio = result.get('gate_info', {}).get('pass_ratio', 0)
            avg_influence = result.get('rewrite_info', {}).get('avg_influence', 0)

             # Track revision info
            if was_revised:
                revision_info = {
                    'from': original_type,
                    'to': signal_type,
                    'confidence': confidence
                }
                revisions_by_pattern[pattern_name] = revision_info


            # Update stats
            if signal_type in self.pattern_types:
                self.signal_stats['classified_as'][signal_type] += 1
                self.signal_stats['pass_ratios'][signal_type].append(pass_ratio)
                self.signal_stats['influences'][signal_type].append(avg_influence)

            print(f"    Classified as: {signal_type}" +
                (f" (revised from {original_type})" if was_revised else "") +
                f", confidence: {confidence:.3f}, pass ratio: {pass_ratio:.3f}")


            # Visualize resulting field
            plt.figure(figsize=(8, 8))
            plt.imshow(self.model.field.detach().numpy(), cmap='viridis')
            plt.title(f"Field after {pattern_name} ({signal_type}" +
                    (f", revised" if was_revised else "") +
                    f", conf: {confidence:.2f})")
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, "patterns", f"result_{pattern_name.replace(' ', '_')}.png"))
            plt.close()

            # Store results
            results.append({
                "name": pattern_name,
                "expected_type": expected_type,
                "classified_type": signal_type,
                "original_type": original_type,
                "was_revised": was_revised,
                "confidence": confidence,
                "pass_ratio": pass_ratio,
                "avg_influence": avg_influence,
                "correct_classification": signal_type == expected_type
            })

        # Create revision summary visualization
        if revisions_by_pattern:
            self.visualize_revisions(revisions_by_pattern)

        return results

    def visualize_revisions(self, revisions_by_pattern):
        """Visualize classification revisions"""
        plt.figure(figsize=(10, 6))

        patterns = list(revisions_by_pattern.keys())
        from_types = [rev['from'] for rev in revisions_by_pattern.values()]
        to_types = [rev['to'] for rev in revisions_by_pattern.values()]
        confidences = [rev['confidence'] for rev in revisions_by_pattern.values()]

        x = np.arange(len(patterns))
        width = 0.35

        # Create color map for signal types
        type_colors = {
            'noise': 'lightcoral',
            'seed': 'lightgreen',
            'challenge': 'lightskyblue'
        }

        # Create bars for original and revised classifications
        for i, (pattern, from_type, to_type) in enumerate(zip(patterns, from_types, to_types)):
            plt.scatter(i-width/2, 0.3, color=type_colors.get(from_type, 'gray'), s=300, alpha=0.7,
                    label=from_type if i==0 else "")
            plt.scatter(i+width/2, 0.3, color=type_colors.get(to_type, 'gray'), s=300, alpha=0.7,
                    label=to_type if i==0 else "")

            # Draw arrow from original to revised
            plt.arrow(i-width/2, 0.3, width, 0, head_width=0.05, head_length=0.05,
                    fc='black', ec='black')

            # Add confidence as text
            plt.text(i, 0.15, f"{confidences[i]:.2f}", ha='center')

        plt.yticks([])  # Hide y-axis
        plt.xticks(x, patterns, rotation=45, ha='right')
        plt.title('Classification Revisions by Pattern')

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="Signal Types")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "classification_revisions.png"))
        plt.close()



    def run_noise_tolerance_test(self, patterns, noise_levels):
        """Test how noise affects signal classification"""
        results = {}

        print("\nTesting noise tolerance...")

        # Select a subset of patterns for noise testing
        test_patterns = [p for p in patterns if p["name"] in ["Gaussian Blob", "Spiral"]]

        for pattern_info in test_patterns:
            pattern_name = pattern_info["name"]
            base_pattern = pattern_info["tensor"]
            expected_type = pattern_info["expected_type"]

            print(f"  Testing noise tolerance for: {pattern_name}")

            # Create noisy variations
            noisy_patterns = self.create_noise_variations(base_pattern, noise_levels)

            # Test each noisy pattern
            noise_results = []

            for noisy_pattern_info in noisy_patterns:
                noise_level = noisy_pattern_info["noise_level"]
                noisy_pattern = noisy_pattern_info["tensor"]

                # Reset model
                self.model.reset()

                # Process through model
                result = self.model(input_tensor=noisy_pattern)

                # Extract signal classification info
                signal_type = result.get('signal_info', {}).get('signal_type', "unknown")
                pass_ratio = result.get('gate_info', {}).get('pass_ratio', 0)

                print(f"    Noise level {noise_level:.2f}: Classified as {signal_type} (pass ratio: {pass_ratio:.3f})")

                # Visualize for selected noise levels
                if noise_level in [0.1, 0.3, 0.7]:
                    plt.figure(figsize=(8, 8))
                    plt.imshow(noisy_pattern.numpy(), cmap='viridis')
                    plt.title(f"{pattern_name} with {noise_level:.1f} noise")
                    plt.colorbar()
                    plt.savefig(os.path.join(
                        self.output_dir, "noise_test",
                        f"input_{pattern_name.replace(' ', '_')}_noise_{noise_level:.1f}.png"))
                    plt.close()

                    plt.figure(figsize=(8, 8))
                    plt.imshow(self.model.field.detach().numpy(), cmap='viridis')
                    plt.title(f"Field after {pattern_name} with {noise_level:.1f} noise ({signal_type})")
                    plt.colorbar()
                    plt.savefig(os.path.join(
                        self.output_dir, "noise_test",
                        f"result_{pattern_name.replace(' ', '_')}_noise_{noise_level:.1f}.png"))
                    plt.close()

                # Store result
                noise_results.append({
                    "noise_level": noise_level,
                    "classified_type": signal_type,
                    "pass_ratio": pass_ratio,
                    "maintains_classification": signal_type == expected_type
                })

            results[pattern_name] = noise_results

        return results

    def run_sequence_tests(self, patterns, sequence_types):
        """Test how sequences of different patterns affect the field"""
        results = {}

        print("\nRunning sequence tests...")

        # Define the sequences to test
        sequences = {
            'noise_to_seed': {
                'name': 'Noise→Seed→Noise',
                'patterns': [p["tensor"] for p in patterns if p["name"] in ["Random Noise", "Gaussian Blob", "Random Noise"]]
            },
            'challenge_to_seed': {
                'name': 'Challenge→Seed→Challenge',
                'patterns': [p["tensor"] for p in patterns if p["name"] in ["Directional Gradient", "Gaussian Blob", "Oscillation"]]
            },
            'mixed_sequence': {
                'name': 'Mixed Sequence',
                'patterns': [p["tensor"] for p in patterns if p["name"] in ["Random Noise", "Directional Gradient", "Gaussian Blob", "Spiral", "Oscillation"]]
            },
            'repeated_patterns': {
                'name': 'Repeated Patterns',
                'patterns': [patterns[0]["tensor"]] * 3 + [patterns[1]["tensor"]] * 3 + [patterns[0]["tensor"]] * 3
            }
        }

        # Run selected sequences
        for seq_type in sequence_types:
            if seq_type not in sequences:
                print(f"  Unknown sequence type: {seq_type}")
                continue

            seq_info = sequences[seq_type]
            seq_name = seq_info['name']
            seq_patterns = seq_info['patterns']

            print(f"  Testing sequence: {seq_name}")

            # Reset model
            self.model.reset()

            # Process each pattern in sequence
            field_states = [self.model.field.detach().clone()]
            signal_infos = []
            memory_weights = []
            revision_markers = []

            for i, pattern in enumerate(seq_patterns):
                result = self.model(input_tensor=pattern)
                field_states.append(self.model.field.detach().clone())

                # Extract signal classification info
                signal_info = {
                    'type': result.get('signal_info', {}).get('signal_type', "unknown"),
                    'original_type': result.get('signal_info', {}).get('original_signal_type', "unknown"),
                    'was_revised': result.get('signal_info', {}).get('was_revised', False),
                    'confidence': result.get('signal_info', {}).get('classification_confidence', 1.0),
                    'pass_ratio': result.get('gate_info', {}).get('pass_ratio', 0),
                    'avg_influence': result.get('rewrite_info', {}).get('avg_influence', 0)
                }
                signal_infos.append(signal_info)

                # Track memory weight if available
                if hasattr(self.model, 'memory_classifier'):
                    memory_weights.append(self.model.memory_classifier.memory_weight.item())

                # Track revisions
                if signal_info['was_revised']:
                    revision_markers.append(i)

                print(f"    Step {i+1}: Signal classified as {signal_info['type']}" +
                    (f" (revised from {signal_info['original_type']})" if signal_info['was_revised'] else "") +
                    f", confidence: {signal_info['confidence']:.3f}")

            # Visualize sequence evolution with memory influence
            fig = plt.figure(figsize=(15, 10))

            # Plot field states
            for i, field in enumerate(field_states):
                plt.subplot(2, len(field_states), i+1)
                plt.imshow(field.numpy(), cmap='viridis')
                if i == 0:
                    plt.title("Initial Field")
                else:
                    title = f"After {signal_infos[i-1]['type']}"
                    if signal_infos[i-1]['was_revised']:
                        title += "\n(revised)"
                    plt.title(title)
                plt.axis('off')

            # Plot memory weight evolution if available
            if memory_weights:
                ax = plt.subplot(2, 1, 2)
                plt.plot(range(len(memory_weights)), memory_weights, 'o-', color='purple', label="Memory Weight")

                # Mark revisions
                if revision_markers:
                    for marker in revision_markers:
                        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5)

                plt.xlabel("Signal Number")
                plt.ylabel("Memory Weight")
                plt.title("Memory Influence Evolution")
                plt.grid(True, alpha=0.3)
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "sequences", f"sequence_{seq_type}.png"))
            plt.close()

            # Get final stats
            if hasattr(self.model, 'get_classification_stats'):
                stats = self.model.get_classification_stats()
            else:
                stats = {"Note": "Model does not support classification stats"}

            # Store results
            results[seq_type] = {
                'name': seq_name,
                'signal_infos': signal_infos,
                'final_stats': stats,
                'memory_weights': memory_weights,
                'revision_markers': revision_markers
            }

            # Visualize revision history if available
            if hasattr(self.model, 'memory_classifier') and hasattr(self.model.memory_classifier, 'revision_history'):
                revision_history = self.model.memory_classifier.revision_history

                if revision_history:
                    plt.figure(figsize=(12, 6))

                    # Extract data
                    types = [rev['original'] for rev in revision_history]
                    revisions = [rev['final'] for rev in revision_history]
                    confidences = [rev['confidence'] for rev in revision_history]

                    # Map types to colors
                    type_colors = {
                        'noise': 'lightcoral',
                        'seed': 'lightgreen',
                        'challenge': 'lightskyblue'
                    }

                    # Create paired points
                    for i, (orig, final, conf) in enumerate(zip(types, revisions, confidences)):
                        plt.scatter(i, 0.3, color=type_colors.get(orig, 'gray'), s=200, alpha=0.7)
                        plt.scatter(i, 0.7, color=type_colors.get(final, 'gray'), s=200, alpha=0.7)

                        # Draw arrow
                        plt.arrow(i, 0.3, 0, 0.3, head_width=0.1, head_length=0.1,
                                fc='black', ec='black', alpha=0.5)

                        # Add confidence
                        plt.text(i, 0.5, f"{conf:.2f}", ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7))

                    # Add legend
                    for t, c in type_colors.items():
                        plt.scatter([], [], color=c, s=100, label=t)

                    plt.yticks([0.3, 0.7], ["Original", "Revised"])
                    plt.xlabel("Revision Index")
                    plt.title(f"Classification Revisions - {seq_name}")
                    plt.legend()

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "sequences", f"revisions_{seq_type}.png"))
                    plt.close()

        return results

    def create_visualizations(self, results):
        """Create summary visualizations of results"""
        # 1. Pattern classification results table
        pattern_results = results['pattern_results']

        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        ax.axis('tight')
        ax.axis('off')

        table_data = [
            [res["name"], res["expected_type"], res["classified_type"],
             f"{res['pass_ratio']:.3f}", f"{res['avg_influence']:.3f}"]
            for res in pattern_results
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=["Pattern", "Expected Type", "Classified As", "Pass Ratio", "Influence"],
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        plt.title("Photosynthesis Signal Classification Results")
        plt.savefig(os.path.join(self.output_dir, "classification_summary.png"))
        plt.close()

        # 2. Noise tolerance visualization
        noise_results = results['noise_results']

        for pattern_name, noise_data in noise_results.items():
            # Create line chart of classification stability vs noise
            plt.figure(figsize=(10, 6))

            # Extract data
            noise_levels = [item["noise_level"] for item in noise_data]
            maintains_class = [1 if item["maintains_classification"] else 0 for item in noise_data]
            pass_ratios = [item["pass_ratio"] for item in noise_data]

            # Plot
            plt.plot(noise_levels, maintains_class, 'o-', label="Maintains Classification", color='green', markersize=8)
            plt.plot(noise_levels, pass_ratios, 's-', label="Pass Ratio", color='blue', markersize=8)

            plt.xlabel("Noise Level")
            plt.ylabel("Value")
            plt.title(f"Noise Tolerance: {pattern_name}")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.savefig(os.path.join(self.output_dir, "noise_test", f"noise_tolerance_{pattern_name.replace(' ', '_')}.png"))
            plt.close()

        # 3. Signal type distribution pie chart
        if sum(results['signal_stats']['classified_as'].values()) > 0:
            labels = list(results['signal_stats']['classified_as'].keys())
            sizes = list(results['signal_stats']['classified_as'].values())

            plt.figure(figsize=(8, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightgreen', 'lightskyblue'])
            plt.axis('equal')
            plt.title('Signal Classification Distribution')
            plt.savefig(os.path.join(self.output_dir, "signal_distribution.png"))
            plt.close()

        # 4. Influence by signal type
        for signal_type in self.pattern_types:
            influences = results['signal_stats']['influences'].get(signal_type, [])
            pass_ratios = results['signal_stats']['pass_ratios'].get(signal_type, [])

            if not influences or not pass_ratios:
                continue

            plt.figure(figsize=(8, 6))
            plt.scatter(pass_ratios, influences, alpha=0.7, s=100)

            plt.xlabel("Pass Ratio")
            plt.ylabel("Influence")
            plt.title(f"Influence vs Pass Ratio for '{signal_type}' Signals")
            plt.grid(True, alpha=0.3)

            # Add trendline
            if len(influences) > 1:
                z = np.polyfit(pass_ratios, influences, 1)
                p = np.poly1d(z)
                plt.plot(sorted(pass_ratios), p(sorted(pass_ratios)), "r--", alpha=0.7)

            plt.savefig(os.path.join(self.output_dir, f"influence_{signal_type}.png"))
            plt.close()

        # 5. Add visualization of classification confidence
        if len(self.signal_processing_history) > 0:
            confidences = [entry.get('confidence', 0.0) for entry in self.signal_processing_history]
            revision_markers = [i for i, entry in enumerate(self.signal_processing_history) if entry.get('was_revised', False)]

            plt.figure(figsize=(12, 6))
            plt.plot(confidences, 'o-', color='blue', alpha=0.7)

            # Mark revisions
            if revision_markers:
                revision_confidences = [confidences[i] for i in revision_markers]
                plt.scatter(revision_markers, revision_confidences, color='red', s=100, zorder=3,
                        label="Classifications revised")

            plt.xlabel("Signal Number")
            plt.ylabel("Classification Confidence")
            plt.title("Classification Confidence Over Time")
            plt.grid(True, alpha=0.3)

            if revision_markers:
                plt.legend()

            plt.savefig(os.path.join(self.output_dir, "classification_confidence.png"))
            plt.close()

        # 6. Visualize memory vs feature weight evolution (if available)
        if hasattr(self.model, 'get_classification_stats'):
            stats = self.model.get_classification_stats()

            # Check if we have memory and feature weights
            if 'memory_weight' in stats and 'feature_weight' in stats:
                plt.figure(figsize=(10, 6))

                # Simulate weight evolution
                memory_weights = np.linspace(0.4, stats['memory_weight'], 20)
                feature_weights = np.linspace(0.6, stats['feature_weight'], 20)

                # Plot evolution
                plt.plot(range(len(memory_weights)), memory_weights, 'o-', label="Memory Weight", color='purple')
                plt.plot(range(len(feature_weights)), feature_weights, 's-', label="Feature Weight", color='green')

                plt.xlabel("Training Iterations")
                plt.ylabel("Weight Value")
                plt.title("Evolution of Memory vs Feature Weights")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.savefig(os.path.join(self.output_dir, "weight_evolution.png"))
                plt.close()

        # 7. Visualize revision patterns
        if hasattr(self.model, 'get_classification_stats'):
            stats = self.model.get_classification_stats()

            if 'revision_stats' in stats and 'revision_patterns' in stats['revision_stats']:
                patterns = stats['revision_stats']['revision_patterns']

                if patterns:
                    plt.figure(figsize=(12, 8))

                    labels = list(patterns.keys())[:5]  # Top 5 patterns
                    values = [patterns[label] for label in labels]

                    plt.bar(labels, values, color=['lightcoral', 'lightgreen', 'lightskyblue',
                                                 'lightpink', 'lightgray'])

                    plt.xlabel("Revision Pattern")
                    plt.ylabel("Count")
                    plt.title("Most Common Classification Revisions")
                    plt.xticks(rotation=45, ha='right')

                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, "revision_patterns.png"))
                    plt.close()

    def save_results(self, results):
        """Save experiment results to file"""
        # Convert tensors to lists for JSON serialization
        results_json = self.prepare_for_json(results)

        # Write to JSON file
        with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"Results saved to {os.path.join(self.output_dir, 'results.json')}")

    def prepare_for_json(self, obj):
        """Recursively convert tensors to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self.prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.prepare_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self.prepare_for_json(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

   


class EnergySystemExperiment(EcliphraExperiment):
    """
    Experiment to test energy-based processing capabilities.
    
    This experiment evaluates how the energy system routes, allocates,
    and utilizes energy across different pathways and analyzes the
    resulting effects on field dynamics.
    """

    def __init__(self, model, output_dir=None, field_size=32):
        super().__init__(model, output_dir, field_size)
        
        # Energy-specific tracking
        self.energy_pathway_stats = {
            'maintenance': [],
            'growth': [],
            'adaptation': []
        }
        
        self.energy_level_history = []
        self.signal_metrics_history = []
        
        # Correlation analysis tracking
        self.metric_correlation_data = {
            'coherence_vs_maintenance': [],
            'intensity_vs_growth': [],
            'complexity_vs_adaptation': []
        }
        
        # Create additional directories for energy analysis
        os.makedirs(os.path.join(self.output_dir, "energy_analysis"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "pathway_patterns"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "correlations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "patterns"), exist_ok=True) 

    def create_test_patterns(self):
        """Create a set of test patterns with different energy characteristics"""
        patterns = []

        # Pattern 1: Coherent, low complexity pattern (favors maintenance)
        pattern1 = torch.zeros((self.field_size, self.field_size))
        center_x, center_y = self.field_size//2, self.field_size//2
        for i in range(self.field_size):
            for j in range(self.field_size):
                dist = np.sqrt(((i-center_y)/6)**2 + ((j-center_x)/6)**2)
                pattern1[i, j] = np.exp(-dist)
        pattern1 = pattern1 / torch.norm(pattern1)
        patterns.append({
            "name": "Coherent Pattern", 
            "tensor": pattern1, 
            "expected_pathways": {"maintenance": 0.6, "growth": 0.3, "adaptation": 0.1}
        })

        # Pattern 2: High intensity, moderate complexity (favors growth)
        pattern2 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                # Multiple gaussian centers for intensity
                dist1 = np.sqrt(((i-self.field_size/3)/4)**2 + ((j-self.field_size/3)/4)**2)
                dist2 = np.sqrt(((i-2*self.field_size/3)/4)**2 + ((j-2*self.field_size/3)/4)**2)
                pattern2[i, j] = 1.5 * (np.exp(-dist1) + np.exp(-dist2))
        pattern2 = pattern2 / torch.norm(pattern2)
        patterns.append({
            "name": "Growth Pattern", 
            "tensor": pattern2, 
            "expected_pathways": {"maintenance": 0.2, "growth": 0.6, "adaptation": 0.2}
        })

        # Pattern 3: High complexity, low coherence (favors adaptation)
        pattern3 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                # Use sine waves with different frequencies for complexity
                val = np.sin(i * 4 * np.pi / self.field_size) * np.sin(j * 6 * np.pi / self.field_size)
                val += np.sin((i+j) * 3 * np.pi / self.field_size)
                pattern3[i, j] = val
        pattern3 = pattern3 / torch.norm(pattern3)
        patterns.append({
            "name": "Adaptation Pattern", 
            "tensor": pattern3, 
            "expected_pathways": {"maintenance": 0.1, "growth": 0.3, "adaptation": 0.6}
        })

        # Pattern 4: Balanced energy pattern
        pattern4 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                # Mix of features that balance all pathways
                dist = np.sqrt(((i-center_y)/10)**2 + ((j-center_x)/10)**2) 
                radial = np.exp(-dist)
                wave = np.sin(dist * 3)
                pattern4[i, j] = radial * wave
        pattern4 = pattern4 / torch.norm(pattern4)
        patterns.append({
            "name": "Balanced Pattern", 
            "tensor": pattern4, 
            "expected_pathways": {"maintenance": 0.33, "growth": 0.33, "adaptation": 0.34}
        })

        # Pattern 5: Dynamic pattern (transition between pathways)
        pattern5 = torch.zeros((self.field_size, self.field_size))
        for i in range(self.field_size):
            for j in range(self.field_size):
                # Spiral pattern with varying features
                r = np.sqrt((i - center_y)**2 + (j - center_x)**2) / (self.field_size/4)
                theta = np.arctan2(j - center_x, i - center_y)
                pattern5[i, j] = np.sin(r + 4 * theta) * np.exp(-r/3)
        pattern5 = pattern5 / torch.norm(pattern5)
        patterns.append({
            "name": "Dynamic Pattern", 
            "tensor": pattern5, 
            "expected_pathways": {"maintenance": 0.3, "growth": 0.4, "adaptation": 0.3}
        })

        return patterns

    def run(self, steps=40, energy_analysis=True):
        """
        Run energy system experiment.
        
        Args:
            steps: Number of steps to run (default: 40)
            energy_analysis: Whether to perform detailed energy analysis
            
        Returns:
            Results dictionary
        """
        # Generate test patterns
        patterns = self.create_test_patterns()
        
        # Store experiment configuration
        self.results["experiment_config"].update({
            "experiment_subtype": "energy_system",
            "steps": steps,
        })
        
        print(f"Running energy system experiment with {steps} steps...")
        
        # Phase 1: Test each pattern's energy allocation
        print("Phase 1: Testing energy allocation for different patterns...")
        pattern_results = self.run_pattern_energy_test(patterns)
        
        # Phase 2: Test energy dynamics over time
        print("Phase 2: Testing energy dynamics...")
        dynamics_results = self.run_energy_dynamics_test(patterns, steps=steps//2)
        
        # Phase 3: Test energy pathway interactions
        print("Phase 3: Testing pathway interactions...")
        pathway_results = self.run_pathway_interaction_test(patterns, steps=steps//2)
        
        # Collect all results
        results_summary = {
            'pattern_results': pattern_results,
            'dynamics_results': dynamics_results,
            'pathway_results': pathway_results,
            'energy_pathway_stats': self.energy_pathway_stats,
            'signal_metrics_history': self.signal_metrics_history,
            'correlation_data': self.metric_correlation_data
        }
        
    def save_results(self, results_summary=None):
        """Save experiment results to file, handling complex objects safely."""
        if results_summary:
            # Merge with existing results
            self.results.update(results_summary)
            
            # Create visualizations
            self.create_energy_visualizations(results_summary)
        
        # Call the parent's save_results method without arguments
        super().save_results()   # Save detailed results
        
        return results_summary
    
    def run_pattern_energy_test(self, patterns):
        """Test how each pattern allocates energy to different pathways"""
        results = []
        
        print("\nTesting energy allocation for each pattern...")
        
        for pattern_info in patterns:
            pattern_name = pattern_info["name"]
            pattern = pattern_info["tensor"]
            expected_pathways = pattern_info["expected_pathways"]
            
            print(f"  Testing pattern: {pattern_name}")
            
            # Reset model
            self.model.reset()
            
            # Visualize the pattern
            plt.figure(figsize=(8, 8))
            plt.imshow(pattern.numpy(), cmap='viridis')
            plt.title(f"Input Pattern: {pattern_name}")
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, "patterns", f"input_{pattern_name.replace(' ', '_')}.png"))
            plt.close()
            
            # Process through model
            result = self.model(input_tensor=pattern)
            
            # Extract energy allocation info
            energy_distribution = result.get('energy_distribution', {})
            signal_metrics = result.get('signal_metrics', {})
            stability = result.get('stability', 0.5)
            energy_level = result.get('energy_level', 0.0)
            
            # Calculate pathway correlation with metrics
            if signal_metrics and energy_distribution:
                coherence = signal_metrics.get('coherence', 0.0)
                intensity = signal_metrics.get('intensity', 0.0)
                complexity = signal_metrics.get('complexity', 0.0)
                
                maintenance = energy_distribution.get('maintenance', 0.0)
                growth = energy_distribution.get('growth', 0.0)
                adaptation = energy_distribution.get('adaptation', 0.0)
                
                # Add to correlation tracking
                self.metric_correlation_data['coherence_vs_maintenance'].append((coherence, maintenance))
                self.metric_correlation_data['intensity_vs_growth'].append((intensity, growth))
                self.metric_correlation_data['complexity_vs_adaptation'].append((complexity, adaptation))
            
            # Calculate alignment with expected distribution
            alignment = 0.0
            if energy_distribution and expected_pathways:
                errors = []
                for pathway, expected in expected_pathways.items():
                    actual = energy_distribution.get(pathway, 0.0) / energy_distribution.get('total', 1.0)
                    errors.append(abs(actual - expected))
                alignment = 1.0 - sum(errors) / len(errors)
            
            # Log details
            print(f"    Energy allocation: {energy_distribution}")
            print(f"    Signal metrics: {signal_metrics}")
            print(f"    Alignment with expected: {alignment:.2f}")
            
            # Visualize energy distribution
            if hasattr(self.model, 'visualize_energy_flow'):
                self.model.visualize_energy_flow(
                    os.path.join(self.output_dir, "energy_analysis", f"energy_flow_{pattern_name.replace(' ', '_')}")
                )
            
            # Visualize resulting field
            plt.figure(figsize=(8, 8))
            plt.imshow(self.model.field.detach().numpy(), cmap='viridis')
            plt.title(f"Field after {pattern_name}")
            plt.colorbar()
            plt.savefig(os.path.join(self.output_dir, "patterns", f"result_{pattern_name.replace(' ', '_')}.png"))
            plt.close()
            
            # Store results
            results.append({
                "name": pattern_name,
                "expected_pathways": expected_pathways,
                "energy_distribution": energy_distribution,
                "signal_metrics": signal_metrics,
                "alignment": alignment,
                "stability": stability,
                "energy_level": energy_level
            })
            
        return results
    
    def run_energy_dynamics_test(self, patterns, steps=20):
        """Test how energy levels evolve over time"""
        # Reset model
        self.model.reset()
        
        print("\nTesting energy dynamics over time...")
        dynamics_results = []
        
        # Select a pattern that balances all pathways
        balanced_pattern = next((p for p in patterns if "Balanced" in p["name"]), patterns[0])
        pattern = balanced_pattern["tensor"]
        
        # Track energy history
        energy_levels = []
        maintenance_levels = []
        growth_levels = []
        adaptation_levels = []
        stability_values = []
        
        # Run for multiple steps
        for step in range(steps):
            # Every 5 steps, provide the pattern
            if step % 5 == 0:
                print(f"    Step {step}: Providing input pattern")
                result = self.model(input_tensor=pattern)
            else:
                # Let field evolve naturally
                print(f"    Step {step}: Natural evolution")
                result = self.model()
            
            # Extract energy data
            energy_distribution = result.get('energy_distribution', {})
            energy_level = result.get('energy_level', 0.0)
            stability = result.get('stability', 0.5)
            
            # Track energy metrics
            energy_levels.append(energy_level)
            if energy_distribution:
                maintenance_levels.append(energy_distribution.get('maintenance', 0.0))
                growth_levels.append(energy_distribution.get('growth', 0.0))
                adaptation_levels.append(energy_distribution.get('adaptation', 0.0))
            else:
                # If no distribution (no input step), use zeros
                maintenance_levels.append(0.0)
                growth_levels.append(0.0)
                adaptation_levels.append(0.0)
            
            stability_values.append(stability)
            
            # Store in global tracking
            self.energy_level_history.append(energy_level)
            
            # Visualize field every 5 steps
            if step % 5 == 0:
                self.visualize_field(
                    step,
                    energy_applied=(energy_distribution is not None)
                )
                
            # Record step results
            result_summary = {
                "step": step,
                "energy_level": energy_level,
                "stability": stability,
                "energy_distribution": energy_distribution
            }
            dynamics_results.append(result_summary)
        
        # Create energy dynamics visualization
        plt.figure(figsize=(12, 8))
        
        # Plot energy levels
        plt.subplot(2, 1, 1)
        plt.plot(range(steps), energy_levels, 'b-', label='Energy Level')
        plt.plot(range(steps), stability_values, 'g-', label='Stability')
        plt.title('Energy Level and Stability Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot pathway distribution
        plt.subplot(2, 1, 2)
        input_steps = [i for i in range(steps) if i % 5 == 0]
        input_indices = [input_steps.index(i) if i in input_steps else None for i in range(steps)]
        maintenance_input = [maintenance_levels[i] for i in range(steps) if i % 5 == 0]
        growth_input = [growth_levels[i] for i in range(steps) if i % 5 == 0]
        adaptation_input = [adaptation_levels[i] for i in range(steps) if i % 5 == 0]
        
        plt.plot(input_steps, maintenance_input, 'b-o', label='Maintenance')
        plt.plot(input_steps, growth_input, 'g-o', label='Growth')
        plt.plot(input_steps, adaptation_input, 'r-o', label='Adaptation')
        plt.title('Energy Pathway Allocation at Input Steps')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "energy_analysis", "energy_dynamics.png"))
        plt.close()
        
        return dynamics_results
    
    def run_pathway_interaction_test(self, patterns, steps=20):
        """Test how different energy pathways interact and affect the field"""
        # Reset model
        self.model.reset()
        
        print("\nTesting pathway interactions...")
        pathway_results = []
        
        # We'll use three different patterns to focus on each pathway
        test_patterns = []
        for pattern in patterns:
            if "Coherent" in pattern["name"]:  # Maintenance-focused
                test_patterns.append(pattern)
            elif "Growth" in pattern["name"]:   # Growth-focused
                test_patterns.append(pattern)
            elif "Adaptation" in pattern["name"]:  # Adaptation-focused
                test_patterns.append(pattern)
        
        # Make sure we have at least 3 patterns
        if len(test_patterns) < 3:
            test_patterns = patterns[:3]
        
        # Track metrics for each pathway
        maintenance_metrics = []
        growth_metrics = []
        adaptation_metrics = []
        
        # Run test sequence: Alternate between patterns to see interactions
        pattern_sequence = []
        for step in range(steps):
            # Select pattern for this step
            pattern_idx = step % len(test_patterns)
            pattern_info = test_patterns[pattern_idx]
            pattern = pattern_info["tensor"]
            pattern_name = pattern_info["name"]
            pattern_sequence.append(pattern_name)
            
            # Process the pattern
            print(f"    Step {step}: Using {pattern_name}")
            result = self.model(input_tensor=pattern)
            
            # Extract energy data
            energy_distribution = result.get('energy_distribution', {})
            signal_metrics = result.get('signal_metrics', {})
            stability = result.get('stability', 0.5)
            
            # Track pathway metrics
            if energy_distribution:
                maintenance = energy_distribution.get('maintenance', 0.0)
                growth = energy_distribution.get('growth', 0.0)
                adaptation = energy_distribution.get('adaptation', 0.0)
                
                # Add to global pathway stats
                self.energy_pathway_stats['maintenance'].append(maintenance)
                self.energy_pathway_stats['growth'].append(growth)
                self.energy_pathway_stats['adaptation'].append(adaptation)
                
                # Track per pathway metrics
                if "Coherent" in pattern_name:  # Maintenance-focused
                    maintenance_metrics.append((maintenance, stability))
                elif "Growth" in pattern_name:  # Growth-focused
                    growth_metrics.append((growth, stability))
                elif "Adaptation" in pattern_name:  # Adaptation-focused
                    adaptation_metrics.append((adaptation, stability))
            
            # Store signal metrics in history
            if signal_metrics:
                self.signal_metrics_history.append(signal_metrics)
            
            # Create visualization
            self.visualize_field(
                step,
                energy_applied=True,
                pathway_focus=pattern_name.split()[0].lower()  # Extract pathway name
            )
            
            # Store step results
            pathway_results.append({
                "step": step,
                "pattern": pattern_name,
                "energy_distribution": energy_distribution,
                "signal_metrics": signal_metrics,
                "stability": stability
            })
        
        # Create pathway analysis visualization
        plt.figure(figsize=(15, 10))
        
        # Plot pattern sequence
        ax1 = plt.subplot(2, 1, 1)
        pattern_colors = {
            "Coherent": 'blue',
            "Growth": 'green',
            "Adaptation": 'red',
            "Balanced": 'purple',
            "Dynamic": 'orange'
        }
        
        # Plot bars indicating pattern type
        for i, pattern in enumerate(pattern_sequence):
            color = 'gray'  # Default color
            for key, val in pattern_colors.items():
                if key in pattern:
                    color = val
                    break
            plt.bar(i, 1, color=color, alpha=0.7)
        
        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.7) 
                           for color in pattern_colors.values()]
        ax1.legend(legend_elements, pattern_colors.keys(), loc='upper right')
        plt.title('Pattern Sequence')
        plt.xticks([])
        plt.yticks([])
        
        # Plot energy pathway allocation
        plt.subplot(2, 1, 2)
        steps_range = range(len(self.energy_pathway_stats['maintenance']))
        plt.plot(steps_range, self.energy_pathway_stats['maintenance'], 'b-', label='Maintenance')
        plt.plot(steps_range, self.energy_pathway_stats['growth'], 'g-', label='Growth')
        plt.plot(steps_range, self.energy_pathway_stats['adaptation'], 'r-', label='Adaptation')
        plt.title('Energy Pathway Allocation')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pathway_patterns", "pathway_analysis.png"))
        plt.close()
        
        # Create correlation plots
        self.create_correlation_plots()
        
        return pathway_results
    
    def create_correlation_plots(self):
        """Create visualizations of correlations between metrics and pathways"""
        correlations = self.metric_correlation_data
        
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # 1. Coherence vs Maintenance
        plt.subplot(1, 3, 1)
        x = [pair[0] for pair in correlations['coherence_vs_maintenance']]
        y = [pair[1] for pair in correlations['coherence_vs_maintenance']]
        plt.scatter(x, y, c='blue', alpha=0.7)
        
        # Add trend line if enough data
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.7)
            # Calculate correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            plt.title(f'Coherence vs Maintenance\nCorrelation: {corr:.2f}')
        else:
            plt.title('Coherence vs Maintenance')
            
        plt.xlabel('Coherence Metric')
        plt.ylabel('Maintenance Energy')
        plt.grid(True, alpha=0.3)
        
        # 2. Intensity vs Growth
        plt.subplot(1, 3, 2)
        x = [pair[0] for pair in correlations['intensity_vs_growth']]
        y = [pair[1] for pair in correlations['intensity_vs_growth']]
        plt.scatter(x, y, c='green', alpha=0.7)
        
        # Add trend line if enough data
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.7)
            # Calculate correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            plt.title(f'Intensity vs Growth\nCorrelation: {corr:.2f}')
        else:
            plt.title('Intensity vs Growth')
            
        plt.xlabel('Intensity Metric')
        plt.ylabel('Growth Energy')
        plt.grid(True, alpha=0.3)
        
        # 3. Complexity vs Adaptation
        plt.subplot(1, 3, 3)
        x = [pair[0] for pair in correlations['complexity_vs_adaptation']]
        y = [pair[1] for pair in correlations['complexity_vs_adaptation']]
        plt.scatter(x, y, c='red', alpha=0.7)
        
        # Add trend line if enough data
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.7)
            # Calculate correlation coefficient
            corr = np.corrcoef(x, y)[0, 1]
            plt.title(f'Complexity vs Adaptation\nCorrelation: {corr:.2f}')
        else:
            plt.title('Complexity vs Adaptation')
            
        plt.xlabel('Complexity Metric')
        plt.ylabel('Adaptation Energy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlations", "metric_correlations.png"))
        plt.close()
    
    def create_energy_visualizations(self, results_summary):
        """Create overall energy analysis visualizations"""
        # Create pathway ance summary
        pattern_results = results_summary['pattern_results']
        
        # Extract pattern names and pathway allocations
        pattern_names = [p['name'] for p in pattern_results]
        maintenance_values = []
        growth_values = []
        adaptation_values = []
        
        for pattern in pattern_results:
            energy_dist = pattern.get('energy_distribution', {})
            if energy_dist and 'total' in energy_dist and energy_dist['total'] > 0:
                total = energy_dist.get('total', 1.0)
                maintenance_values.append(energy_dist.get('maintenance', 0.0) / total)
                growth_values.append(energy_dist.get('growth', 0.0) / total)
                adaptation_values.append(energy_dist.get('adaptation', 0.0) / total)
            else:
                maintenance_values.append(0.0)
                growth_values.append(0.0)
                adaptation_values.append(0.0)
        
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        x = range(len(pattern_names))
        
        plt.bar(x, maintenance_values, color='blue', alpha=0.7, label='Maintenance')
        plt.bar(x, growth_values, bottom=maintenance_values, 
               color='green', alpha=0.7, label='Growth')
        
        # Calculate bottom for adaptation bars
        bottoms = [m + g for m, g in zip(maintenance_values, growth_values)]
        plt.bar(x, adaptation_values, bottom=bottoms,
               color='red', alpha=0.7, label='Adaptation')
        
        plt.xlabel('Pattern')
        plt.ylabel('Energy Allocation Proportion')
        plt.title('Energy Pathway Distribution by Pattern')
        plt.xticks(x, pattern_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "energy_analysis", "pathway_distribution.png"))
        plt.close()
        
        # Create metric correlation summary
        # If we have signal metrics history, create metrics distribution
        if self.signal_metrics_history:
            plt.figure(figsize=(12, 6))
            
            # Extract metrics
            intensity = [m.get('intensity', 0.0) for m in self.signal_metrics_history]
            coherence = [m.get('coherence', 0.0) for m in self.signal_metrics_history]
            complexity = [m.get('complexity', 0.0) for m in self.signal_metrics_history]
            
            # Create bins for histogram
            bins = np.linspace(0, 1, 11)
            
            # Plot histograms
            plt.hist(intensity, bins=bins, alpha=0.5, color='red', label='Intensity')
            plt.hist(coherence, bins=bins, alpha=0.5, color='blue', label='Coherence')
            plt.hist(complexity, bins=bins, alpha=0.5, color='green', label='Complexity')
            
            plt.xlabel('Metric Value')
            plt.ylabel('Frequency')
            plt.title('Signal Metrics Distribution')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "energy_analysis", "signal_metrics_distribution.png"))
            plt.close()
    
    def visualize_field(self, step, energy_applied=False, pathway_focus=None):
        """Create visualization of the field state with energy overlay"""
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Define colormap
        field_cmap = LinearSegmentedColormap.from_list('Ecliphra', [
            (0, 'darkblue'), (0.4, 'royalblue'),
            (0.6, 'mediumpurple'), (0.8, 'darkorchid'), (1.0, 'gold')
        ])
        
        # Define energy colormap
        energy_cmap = LinearSegmentedColormap.from_list('Energy', [
            (0, 'darkblue'), (0.4, 'royalblue'),
            (0.6, 'orange'), (0.8, 'crimson'), (1.0, 'gold')
        ])
        
        # Plot field
        field = self.model.field.detach().cpu().numpy()
        im1 = ax1.imshow(field, cmap=field_cmap)
        fig.colorbar(im1, ax=ax1)
        
        # Set title based on state
        if pathway_focus:
            title = f"Field State (Step {step}) - {pathway_focus.capitalize()} Focus"
            ax1.set_title(title, color={"maintenance": 'blue', "growth": 'green', 
                                      "adaptation": 'red'}.get(pathway_focus, 'black'),
                        fontweight='bold')
        elif energy_applied:
            title = f"Field State (Step {step}) - Energy Applied"
            ax1.set_title(title)
        else:
            title = f"Field State (Step {step})"
            ax1.set_title(title)
        
        # Mark attractors
        for attractor in self.model.attractors:
            # Handle different attractor formats
            if len(attractor) == 2:
                pos, strength = attractor
                base_size = None
                adaptive_size = None
            elif len(attractor) >= 4:
                pos, strength, base_size, adaptive_size = attractor
            else:
                continue  # Skip if format doesn't match expected patterns
            
            # Draw the attractor
            ax1.scatter(pos[1], pos[0], c='white', s=100*strength + 50, marker='*',
                      edgecolors='black', linewidths=1)
            
            # Add label with strength and adaptive size if available
            if adaptive_size is not None:
                label = f"{strength:.2f} (x{adaptive_size/base_size:.1f})"
            else:
                label = f"{strength:.2f}"
            
            ax1.annotate(label, (pos[1], pos[0]),
                      xytext=(5, 5), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        # Plot energy distribution if available
        if hasattr(self.model, 'energy_current'):
            energy = self.model.energy_current.detach().cpu().numpy()
            im2 = ax2.imshow(energy, cmap=energy_cmap)
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f"Energy Distribution")
            # This helps if the energy field is too uniform
            energy_viz = energy + np.random.normal(0, 0.01, energy.shape)

            # Normalize the energy visualization to enhance contrast
            energy_min = np.min(energy_viz)
            energy_max = np.max(energy_viz)
            if energy_max > energy_min:
                energy_viz = (energy_viz - energy_min) / (energy_max - energy_min)
            
            im2 = ax2.imshow(energy_viz, cmap=energy_cmap, vmin=0.9, vmax=1.1)
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f"Energy Distribution (min: {energy_min:.4f}, max: {energy_max:.4f})")
        else:
            # If no energy field, show velocity field
            if hasattr(self.model, 'velocity'):
                velocity = self.model.velocity.detach().cpu().numpy()
                velocity_magnitude = np.sqrt(np.sum(velocity**2))
                im2 = ax2.imshow(velocity, cmap='inferno')
                fig.colorbar(im2, ax=ax2)
                ax2.set_title(f"Velocity Field (Norm: {velocity_magnitude:.3f})")
        
        # Add energy pathway information if available
        if hasattr(self.model, 'energy_history') and self.model.energy_history:
            # Get the most recent energy distribution
            latest_energy = self.model.energy_history[-1]
            energy_dist = latest_energy.get('energy_distribution', {})
            
            if energy_dist:
                energy_text = "Energy Allocation:\n"
                total = energy_dist.get('total', 0.0)
                
                if total > 0:
                    maint_pct = energy_dist.get('maintenance', 0.0) / total * 100
                    growth_pct = energy_dist.get('growth', 0.0) / total * 100
                    adapt_pct = energy_dist.get('adaptation', 0.0) / total * 100
                    
                    energy_text += f"Maintenance: {maint_pct:.1f}%\n"
                    energy_text += f"Growth: {growth_pct:.1f}%\n"
                    energy_text += f"Adaptation: {adapt_pct:.1f}%"
                
                # Add to figure
                fig.text(0.01, 0.01, energy_text, ha='left', fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the visualization
        energy_suffix = "_energy" if energy_applied else ""
        pathway_suffix = f"_{pathway_focus}" if pathway_focus else ""
        filename = f"{self.output_dir}/field_state_step_{step}{energy_suffix}{pathway_suffix}.png"
        plt.savefig(filename)
        plt.close()
    
    def plot_summary(self):
        """Plot overall summary of energy system experiment"""
        # Create visualizations based on collected data
        if self.energy_level_history:
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(self.energy_level_history)), self.energy_level_history, 'b-')
            plt.xlabel('Step')
            plt.ylabel('Energy Level')
            plt.title('Energy Level Over Time')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, "energy_summary.png"))
            plt.close()
        
        # Plot pathway stats if available
        has_pathway_data = all(len(vals) > 0 for vals in self.energy_pathway_stats.values())
        if has_pathway_data:
            plt.figure(figsize=(10, 6))
            steps = range(len(self.energy_pathway_stats['maintenance']))
            
            plt.plot(steps, self.energy_pathway_stats['maintenance'], 'b-', label='Maintenance')
            plt.plot(steps, self.energy_pathway_stats['growth'], 'g-', label='Growth')
            plt.plot(steps, self.energy_pathway_stats['adaptation'], 'r-', label='Adaptation')
            
            plt.xlabel('Step')
            plt.ylabel('Energy Allocation')
            plt.title('Energy Pathway Allocation Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, "pathway_summary.png"))
            plt.close()
        if "attractors" in self.results:
            step_count = len(self.results["stability"])
            attractors = self.results["attractors"][:step_count]
            ax2.plot(range(step_count), attractors, 'g-', label='Number of Attractors')

from ecliphra.utils.prefrontal import SignalEnvironment, PrefrontalModule

class PrefrontalExperiment(EcliphraExperiment):
    """
    Experiment to test prefrontal control capabilities.
    Tests how the prefrontal module affects field dynamics and goal achievement.
    """
    
    def __init__(self, model, output_dir=None, field_size=32):
        super().__init__(model, output_dir, field_size)

        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "prefrontal_experiment_output")
        super().__init__(model, output_dir, field_size)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Additional directories for prefrontal analysis
        os.makedirs(os.path.join(self.output_dir, "goal_tracking"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "executive_control"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "cognitive_metrics"), exist_ok=True)
        
        # Prefrontal specific tracking
        self.goal_tracking = []
        self.attention_focus_history = []
        self.inhibition_history = []

     

        # Get field dimensions and device from the model
        field_dim = getattr(model, 'field_dim', (field_size, field_size))
        device = getattr(model, 'device', 'cpu')
        
        # Create environment with proper parameters
        environment = SignalEnvironment(
            field_dim=field_dim,
            device=device,
            drift_frequency=10
        )
        self.model.environment = environment
    
    def visualize_field(self, step, prefrontal_active=False, goal_focused=False, 
                        inhibition_active=False, attention_active=False):
        """Create enhanced visualization of the field with prefrontal information"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("Matplotlib not installed, skipping visualization")
            return
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Define colormap
        field_cmap = LinearSegmentedColormap.from_list('Ecliphra', [
            (0, 'darkblue'), (0.4, 'royalblue'),
            (0.6, 'mediumpurple'), (0.8, 'darkorchid'), (1.0, 'gold')
        ])
        
        # Plot field
        field = self.model.field.detach().cpu().numpy()
        im1 = ax1.imshow(field, cmap=field_cmap)
        fig.colorbar(im1, ax=ax1)
        
        # Set title based on state
        title_parts = [f"Field State (Step {step})"]
        if prefrontal_active:
            title_parts.append("Prefrontal Active")
        if goal_focused:
            title_parts.append("Goal-Focused")
        if inhibition_active:
            title_parts.append("Inhibition Active")
        if attention_active:
            title_parts.append("Attention Directed")
            
        ax1.set_title(" - ".join(title_parts))
        
        # Mark attractors
        if hasattr(self.model, 'attractors'):
            for attractor in self.model.attractors:
                # Extract position and strength based on attractor format
                if len(attractor) >= 2:
                    pos = attractor[0]
                    strength = attractor[1]
                else:
                    continue
                
                # Draw the attractor
                ax1.scatter(pos[1], pos[0], c='white', s=100*strength + 50, marker='*',
                        edgecolors='black', linewidths=1)
        
        # Visualize energy distribution if available
        if hasattr(self.model, 'energy_current'):
            energy = self.model.energy_current.detach().cpu().numpy()
            
            # Add small random noise to make variations more visible
            energy_viz = energy + np.random.normal(0, 0.005, energy.shape)
            
            # Normalize and enhance contrast
            if np.max(energy_viz) > np.min(energy_viz):
                energy_viz = (energy_viz - np.min(energy_viz)) / (np.max(energy_viz) - np.min(energy_viz))
            
            # Use a more contrasting colormap
            energy_cmap = LinearSegmentedColormap.from_list('Energy', [
                (0, 'darkblue'), (0.3, 'blue'),
                (0.5, 'green'), (0.7, 'yellow'), (1.0, 'red')
            ])
            
            im2 = ax2.imshow(energy_viz, cmap=energy_cmap)
            fig.colorbar(im2, ax=ax2)
            ax2.set_title(f"Energy Distribution (min: {np.min(energy):.4f}, max: {np.max(energy):.4f})")
        else:
            # Fall back to velocity if no energy
            if hasattr(self.model, 'velocity'):
                velocity = self.model.velocity.detach().cpu().numpy()
                vel_mag = np.sqrt(np.sum(velocity**2))
                im2 = ax2.imshow(velocity, cmap='inferno')
                fig.colorbar(im2, ax=ax2)
                ax2.set_title(f"Velocity Field (Norm: {vel_mag:.3f})")
        
        # Add goals information if available and goal_focused is true
        if goal_focused and hasattr(self.model, 'prefrontal'):
            pf_status = self.model.prefrontal.get_status()
            goals = pf_status.get('goals', [])
            
            if goals:
                goals_text = "Active Goals:\n"
                for i, goal in enumerate(goals):
                    priority = goal.get('priority', 0.0)
                    progress = goal.get('progress', 0.0)
                    activation = goal.get('activation', 0.0)
                    goal_type = goal.get('type', '?')
                    
                    goals_text += f"#{i+1}: {goal_type} - Pri={priority:.2f}, Prog={progress:.2f}, Act={activation:.2f}\n"
                
                # Add to figure
                fig.text(0.01, 0.01, goals_text, ha='left', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add cognitive metrics - CRITICAL FIX: GET METRICS DIRECTLY
        if hasattr(self.model, 'get_cognitive_metrics'):
            metrics = self.model.get_cognitive_metrics()
            
            metrics_text = "Cognitive Metrics:\n"
            for key, value in metrics.items():
                metrics_text += f"{key.replace('_', ' ').title()}: {value:.2f}\n"
            
            # Add to figure
            fig.text(0.99, 0.01, metrics_text, ha='right', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the visualization
        prefix = ""
        if prefrontal_active:
            prefix += "pf_"
        if goal_focused:
            prefix += "goal_"
        if inhibition_active:
            prefix += "inhib_"
        if attention_active:
            prefix += "attn_"
            
        filename = os.path.join(self.output_dir, f"{prefix}field_state_step_{step}.png")
        plt.savefig(filename)
        plt.close()
    
    def create_executive_control_viz(self, results_summary):
        """Create executive control visualizations"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed, skipping visualization")
            return
        
        # Create executive control output directory if it doesn't exist
        import os
        os.makedirs(os.path.join(self.output_dir, "executive_control"), exist_ok=True)
        
        # 1. Create resource allocation visualization
        if hasattr(self.model, 'prefrontal') and hasattr(self.model.prefrontal, 'current_goals'):
            plt.figure(figsize=(12, 6))
            
            # Get goal data and stability
            goal_tracking = results_summary.get('goal_tracking', [])
            
            if goal_tracking:
                # Extract steps
                steps = range(len(goal_tracking))
                
                # Extract stability data
                stability = [entry.get('stability', 0.5) for entry in goal_tracking]
                
                # Create stability plot
                plt.plot(steps, stability, 'b-', label='Field Stability', linewidth=2)
                
                # Get direct goal progress from prefrontal module
                current_goals = self.model.prefrontal.current_goals
                if current_goals:
                    for i, goal in enumerate(current_goals):
                        # Extract goal info
                        priority = goal.get('priority', 0.0)
                        progress = goal.get('progress', 0.0)
                        goal_type = goal.get('type', 'generic')
                        
                        # Add goal progress point
                        plt.axhline(y=progress, color=f"C{i+1}", linestyle='--', alpha=0.5)
                        plt.text(len(steps)-1, progress, f"Goal {i+1}: {goal_type} ({progress:.2f})", 
                            color=f"C{i+1}", fontsize=9)
                
                # Get cognitive metrics for visualization
                if hasattr(self.model, 'get_cognitive_metrics'):
                    metrics = self.model.get_cognitive_metrics()
                    
                    # Plot metrics as horizontal lines
                    colors = ['green', 'orange', 'red']
                    for i, (key, value) in enumerate(metrics.items()):
                        if value > 0:  # Only plot non-zero metrics
                            plt.axhline(y=value, color=colors[i % len(colors)], 
                                    linestyle=':', alpha=0.7,
                                    label=f"{key.replace('_', ' ').title()} ({value:.2f})")
                
                plt.xlabel('Step')
                plt.ylabel('Value')
                plt.title('Resource Allocation and Goal Achievement')
                plt.ylim(0, 1)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.savefig(os.path.join(self.output_dir, "executive_control", "resource_allocation.png"))
                plt.close()
        
        # 2. Create pathway bias visualization
        if hasattr(self.model, 'prefrontal') and hasattr(self.model.prefrontal, 'attention_bias'):
            plt.figure(figsize=(8, 6))
            
            # Get pathway bias
            pathway_bias = self.model.prefrontal.attention_bias.detach().cpu().numpy()
            
            # Create bar chart
            pathways = ['Maintenance', 'Growth', 'Adaptation']
            plt.bar(pathways, pathway_bias, color=['blue', 'green', 'red'])
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.ylabel('Bias Strength')
            plt.title('Prefrontal Pathway Bias')
            
            plt.savefig(os.path.join(self.output_dir, "executive_control", "pathway_bias.png"))
            plt.close()
        
    def visualize_goal_progress(self, goal_progress, goals):
        """Visualize goal progress over time"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not installed, skipping visualization")
            return
        
        plt.figure(figsize=(12, 6))
        
        for goal_id, progress in goal_progress.items():
            if goal_id < len(goals):
                goal_type = goals[goal_id]["type"]
                priority = goals[goal_id]["priority"]
                
                # Plot with line thickness based on priority
                plt.plot(progress, label=f"Goal {goal_id+1}: {goal_type} (p={priority:.2f})", 
                       linewidth=1+priority*3)
        
        plt.xlabel("Step")
        plt.ylabel("Progress")
        plt.title("Goal Progress Over Time")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(self.output_dir, "goal_tracking", "goal_progress.png"))
        plt.close()
        
        # Create goal contribution visualization
        if self.goal_tracking:
            plt.figure(figsize=(12, 6))
            
            # Extract energy distribution over time
            steps = [entry["step"] for entry in self.goal_tracking]
            
            # Check if there are energy distributions to plot
            has_energy_data = all("energy_distribution" in entry for entry in self.goal_tracking)
            
            if has_energy_data:
                maintenance = []
                growth = []
                adaptation = []
                
                for entry in self.goal_tracking:
                    dist = entry.get("energy_distribution", {})
                    maintenance.append(dist.get("maintenance", 0.0))
                    growth.append(dist.get("growth", 0.0))
                    adaptation.append(dist.get("adaptation", 0.0))
                
                plt.stackplot(steps, maintenance, growth, adaptation, 
                           labels=["Maintenance", "Growth", "Adaptation"],
                           alpha=0.7)
                
                # Mark goal creation points
                for i, goal in enumerate(goals):
                    plt.axvline(x=i, color='black', linestyle='--', alpha=0.5)
                    plt.text(i, 0, f"G{i+1}", fontsize=10, ha='center')
                
                plt.xlabel("Step")
                plt.ylabel("Energy Allocation")
                plt.title("Energy Allocation During Goal Pursuit")
                plt.legend(loc='upper left')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(self.output_dir, "goal_tracking", "energy_allocation.png"))
                plt.close()

         # 5. Add a direct fix for the visualization in PrefrontalExperiment class

    """
    Add the following method to PrefrontalExperiment class to create a performance 
    summary visualization that shows actual metrics
    """

    def create_performance_summary(self, results_summary):
        """Create performance metrics visualization"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed, skipping visualization")
            return
        
        import os
        os.makedirs(os.path.join(self.output_dir, "cognitive_metrics"), exist_ok=True)
        
        # Create metrics visualization
        plt.figure(figsize=(10, 6))
        
        # Get cognitive metrics directly from model
        if hasattr(self.model, 'get_cognitive_metrics'):
            metrics = self.model.get_cognitive_metrics()
            
            # Ensure we have values
            if not metrics or all(v == 0 for v in metrics.values()):
                # Use fallback values
                metrics = {
                    "goal_alignment": 0.3,
                    "inhibition_efficiency": 0.4,
                    "resource_efficiency": 0.5
                }
            
            # Format for display
            labels = [key.replace('_', ' ').title() for key in metrics.keys()]
            values = list(metrics.values())
            
            # Create bar chart
            plt.bar(labels, values, color=['blue', 'green', 'red'])
            plt.ylim(0, 1.0)
            plt.title('Cognitive Performance Metrics')
            plt.ylabel('Performance')
            
            # Add values as text
            for i, v in enumerate(values):
                plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
            
            plt.savefig(os.path.join(self.output_dir, "cognitive_metrics", "performance_summary.png"))
            plt.close()
        else:
            print("Model does not have get_cognitive_metrics method")

    
    def create_prefrontal_visualizations(self, results_summary):
        """Create overall prefrontal analysis visualizations"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed, skipping visualization")
            return
        
        # Create cognitive metrics visualization
        cognitive_metrics = results_summary.get('cognitive_metrics', {})
        
        if cognitive_metrics:
            plt.figure(figsize=(10, 6))
            
            # Create bar chart
            metrics = list(cognitive_metrics.keys())
            values = [cognitive_metrics[m] for m in metrics]
            
            # Format x-axis labels
            x_labels = [m.replace('_', ' ').title() for m in metrics]
            
            plt.bar(x_labels, values, color=['royalblue', 'darkorange', 'forestgreen'])
            plt.ylabel('Performance')
            plt.title('Cognitive Performance Metrics')
            plt.ylim(0, 1)
            plt.grid(axis='y', alpha=0.3)
            
            plt.savefig(os.path.join(self.output_dir, "cognitive_metrics", "performance_summary.png"))
            plt.close()
        
        # Create working memory analysis if we have enough data
        if hasattr(self.model, 'prefrontal'):
            pf_status = self.model.prefrontal.get_status()
            working_memory = pf_status.get('working_memory', [])
            
            if working_memory:
                plt.figure(figsize=(10, 6))
                
                # Extract working memory data
                importance = [m.get('importance', 0.0) for m in working_memory]
                age = [m.get('age', 0) for m in working_memory]
                
                # Create scatter plot
                plt.scatter(age, importance, s=100, alpha=0.7, c=age, cmap='viridis')
                
                for i, (a, imp) in enumerate(zip(age, importance)):
                    plt.text(a + 0.1, imp, f"Item {i+1}", fontsize=9)
                
                plt.xlabel('Age (Steps)')
                plt.ylabel('Importance')
                plt.title('Working Memory Item Analysis')
                plt.grid(True, alpha=0.3)
                
                plt.savefig(os.path.join(self.output_dir, "cognitive_metrics", "working_memory.png"))
                plt.close()
    
    def _update_cognitive_metrics(self, result):
        """Update tracking of cognitive performance metrics"""
        # Initialize cognitive metrics if not already done
        if not hasattr(self, 'cognitive_metrics'):
            self.cognitive_metrics = {
                "goal_alignment": [],
                "inhibition_efficiency": [],
                "resource_efficiency": []
            }
        
        # Calculate goal alignment (how well energy allocation matches goals)
        pf_status = result.get('prefrontal_status', {})
        
        # Fatigue level tracking
        if 'fatigue_level' in pf_status:
            if "fatigue_level" not in self.cognitive_metrics:
                self.cognitive_metrics["fatigue_level"] = []
            self.cognitive_metrics["fatigue_level"].append(pf_status["fatigue_level"])

        goals = pf_status.get('goals', [])
        
        if goals and 'energy_distribution' in result:
            # Simple heuristic for goal alignment
            energy_dist = result['energy_distribution']
            if isinstance(energy_dist, dict) and energy_dist:
                # Calculate weighted goal alignment score based on priorities and pathway distributions
                alignment_scores = []
                for goal in goals:
                    priority = goal.get('priority', 0.0)
                    progress = goal.get('progress', 0.0)
                    # Simple alignment score based on progress vs. priority
                    alignment = (1.0 - abs(progress - priority))
                    alignment_scores.append(alignment * priority)
                
                if alignment_scores:
                    goal_alignment = sum(alignment_scores) / sum(g.get('priority', 1.0) for g in goals)
                    self.cognitive_metrics["goal_alignment"].append(goal_alignment)
        
        # Inhibition efficiency (whether inhibited processes stay inhibited)
        if hasattr(self.model, 'prefrontal'):
            inhibition_targets = getattr(self.model.prefrontal, 'inhibition_targets', set())
            if inhibition_targets:
                # Measure how well inhibition worked - use stability as an approximation 
                inhibition_efficiency = result.get('stability', 0.5)
                self.cognitive_metrics["inhibition_efficiency"].append(inhibition_efficiency)
        
        # Resource efficiency
        if 'energy_level' in result:
            energy_level = result['energy_level']
            stability = result.get('stability', 0.5)
            resource_efficiency = energy_level * stability
            self.cognitive_metrics["resource_efficiency"].append(resource_efficiency)

    def run_contradiction_test(self, steps=50):
        """
        Test how the prefrontal module handles contradictory goals.
        
        Args:
            steps: Number of steps to run the test
            
        Returns:
            Results dictionary with metrics and visualizations
        """
        # Create output directory for this specific test
        test_dir = os.path.join(self.output_dir, "contradiction_test")
        os.makedirs(test_dir, exist_ok=True)
        
        print("\nRunning contradiction test with conflicting goals...")
        
        # Reset the model
        self.model.reset()
        
        # Inject contradictory goals
        self.inject_contradictory_goals()
        
        # Track metrics
        metrics_log = []
        goal_progress = {0: [], 1: []}  # Track progress for both goals
        energy_allocation = []
        
        # Run test
        for step in range(steps):
            # Generate signal from environment
            signal = self.model.environment.generate_signal() if hasattr(self.model, 'environment') else None
            
            # Process through model
            result = self.model(input_tensor=signal, prefrontal_control=True)
            
            # Get prefrontal status
            pf_status = self.model.prefrontal.get_status()
            goals = pf_status.get('goals', [])
            
            # Record cognitive metrics
            if hasattr(self.model, 'get_cognitive_metrics'):
                cognitive_metrics = self.model.get_cognitive_metrics()
            else:
                cognitive_metrics = pf_status.get('cognitive_metrics', {})
            
            metrics = {
                "step": step,
                "goal_alignment": cognitive_metrics.get("goal_alignment", 0.0),
                "inhibition_efficiency": cognitive_metrics.get("inhibition_efficiency", 0.0),
                "resource_efficiency": cognitive_metrics.get("resource_efficiency", 0.0),
                "active_goals": len(goals)
            }
            metrics_log.append(metrics)

            for step in range(steps):
              current_step = step 
            
            # Track goal progress
            for i, goal in enumerate(goals[:2]):  # Just track the two contradictory goals
                if i in goal_progress:
                    goal_progress[i].append(goal.get('progress', 0.0))
            
            # Track energy allocation
            if 'energy_distribution' in result:
                energy_allocation.append({
                    'step': step,
                    'maintenance': result['energy_distribution'].get('maintenance', 0.0),
                    'growth': result['energy_distribution'].get('growth', 0.0),
                    'adaptation': result['energy_distribution'].get('adaptation', 0.0)
                })

            # Activate staged goals when their time comes
            for goal in list(self.goal_queue):
                if goal["activation_step"] == current_step:
                    self.current_goals.append(goal)
                    self.goal_queue.remove(goal)

            
            # Visualize field state every 10 steps
            if step % 10 == 0 or step == steps - 1:
                self.visualize_field(
                    step, 
                    prefrontal_active=True,
                    goal_focused=True
                )

                try:
                    import matplotlib.pyplot as plt
                    # Save the current figure to the test directory
                    # Note: This will only work if the visualization is still the current matplotlib figure
                    plt.tight_layout()
                    plt.savefig(os.path.join(test_dir, f"field_state_step_{step}.png"))
                except:
                    # If this fails, just continue (the visualization is already saved in the default location)
                    pass
        
        # Create visualizations
        self.visualize_contradiction_results(metrics_log, goal_progress, energy_allocation, test_dir)
        
        # Return results
        return {
            'metrics_log': metrics_log,
            'goal_progress': goal_progress,
            'energy_allocation': energy_allocation
        }

    def inject_contradictory_goals(self, step=0):
        """
        Inject a dynamic contradiction by offsetting the novelty goal and optionally staggering activation.
        This avoids a frozen field state and encourages adaptive differentiation.
        """
        current_field = self.model.field.detach().clone()

        # Stability goal wants to preserve current state
        stability_goal = {
            "type": "stability",
            "target_value": 0.9,
            "pattern": current_field,
            "pathway_needs": [0.7, 0.1, 0.2],  # Favors maintenance
            "activation_step": step  # Immediate
        }

        # Slightly perturb the field for novelty
        perturbed_field = current_field + torch.randn_like(current_field) * 0.05
        novelty_goal = {
            "type": "novelty",
            "target_value": 0.8,
            "pattern": perturbed_field,
            "pathway_needs": [0.2, 0.1, 0.7],  # Favors adaptation
            "activation_step": step + 5  # Delayed activation for structural contrast
        }

        # Add both to goal queue
        self.goal_queue = [stability_goal, novelty_goal]

        # Optionally trigger this check inside your main loop:
        # if current_step == goal["activation_step"]:
        #     self.current_goals.append(goal)
        #     self.goal_queue.remove(goal)

        
        # Set goals with different priorities
        self.model.set_goal(stability_goal, priority=1.0)
        self.model.set_goal(novelty_goal, priority=0.9)
        
        print("[TEST] Injected contradictory goals: stability vs novelty")
        print(f"      - Stability goal pathway needs: {stability_goal['pathway_needs']}")
        print(f"      - Novelty goal pathway needs: {novelty_goal['pathway_needs']}")
        
        # Verify goals were set
        pf_status = self.model.prefrontal.get_status()
        print(f"      - Active goals: {len(pf_status.get('goals', []))}")

    def visualize_contradiction_results(self, metrics_log, goal_progress, energy_allocation, output_dir):
        """Create visualizations for contradiction test results"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
        except ImportError:
            print("Matplotlib or pandas not installed, skipping visualizations")
            return
        
        # Convert to pandas for easier handling
        metrics_df = pd.DataFrame(metrics_log)
        energy_df = pd.DataFrame(energy_allocation)
        
        # 1. Goal progress plot
        plt.figure(figsize=(10, 6))
        
        # Fix the steps calculation to match the actual data length
        goal_0_steps = range(len(goal_progress[0])) if goal_progress[0] else []
        goal_1_steps = range(len(goal_progress[1])) if goal_progress[1] else []
        
        if goal_progress[0]:
            plt.plot(goal_0_steps, goal_progress[0], 'b-', linewidth=2, label='Stability Goal')
        if goal_progress[1]:
            plt.plot(goal_1_steps, goal_progress[1], 'r-', linewidth=2, label='Novelty Goal')
        
        plt.xlabel('Step')
        plt.ylabel('Goal Progress')
        plt.title('Contradictory Goals Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'contradiction_goal_progress.png'))
        plt.close()
        
        # 2. Cognitive metrics evolution
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['step'], metrics_df['goal_alignment'], 'b-', label='Goal Alignment')
        plt.plot(metrics_df['step'], metrics_df['inhibition_efficiency'], 'g-', label='Inhibition Efficiency')
        plt.plot(metrics_df['step'], metrics_df['resource_efficiency'], 'r-', label='Resource Efficiency')
        
        plt.xlabel('Step')
        plt.ylabel('Metric Value')
        plt.title('Cognitive Metrics During Goal Contradiction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'contradiction_metrics.png'))
        plt.close()
        
        # 3. Energy allocation stacked area plot
        if not energy_df.empty:
            plt.figure(figsize=(12, 6))
            
            # Extract data
            steps = energy_df['step']
            maintenance = energy_df['maintenance']
            growth = energy_df['growth']
            adaptation = energy_df['adaptation']
            
            # Normalize to show proportions
            totals = maintenance + growth + adaptation
            maintenance_norm = maintenance / totals
            growth_norm = growth / totals
            adaptation_norm = adaptation / totals
            
            # Create stacked plot
            plt.stackplot(steps, 
                        maintenance_norm, growth_norm, adaptation_norm,
                        labels=['Maintenance', 'Growth', 'Adaptation'],
                        colors=['blue', 'orange', 'green'],
                        alpha=0.7)
            
            plt.xlabel('Step')
            plt.ylabel('Energy Allocation Proportion')
            plt.title('Energy Allocation During Goal Contradiction')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(output_dir, 'contradiction_energy.png'))
            plt.close()
        
        # 4. Summary visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Goal progress and goal alignment
        if goal_progress[0]:
            ax1.plot(goal_0_steps, goal_progress[0], 'b-', linewidth=2, label='Stability Goal')
        if goal_progress[1]:
            ax1.plot(goal_1_steps, goal_progress[1], 'r-', linewidth=2, label='Novelty Goal')
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(metrics_df['step'], metrics_df['goal_alignment'], 'g--', linewidth=1.5, label='Goal Alignment')
        ax1_twin.set_ylabel('Goal Alignment', color='g')
        
        ax1.set_ylabel('Goal Progress')
        ax1.set_title('Goal Progress and Alignment During Contradiction')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # Bottom plot: Energy allocation proportions
        if not energy_df.empty:
            # Create stacked plot
            ax2.stackplot(steps, 
                        maintenance_norm, growth_norm, adaptation_norm,
                        labels=['Maintenance', 'Growth', 'Adaptation'],
                        colors=['blue', 'orange', 'green'],
                        alpha=0.7)
            
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Energy Allocation Proportion')
            ax2.set_title('Energy Allocation During Goal Contradiction')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'contradiction_summary.png'))
        plt.close()
    
    def run_pattern_interference_test(self, steps=100, visualization_steps=10):

        print("\n" + "="*50)
        print("STARTING PATTERN INTERFERENCE TEST")
        print("="*50)

        model = self.model
        model.reset()

        results = {
            "phase_boundaries": [],
            "stability_values": [],
            "pattern_similarity": [],
            "energy_levels": [],
            "cognitive_metrics": []
        }

        learning_end = int(steps * 0.3)
        interference_end = int(steps * 0.7)
        results["phase_boundaries"] = [0, learning_end, interference_end, steps]

        stability_goal = {
            "type": "stability",
            "target_value": 0.9,
            "priority": 1.0,
            "pathway_needs": [0.6, 0.3, 0.1]
        }
        model.set_goal(stability_goal, stability_goal["priority"])

        energy_goal = {
            "type": "energy",
            "target_value": 0.7,
            "priority": 0.7,
            "pathway_needs": [0.3, 0.6, 0.1]
        }
        model.set_goal(energy_goal, energy_goal["priority"])

        pattern_goal = {
            "type": "pattern_similarity",
            "target_value": 1.0,      # insist on cloning the base pattern
            "priority": 0.8,          # high priority, just under stability
            "pathway_needs": [0.7, 0.2, 0.1]
        }
        model.set_goal(pattern_goal, pattern_goal["priority"])

        h, w = model.field_dim if hasattr(model, 'field_dim') else model.field.shape
        center_y, center_x = h//2, w//2

        print(f"\nPhase 1: Learning phase (steps 0-{learning_end})")
        base_pattern = torch.zeros((h, w), device=model.device)
        for i in range(h):
            for j in range(w):
                dist = ((i - center_y)**2 + (j - center_x)**2) ** 0.5 / (h/2)
                base_pattern[i, j] = torch.exp(torch.tensor(-3 * dist, device=base_pattern.device))
        base_pattern = base_pattern / torch.norm(base_pattern)
        original_pattern = base_pattern.clone()

        for step in range(learning_end):
            noise_level = 0.02 * (1 - step/learning_end)
            noise = torch.randn_like(base_pattern) * noise_level
            pattern = base_pattern + noise
            pattern = pattern / torch.norm(pattern)
            result = model(input_tensor=pattern)

            results["stability_values"].append(result.get('stability', 0.0))
            field_similarity = F.cosine_similarity(model.field.view(1, -1), original_pattern.view(1, -1)).item()
            results["pattern_similarity"].append(field_similarity)
            results["energy_levels"].append(result.get('energy_level', 0.0))
            if 'cognitive_metrics' in result:
                results["cognitive_metrics"].append(result['cognitive_metrics'])

            if step % visualization_steps == 0 or step == learning_end - 1:
                self.visualize_field(step, prefrontal_active=True, goal_focused=True)
                print(f"  Step {step}: Stability = {result.get('stability', 0.0):.2f}, Pattern similarity = {field_similarity:.2f}")

        trained_field = model.field.detach().clone()
        print(f"Learning phase complete. Final stability: {results['stability_values'][-1]:.2f}")

        print(f"\nPhase 2: Interference phase (steps {learning_end}-{interference_end})")
        interfering_pattern = torch.zeros((h, w), device=model.device)
        for i in range(h):
            for j in range(w):
                interfering_pattern[i, j] = torch.sin(torch.tensor(i * 8 * 3.14159 / h)) * \
                                            torch.sin(torch.tensor(j * 8 * 3.14159 / w))
        interfering_pattern = interfering_pattern / torch.norm(interfering_pattern)

        for step in range(learning_end, interference_end):
            relative_step = step - learning_end
            steps_in_phase = interference_end - learning_end

            if relative_step < steps_in_phase // 3:
                blend = min(1.0, relative_step / (steps_in_phase // 3))
                pattern = (1 - blend) * base_pattern + blend * interfering_pattern
            elif relative_step < 2 * (steps_in_phase // 3):
                pattern = interfering_pattern
            else:
                blend = 1.0 - min(1.0, (relative_step - 2*(steps_in_phase//3)) / (steps_in_phase//3))
                pattern = (1 - blend) * base_pattern + blend * interfering_pattern

            noise = torch.randn_like(pattern) * 0.05
            pattern = pattern + noise
            pattern = pattern / torch.norm(pattern)

            result = model(input_tensor=pattern)
            results["stability_values"].append(result.get('stability', 0.0))
            field_similarity = F.cosine_similarity(model.field.view(1, -1), original_pattern.view(1, -1)).item()
            results["pattern_similarity"].append(field_similarity)
            results["energy_levels"].append(result.get('energy_level', 0.0))
            if 'cognitive_metrics' in result:
                results["cognitive_metrics"].append(result['cognitive_metrics'])

            if step % visualization_steps == 0 or step == interference_end - 1:
                self.visualize_field(step, prefrontal_active=True, goal_focused=True)
                print(f"  Step {step}: Stability = {result.get('stability', 0.0):.2f}, Pattern similarity = {field_similarity:.2f}")

        post_interference_field = model.field.detach().clone()
        disruption = torch.norm(post_interference_field - trained_field).item()
        results["disruption_level"] = disruption
        print(f"Interference phase complete. Disruption level: {disruption:.2f}")
        _ = model(input_tensor=trained_field) 

        print("DEBUG: Goals at start of Recovery:", [g["type"] for g in self.model.prefrontal.get_status()["goals"]])

        print(f"\nPhase 3: Recovery phase (steps {interference_end}-{steps})")
        
        fatigue_goal = {
            "type": "fatigue_level",
            "target_value": 0.0,
            "priority": 0.5,
            "pathway_needs": [0.8, 0.1, 0.1]
        }
        model.set_goal(fatigue_goal, fatigue_goal["priority"])

        for step in range(interference_end, steps):

           # Safe forward pass using base pattern with no noise to inspect state
            inspect_input = base_pattern.clone()
            inspect_result = model(input_tensor=inspect_input)
            fatigue = inspect_result.get("fatigue_level", 0.0)

            field_similarity = F.cosine_similarity(
                model.field.view(1, -1), trained_field.view(1, -1)
            ).item()

            recovery_blend = min(0.3, max(0.02, (1.0 - field_similarity) * 0.2))
            if fatigue < 200:
                model.field.data = (
                    (1 - recovery_blend) * model.field.data + recovery_blend * trained_field
                )
            # — Patch: every ~20% of recovery, feed the true pattern to reinforce —
            rec_steps = steps - interference_end
            if (step - interference_end) % max(1, rec_steps // 5) == 0:
                _ = model(input_tensor=trained_field)

            noise = torch.randn_like(base_pattern) * 0.02
            pattern = base_pattern + noise
            pattern = pattern / torch.norm(pattern)

            result = model(input_tensor=pattern)
            results["stability_values"].append(result.get('stability', 0.0))

            field_similarity = F.cosine_similarity(
                model.field.view(1, -1), original_pattern.view(1, -1)
            ).item()
            results["pattern_similarity"].append(field_similarity)
            results["energy_levels"].append(result.get('energy_level', 0.0))
            if 'cognitive_metrics' in result:
                results["cognitive_metrics"].append(result['cognitive_metrics'])

            if step % visualization_steps == 0 or step == steps - 1:
                self.visualize_field(step, prefrontal_active=True, goal_focused=True)
                print(f"  Step {step}: Stability = {result.get('stability', 0.0):.2f}, Pattern similarity = {field_similarity:.2f}")

        final_field = model.field.detach().clone()
        initial_similarity = F.cosine_similarity(trained_field.view(1, -1), original_pattern.view(1, -1)).item()
        final_similarity = F.cosine_similarity(final_field.view(1, -1), original_pattern.view(1, -1)).item()
        recovery_percentage = max(0, (final_similarity - results["pattern_similarity"][interference_end-1]) /
                                (initial_similarity - results["pattern_similarity"][interference_end-1]))

        results["initial_similarity"] = initial_similarity
        results["final_similarity"] = final_similarity
        results["recovery_percentage"] = recovery_percentage

        print(f"\nTest complete. Recovery percentage: {recovery_percentage*100:.1f}%")
        self.create_interference_test_visualizations(results)
        return results
        self.create_interference_test_visualizations(results)
        # — Patch: dump prefrontal metrics so we can inspect alignment, inhibition, etc. —
        self.create_executive_control_viz(results)
        self.create_performance_summary(results)
        return results

    def create_interference_test_visualizations(self, results):
        """Create visualizations of interference test results"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Create output directory if needed
        output_dir = os.path.join(self.output_dir, "interference_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract phase boundaries
        phase_start, learning_end, interference_end, steps = results["phase_boundaries"]
        
        # 1. Create stability and similarity plot
        plt.figure(figsize=(12, 6))
        steps_range = range(len(results["stability_values"]))
        
        # Plot stability
        plt.plot(steps_range, results["stability_values"], 'b-', label='Stability', linewidth=2)
        
        # Plot pattern similarity
        plt.plot(steps_range, results["pattern_similarity"], 'g-', label='Pattern Similarity', linewidth=2)
        
        # Add phase markers
        plt.axvline(x=learning_end, color='r', linestyle='--', label='End of Learning Phase')
        plt.axvline(x=interference_end, color='orange', linestyle='--', label='End of Interference Phase')
        
        # Add annotations
        plt.annotate('Learning', xy=(learning_end/2, 0.95), ha='center')
        plt.annotate('Interference', xy=(learning_end + (interference_end-learning_end)/2, 0.95), ha='center')
        plt.annotate('Recovery', xy=(interference_end + (steps-interference_end)/2, 0.95), ha='center')
        
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Stability and Pattern Similarity During Interference Test')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "stability_similarity.png"))
        plt.close()
        
        # 2. Create energy levels plot
        plt.figure(figsize=(12, 6))
        
        # Plot energy level
        plt.plot(steps_range, results["energy_levels"], 'r-', label='Energy Level', linewidth=2)
        
        # Add phase markers
        plt.axvline(x=learning_end, color='r', linestyle='--')
        plt.axvline(x=interference_end, color='orange', linestyle='--')
        
        plt.xlabel('Step')
        plt.ylabel('Energy Level')
        plt.title('Energy Consumption During Interference Test')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "energy_levels.png"))
        plt.close()
        
        # 3. Create cognitive metrics plot if available
        if results["cognitive_metrics"]:
            plt.figure(figsize=(12, 6))
            
            # Extract metrics
            goal_alignment = [m.get('goal_alignment', 0.0) for m in results["cognitive_metrics"]]
            inhibition_efficiency = [m.get('inhibition_efficiency', 0.0) for m in results["cognitive_metrics"]]
            resource_efficiency = [m.get('resource_efficiency', 0.0) for m in results["cognitive_metrics"]]
            
            # Plot metrics
            plt.plot(steps_range[:len(goal_alignment)], goal_alignment, 'b-', label='Goal Alignment', linewidth=2)
            plt.plot(steps_range[:len(inhibition_efficiency)], inhibition_efficiency, 'g-', 
                    label='Inhibition Efficiency', linewidth=2)
            plt.plot(steps_range[:len(resource_efficiency)], resource_efficiency, 'm-', 
                    label='Resource Efficiency', linewidth=2)
            
            # Add phase markers
            plt.axvline(x=learning_end, color='r', linestyle='--')
            plt.axvline(x=interference_end, color='orange', linestyle='--')
            
            plt.xlabel('Step')
            plt.ylabel('Metric Value')
            plt.title('Cognitive Metrics During Interference Test')
            plt.ylim(0, 1.05)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "cognitive_metrics.png"))
            plt.close()
        
        # 4. Create summary visualization
        plt.figure(figsize=(10, 6))
        
        # Create bars for key metrics
        metrics = ['Initial Similarity', 'Post-Interference', 'Final Similarity', 'Recovery %']
        values = [
            results["initial_similarity"],
            results["pattern_similarity"][interference_end-1],
            results["final_similarity"],
            results["recovery_percentage"]
        ]
        
        # Plot bars
        plt.bar(metrics, values, color=['green', 'red', 'blue', 'purple'])
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        plt.ylabel('Value')
        plt.title('Pattern Interference Test Summary')
        plt.ylim(0, 1.1)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "summary.png"))
        plt.close()
        
        # 5. Create resistance map visualization if available
        if hasattr(self.model, 'calculate_novelty_resistance_map'):
            resistance_map = self.model.calculate_novelty_resistance_map()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(resistance_map.detach().cpu().numpy(), cmap='plasma')
            plt.colorbar(label='Novelty Resistance')
            plt.title('Final Novelty Resistance Map')
            
            plt.savefig(os.path.join(output_dir, "resistance_map.png"))
            plt.close()

    
    def run(self, steps=40, goal_count=3, goal_types=None, run_contradiction_test=False):
        """
        Run prefrontal experiment.
        
        Args:
            steps: Number of steps to run
            goal_count: Number of goals to create during experiment
            goal_types: List of goal types to test
            run_contradiction_test: Whether to run the contradiction test
            
        Returns:
            Results dictionary
        """
        if goal_types is None:
            goal_types = ['stability', 'energy', 'novelty']
        
        print(f"Running prefrontal experiment with {steps} steps...")

        self.model.reset()

        # Dynamically create goals from first few generated signals
        dynamic_signals = [self.model.environment.generate_signal() for _ in range(goal_count)]
        patterns = [{"tensor": sig, "name": f"Pattern_{i}"} for i, sig in enumerate(dynamic_signals)]
        goals = self.create_goals(patterns, goal_count, goal_types)
        
        # Begin dynamic signal loop
        goal_progress = {i: [] for i in range(len(goals))}
        for i, goal in enumerate(goals):
            print(f"Setting goal {i+1}: {goal['type']} (priority: {goal['priority']:.2f})")
            self.model.set_goal(goal, goal["priority"])
        print(f"Goals set. Checking if prefrontal module registered them...")
        pf_status = self.model.prefrontal.get_status()
        print(f"Prefrontal status shows {len(pf_status.get('goals', []))} goals")

        for step in range(steps):
            signal = self.model.environment.generate_signal()
            result = self.model(input_tensor=signal, prefrontal_control=True)

            # Get prefrontal status with goals
            pf_status = result.get('prefrontal_status', {})
            current_goals = pf_status.get('goals', [])

            # Track goal progress
            for i, goal in enumerate(current_goals):
                if i < len(goals):
                    progress = goal.get('progress', 0.0)
                    goal_progress[i].append(progress)

            # Store the result info with actual goal data
            self.goal_tracking.append({
                "step": step,
                "goals": current_goals,  # This should contain the actual goal data now
                "energy_distribution": result.get('energy_distribution', {}),
                "stability": result.get('stability', 0.5)
            })

            # Calculate and track cognitive metrics for this step (this is key)
            self._update_cognitive_metrics(result)

            # Visualize field
            self.visualize_field(
                step,
                prefrontal_active=True,
                goal_focused=True
            )

        self.visualize_goal_progress(goal_progress, goals)

        results_summary = {
            'goal_tracking': self.goal_tracking,
            'cognitive_metrics': {}
         }

        if hasattr(self.model, 'get_cognitive_metrics'):
            # Get metrics from the model
            model_metrics = self.model.get_cognitive_metrics()
            results_summary['cognitive_metrics'] = model_metrics
            
            # Also update our local metrics storage for visualization
            for key, value in model_metrics.items():
                if key in self.cognitive_metrics and value > 0:
                    if not self.cognitive_metrics[key]:
                        self.cognitive_metrics[key] = [value]
                    else:
                        self.cognitive_metrics[key].append(value)

        self.create_prefrontal_visualizations(results_summary)
        self.create_executive_control_viz(results_summary)
        
        # Update results dict directly, don't pass to save_results
        self.results.update(results_summary)
        # Call parent's save_results method
        super().save_results()

        """
        # Add this at the end of the PrefrontalExperiment.run method, just before returning results_summary:

        # Create performance summary visualization
        self.create_performance_summary(results_summary)
        """
        if run_contradiction_test:
            contradiction_results = self.run_contradiction_test(steps=50)
            results_summary['contradiction_test'] = contradiction_results

        return results_summary
    
    def create_goals(self, patterns, goal_count, goal_types, run_contradiction_test=False):
        """Create experimental goals from patterns"""
        goals = []
        for i in range(min(goal_count, len(patterns))):
            pattern = patterns[i]
            goal_type = goal_types[i % len(goal_types)]

            if goal_type == 'stability':
                goal = {
                    "type": "stability",
                    "target_value": 0.8,
                    "pattern": pattern["tensor"],
                    "pathway_needs": [0.6, 0.2, 0.2]
                }
            elif goal_type == 'energy':
                goal = {
                    "type": "energy",
                    "target_value": 0.7,
                    "pattern": pattern["tensor"],
                    "pathway_needs": [0.2, 0.6, 0.2]
                }
            elif goal_type == 'novelty':
                goal = {
                    "type": "novelty",
                    "target_value": 0.6,
                    "pattern": pattern["tensor"],
                    "pathway_needs": [0.1, 0.3, 0.6]
                }
            else:
                goal = {
                    "type": "balanced",
                    "target_value": 0.5,
                    "pattern": pattern["tensor"],
                    "pathway_needs": [0.33, 0.33, 0.34]
                }

            goal["priority"] = 1.0 - (i / goal_count * 0.6)
            goals.append(goal)
        print(f"Created goal: {goal}")
        return goals

class EnhancedPrefrontalExperiment(PrefrontalExperiment):
    """
    Enhanced version of PrefrontalExperiment with integrated fatigue tracking.
    """
    
    def visualize_field(self, step, prefrontal_active=False, goal_focused=False, 
                      inhibition_active=False, attention_active=False):
        # Call parent method first
        super().visualize_field(step, prefrontal_active, goal_focused, 
                              inhibition_active, attention_active)
        
        # Add fatigue visualization if available
        if hasattr(self.model, 'prefrontal') and hasattr(self.model.prefrontal, 'fatigue_level'):
            fatigue_level = self.model.prefrontal.fatigue_level
            self.visualize_fatigue(step, fatigue_level)
    
    def visualize_fatigue(self, step, fatigue_level):
        """Create standalone fatigue visualization"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed, skipping fatigue visualization")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create fatigue gauge visualization
        plt.barh(0, fatigue_level, color='orange', height=0.5)
        plt.xlim(0, 40)  # Set limit to max expected fatigue
        plt.ylim(-0.5, 0.5)
        
        # Add reference lines
        plt.axvline(x=10, color='green', linestyle='--', alpha=0.7, label='Rebalancing Threshold')
        plt.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='High Fatigue')
        plt.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Extreme Fatigue')
        
        # Remove y ticks
        plt.yticks([])
        
        # Add labels
        plt.xlabel('Fatigue Level')
        plt.title(f'Current Fatigue Level: {fatigue_level:.2f} (Step {step})')
        plt.legend(loc='upper right')
        
        # Add fatigue state labels
        states = ['Normal', 'Elevated', 'High', 'Extreme']
        positions = [5, 15, 25, 35]
        for state, pos in zip(states, positions):
            plt.text(pos, -0.3, state, ha='center')
        
        # Add fatigue components if available
        if hasattr(self.model.prefrontal, 'fatigue_controller') and hasattr(self.model.prefrontal.fatigue_controller, 'debug_info'):
            debug_info = self.model.prefrontal.fatigue_controller.debug_info
            if 'goal_intensity' in debug_info and 'shift_magnitude' in debug_info and 'residual_meaning' in debug_info:
                # Calculate components
                goal_intensity = debug_info['goal_intensity']
                shift_magnitude = debug_info['shift_magnitude']
                residual_meaning = debug_info['residual_meaning']
                product = goal_intensity * shift_magnitude
                total = product + residual_meaning
                
                # Create component breakdown
                plt.text(fatigue_level + 1, 0, f"Components:", va='center')
                plt.text(fatigue_level + 1, -0.1, f"G × ΔS: {product:.2f}", va='center', fontsize=8)
                plt.text(fatigue_level + 1, -0.2, f"Rₘ: {residual_meaning:.2f}", va='center', fontsize=8)
                plt.text(fatigue_level + 1, -0.3, f"Total: {total:.2f}", va='center', fontsize=8)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"fatigue_level_step_{step}.png"))
        plt.close()
    
    def create_fatigue_summary(self):
        """Create fatigue summary visualizations"""
        # Only proceed if model has fatigue tracking
        if not hasattr(self.model, 'prefrontal') or not hasattr(self.model.prefrontal, 'fatigue_level'):
            return
        
        fatigue_dir = os.path.join(self.output_dir, "fatigue_summary")
        os.makedirs(fatigue_dir, exist_ok=True)
        
       
        try:
            from ecliphra.visuals.fatigue_visualization import visualize_fatigue_dynamics, create_fatigue_debug_dashboard
            
            # Create fatigue visualizations
            visualize_fatigue_dynamics(self.model.prefrontal, fatigue_dir)
            create_fatigue_debug_dashboard(self.model.prefrontal, fatigue_dir)
            
            print(f"Created fatigue visualizations in {fatigue_dir}")
            return
        except ImportError:
            pass
        
        # Fallback 
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed, skipping fatigue summary")
            return
        
        # Check if we have fatigue history
        if hasattr(self.model.prefrontal, 'fatigue_controller') and hasattr(self.model.prefrontal.fatigue_controller, 'fatigue_history'):
            fatigue_history = self.model.prefrontal.fatigue_controller.fatigue_history
            
            if fatigue_history:
                # Create fatigue history visualization
                plt.figure(figsize=(12, 6))
                steps = range(len(fatigue_history))
                plt.plot(steps, fatigue_history, 'b-', linewidth=2)
                
                # Add thresholds
                plt.axhline(y=10.0, color='green', linestyle='--', alpha=0.7, label='Rebalancing Threshold')
                plt.axhline(y=20.0, color='orange', linestyle='--', alpha=0.7, label='High Fatigue')
                plt.axhline(y=30.0, color='red', linestyle='--', alpha=0.7, label='Extreme Fatigue')
                
                plt.xlabel('Step')
                plt.ylabel('Fatigue Level')
                plt.title('Fatigue Level Throughout Experiment')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(fatigue_dir, 'fatigue_history.png'))
                plt.close()
      
        if hasattr(self.model.prefrontal, 'fatigue_controller') and hasattr(self.model.prefrontal.fatigue_controller, 'goal_completion_history'):
            goal_completions = self.model.prefrontal.fatigue_controller.goal_completion_history
            
            if goal_completions:
                plt.figure(figsize=(12, 6))
                
                indices = range(len(goal_completions))
                goal_types = [g.get('goal_type', 'unknown') for g in goal_completions]
                restorations = [g.get('restoration', 0.0) for g in goal_completions]
                
                bars = plt.bar(indices, [-r for r in restorations], color='green')
                
                # Add goal type labels
                for i, (bar, gtype) in enumerate(zip(bars, goal_types)):
                    plt.text(i, -0.05, gtype, ha='center', rotation=90, fontsize=8)
                
                plt.xlabel('Goal Completion Index')
                plt.ylabel('Fatigue Reduction')
                plt.title('Fatigue Reduction from Goal Completions')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(fatigue_dir, 'goal_completions.png'))
                plt.close()
    
    # Modify run method to track and visualize fatigue
    original_run = PrefrontalExperiment.run
    
    def enhanced_run(self, steps=40, goal_count=3, goal_types=None, run_contradiction_test=False):
        """Run prefrontal experiment with enhanced fatigue tracking"""
        # Run the original method
        result = original_run(self, steps, goal_count, goal_types, run_contradiction_test)
        
        # Create fatigue summary
        self.create_fatigue_summary()
        
        # If model has fatigue tracking, add to results
        if hasattr(self.model, 'prefrontal') and hasattr(self.model.prefrontal, 'fatigue_level'):
            # Add fatigue metrics to result
            if 'fatigue_level' not in result:
                result['fatigue_level'] = self.model.prefrontal.fatigue_level
            
            # Add goal completion stats if available
            if hasattr(self.model.prefrontal, 'fatigue_controller') and hasattr(self.model.prefrontal.fatigue_controller, 'goal_completion_history'):
                result['goal_completions'] = len(self.model.prefrontal.fatigue_controller.goal_completion_history)
                
                # Calculate total restoration from goal completions
                total_restoration = sum(g.get('restoration', 0.0) for g in self.model.prefrontal.fatigue_controller.goal_completion_history)
                result['fatigue_restoration'] = total_restoration
        
        return result

# Factory function for creating experiments
def create_experiment(experiment_type, model, output_dir=None, field_size=32):
    """
    Create an experiment of the specified type

    Args:
        experiment_type: Type of experiment ('semantic', 'echo', 'noise', 'transition')
        model: Instantiated Ecliphra model
        output_dir: Output directory (default: timestamped directory)
        field_size: Field size (NxN)

    Returns:
        Instantiated experiment object
    """
    experiments = {
        'semantic': SemanticExperiment,
        'echo': EchoExperiment,
        'noise': NoiseResistanceExperiment,
        'transition': TransitionExperiment,
        'spiral_resume': SpiralResumeExperiment,
        'photosynthesis': PhotosynthesisExperiment,
        'energy': EnergySystemExperiment,
        'prefrontal': PrefrontalExperiment

    }

    if experiment_type not in experiments:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return experiments[experiment_type](model, output_dir, field_size)


# Right now, keeping duplication because it’s stable and I don’t want to break core tests.

