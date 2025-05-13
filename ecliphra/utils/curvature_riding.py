"""
Curvature Riding Implementation for Ecliphra

This module enhances the Ecliphra system with the ability to "ride" field curvature
rather than fighting against contradictions. This will be more efficient for resource
usage and better handling of complex scenarios.

The implementation adds:
1. Field curvature detection
2. Adaptive response mechanisms
3. Flow-based prefrontal control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ecliphra.utils.enhanced_fatigue_controller import EnhancedFatigueController
from ecliphra.utils.prefrontal import PrefrontalModule


def detect_field_curvature(self, field=None):
    """
    Detect and measure the curvature of the field to identify contradictions.
    
    Args:
        field: Optional field tensor (uses self.field if None)
        
    Returns:
        Dict with curvature magnitude and directional components
    """
    if field is None:
        field = self.field
    
    # Laplacian (∇²F) for field curvature
    laplacian = self.compute_laplacian()
    
    grad_y = torch.zeros_like(field)
    grad_x = torch.zeros_like(field)

    grad_y[:-1, :] = field[1:, :] - field[:-1, :]
    grad_x[:, :-1] = field[:, 1:] - field[:, :-1]
 
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    
    # Just how strong the contradiction is
    curvature_magnitude = torch.mean(torch.abs(laplacian)).item()
    
    # Where the contradiction is pulling at
    flow_direction_x = torch.mean(grad_x).item()
    flow_direction_y = torch.mean(grad_y).item()
    
    principal_direction = math.atan2(flow_direction_y, flow_direction_x)
    
    # Curvature heterogeneity, how complex the contradiction is
    curvature_variance = torch.var(laplacian).item()
    
    # Tension in the field
    strain_energy = torch.mean(grad_magnitude).item()
    
    curvature_magnitude *= 1.5  # Amplify the detected curvature

    return {
        "magnitude": curvature_magnitude,
        "flow_x": flow_direction_x,
        "flow_y": flow_direction_y,
        "principal_direction": principal_direction,
        "variance": curvature_variance,
        "strain_energy": strain_energy,
        "tensor": laplacian.detach().clone()
    }



def calculate_adaptive_response(self, curvature_data, stability, fatigue_level=0.0):
    """
    Finding response parameters based on field curvature.
    
    Args:
        curvature_data: Dict from detect_field_curvature
        stability: Current field stability
        fatigue_level: Current fatigue level 0+, higher just means more fatigue
        
    Returns:
        Dict with adaptive response parameters
    """
    
    base_scaling = 0.5 # can be adjusted
    
    # Less stable means stronger response
    stability_factor = 1.0 / (stability + 0.2)  # Prevent division by zero

    # Enhanced fatigue controller added
    fatigue_factor = self.fatigue_controller.compute_fatigue_factor()

    desired_acceleration = curvature_data["magnitude"] * base_scaling * stability_factor* (1.0 - fatigue_factor * 0.5)
    
    # Lower stability means more alignment with flow
    alignment_factor = max(0.2, min(0.9, (1.0 - stability) + fatigue_factor * 0.3))

     # Adjusted by fatigue (update)
    if fatigue_level > 15.0:
        # When highly fatigued, strongly favor adaptation over maintenance
        maintenance_adj = -curvature_data["magnitude"] * 0.5  # Reducing stability efforts more
        growth_adj = -curvature_data["magnitude"] * 0.1  # Slight reduce 
        adaptation_adj = curvature_data["magnitude"] * 0.6  # Increase 
    else:
        maintenance_adj = -curvature_data["magnitude"] * 0.3  
        growth_adj = 0.0  # Neutral 
        adaptation_adj = curvature_data["magnitude"] * 0.4  
    
    
    # higher curvature gives more selective memory
    memory_retention = max(0.3, 1.0 - curvature_data["magnitude"] - fatigue_factor * 0.2)
    
    # faster updates
    goal_update_rate = 0.1 + curvature_data["magnitude"] * 0.4
    if fatigue_level > 20.0:
        goal_update_rate *= 0.7  # Slow down goal updates when highly fatigued
    
    # lower for high curvature to go with the flow
    inhibition_strength = max(0.1, 0.6 - curvature_data["magnitude"] * 0.5 - fatigue_factor * 0.2)
    
    return {
        "desired_acceleration": desired_acceleration,
        "alignment_factor": alignment_factor,
        "energy_adjustments": {
            "maintenance": maintenance_adj,
            "growth": growth_adj,
            "adaptation": adaptation_adj
        },
        "memory_retention": memory_retention,
        "goal_update_rate": goal_update_rate,
        "inhibition_strength": inhibition_strength,
        "fatigue_factor": fatigue_factor
    }

def apply_curvature_based_control(self, field, energy_distribution, curvature_data, stability, signal_metrics=None):
    """
    Helps to flow with contradictions
    instead of fighting against them.
    
    Args:
        field: Current field tensor
        energy_distribution: Current energy
        curvature_data: Field curvature 
        stability: Current field stability
        signal_metrics: From processing
        
    Returns:
        Tuple of (modified field, modified energy distribution)
    """
    fatigue_level = 0.0
    if hasattr(self, 'fatigue_level'):
        fatigue_level = self.fatigue_level
    response = self.calculate_adaptive_response(curvature_data, stability, fatigue_level)
     # for novelty calculation
    previous_field = field.detach().clone()
    
    # adaptive energy allocation
    if energy_distribution:
        # match the curvature
        for pathway, adjustment in response["energy_adjustments"].items():
            if pathway in energy_distribution:
                energy_distribution[pathway] = max(0.1, 
                    energy_distribution[pathway] + adjustment * response["alignment_factor"])

        pathway_sum = sum(energy_distribution.get(k, 0) for k in ["maintenance", "growth", "adaptation"])
        total_energy = energy_distribution.get("total", pathway_sum)
        
        if pathway_sum > 0:
            for pathway in ["maintenance", "growth", "adaptation"]:
                if pathway in energy_distribution:
                    energy_distribution[pathway] = energy_distribution[pathway] / pathway_sum * total_energy
    
    # When super fatigued, lower total energy
    if fatigue_level > 20.0 and energy_distribution:
        energy_reduction = min(0.3, fatigue_level / 100.0)  # Up to 30% reduction
        if "total" in energy_distribution:
            energy_distribution["total"] *= (1.0 - energy_reduction)
            
            for pathway in ["maintenance", "growth", "adaptation"]:
                if pathway in energy_distribution:
                    energy_distribution[pathway] *= (1.0 - energy_reduction)
    
    fatigue_memory_factor = max(0.0, min(0.5, fatigue_level / 40.0))
    adjusted_retention = response["memory_retention"] * (1.0 - fatigue_memory_factor)
    self.adjust_memory_retention(response["memory_retention"])

    # Lower goal update rate when fatigued...remember it's tired
    update_rate = response["goal_update_rate"]
    if fatigue_level > 15.0:
        # More tired means slower to conserve energy
        update_rate *= max(0.4, 1.0 - fatigue_level / 50.0)
    
    self.update_goals_adaptively(
        response["goal_update_rate"], 
        field=field,
        previous_field=previous_field,
        curvature_data=curvature_data
    )
    
    inhibition_factor = max(0.5, 1.0 - fatigue_level / 30.0)
    self.inhibitory_strength.data = torch.tensor(
        response["inhibition_strength"], device=self.device)
    
    # follow curvature, don't fight it
    modulated_field = self.apply_flow_based_modulation(
        field, curvature_data, response["alignment_factor"])

    # When it's just too much, add energy-conservation field smoothing
    if fatigue_level > 25.0:
        # This reduces energy cost of maintaining complex patterns
        smoothing_factor = min(0.3, (fatigue_level - 25.0) / 50.0)
        smoothed_field = F.avg_pool2d(
            modulated_field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
        modulated_field = (1.0 - smoothing_factor) * modulated_field + smoothing_factor * smoothed_field

    self.last_curvature_response = response
    
    return modulated_field, energy_distribution

def adjust_memory_retention(self, retention_factor):
    """
    Adjust working memory retention based on field curvature.
    
    Args:
        retention_factor: Factor determining how much memory to retain (0-1)
    """
    if not hasattr(self, 'working_memory') or not self.working_memory:
        return
    
    # Sort by importance * recency
    sorted_memory = []
    for item in self.working_memory:
        # New is way more important
        recency = max(0.1, 1.0 - item.get("age", 0) / 10)
        combined_importance = item.get("importance", 0.5) * recency
        sorted_memory.append((item, combined_importance))
    
    sorted_memory.sort(key=lambda x: x[1], reverse=True)
    
    # Only the top items based on retention factor
    keep_count = max(1, int(len(self.working_memory) * retention_factor))
    self.working_memory = [item for item, _ in sorted_memory[:keep_count]]
    
    if hasattr(self, '_update_context_embedding'):
        self._update_context_embedding()

def update_novelty_progress(self, field, previous_field, curvature_data):
    """
    Getting novelty updates from the field changes and curvature.
    
    Returns:
        float: Novelty progress increment (0-1)
    """
    if previous_field is not None:
        field_diff = torch.norm(field - previous_field).item()
        normalized_diff = min(1.0, field_diff * 10)
    else:
        normalized_diff = 0.0
    
    # Higher gives higher novelty potential
    curvature_factor = min(1.0, curvature_data["magnitude"] * 20)
    
    attractor_change = 0.0
    if hasattr(self, 'previous_attractors') and hasattr(self, 'attractors'):
        if len(self.attractors) != len(self.previous_attractors):
            attractor_change = 0.2  # Significant change
        else:
            for i, att in enumerate(self.attractors):
                if i < len(self.previous_attractors):
                    pos1 = att[0]
                    pos2 = self.previous_attractors[i][0]
                    if abs(pos1[0] - pos2[0]) > 2 or abs(pos1[1] - pos2[1]) > 2:
                        attractor_change = 0.1  # Moderate change
                        break
    
    novelty_increment = (normalized_diff * 0.5 + 
                         curvature_factor * 0.3 + 
                         attractor_change * 0.2)

   # Applying decay
    novelty_progress = 0.0
    for goal in self.current_goals:
        if goal.get("type", "") == "novelty":
            novelty_progress = goal.get("progress", 0.0)
            break
    
    # Just a little decay
    decay_factor = 0.0
    if novelty_progress > 0.4:
        decay_factor = (novelty_progress - 0.4) * (1.25)  
        decay_factor = min(decay_factor, 1.0)  
        
    final_increment = novelty_increment * (1.0 - decay_factor * 0.7) + 0.05 * decay_factor

    if hasattr(self, 'attractors'):
        self.previous_attractors = self.attractors.copy()
    
    return novelty_increment

def update_goals_adaptively(self, update_rate, field=None, previous_field=None, curvature_data=None):
    if not hasattr(self, 'current_goals') or not self.current_goals:
        return
    
    goals_updated = [] # get the list
    
    for goal in self.current_goals:
        original_progress = goal.get("progress", 0.0)
        
        progress_increment = update_rate * 0.05
        
        # Special calculation for novelty because it is special
        if goal.get("type", "") == "novelty" and field is not None and curvature_data is not None:
            novelty_increment = self.update_novelty_progress(field, previous_field, curvature_data)
            progress_increment = max(progress_increment, novelty_increment)
        
        goal["progress"] = min(1.0, original_progress + progress_increment)
        
        if goal["progress"] > 0.8:
            # Decrease activation more rapidly for goals close to being done
            goal["activation"] = goal.get("activation", 1.0) * 0.8
        else:
            # For goals that align with the current flow
            activation_delta = update_rate * 0.05
            goal["activation"] = min(1.0, goal.get("activation", 0.5) + activation_delta)
   
        if goal["progress"] > original_progress:
            goals_updated.append(goal.get("type", "unknown"))
    
    # Removing goals
    self.current_goals = [
        goal for goal in self.current_goals
        if goal.get("progress", 0) < 0.95 and goal.get("activation", 0) > 0.1
    ]
  
    if goals_updated and hasattr(self, 'decision_history'):
        self.decision_history.append({
            "type": "adaptive_goal_update",
            "update_rate": update_rate,
            "goals_updated": goals_updated
        })

def implement_consolidation_phase(self, field, energy_distribution, fatigue_level, curvature_data):
    """
    For when fatigue reaches high levels.
    
    This allows the system to integrate "learned" patterns while lowering their energy consumption.
    
    Args:
        field: Current field tensor
        energy_distribution: Current energy distribution
        fatigue_level: Current fatigue level
        curvature_data: Field curvature data
        
    Returns:
        Tuple of (modified field, modified energy distribution, consolidation_active)
    """
    consolidation_triggered = False
    recovery_amount = 0.0  # Default if not triggered

    curvature_mag = curvature_data.get("magnitude", 0.0)
    base_threshold = 25.0
    curvature_influence = curvature_mag * 15.0
    consolidation_threshold = base_threshold - curvature_influence

    if fatigue_level > consolidation_threshold:
        consolidation_triggered = True
        recovery_amount = min(5.0, fatigue_level * 0.1)

        print(f"[CONSOLIDATION] Triggered -> Fatigue: {fatigue_level:.2f}, Curvature: {curvature_mag:.3f}, Recovery: {recovery_amount:.2f}")

        kernel = torch.ones((3, 3), device=self.device) / 9.0
        field_smoothed = F.conv2d(field.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)
        field_smoothed = field_smoothed.squeeze(0).squeeze(0)

        # Shrink the extremes
        field_suppressed = torch.tanh(field_smoothed) * 0.8

        field = field_suppressed.clone()

        modified_distribution = {
            "maintenance": 0.7,
            "growth": 0.15,
            "adaptation": 0.15,
            "total": 1.0
        }

    else:
        modified_distribution = energy_distribution

    return field, modified_distribution, consolidation_triggered, recovery_amount

def track_consolidated_knowledge(self):
    """
    Finding what patterns the system has learned through consolidation.
    
    Returns:
        Dict with analysis of consolidated patterns
    """
    if not hasattr(self, 'consolidation_history'):
        self.consolidation_history = []
    
    field = self.field.detach().clone()
    
    # Find attractors and their relationships
    attractor_relationships = {}
    if hasattr(self, 'attractors') and len(self.attractors) >= 2:
        # Get top two 
        top_attractors = sorted(self.attractors, key=lambda x: x[1], reverse=True)[:2]
        
        # Finding distance between them
        pos1, pos2 = top_attractors[0][0], top_attractors[1][0]
        dx = pos2[1] - pos1[1]
        dy = pos2[0] - pos1[0]
        distance = torch.sqrt(torch.tensor(float(dx**2 + dy**2)))
        angle = torch.atan2(torch.tensor(float(dy)), torch.tensor(float(dx)))
        
        attractor_relationships = {
            "distance": distance.item(),
            "angle": angle.item(),
            "strength_ratio": top_attractors[0][1] / (top_attractors[1][1] + 1e-10)
        }
    
    # Extract frequency components
    fft = torch.fft.rfft2(field)
    fft_mag = torch.abs(fft)
    
    # Getting dominant frequency characteristics
    total_energy = torch.sum(fft_mag)
    low_freq = torch.sum(fft_mag[:field.shape[0]//4, :field.shape[1]//4]) / total_energy
    high_freq = torch.sum(fft_mag[field.shape[0]//4:, field.shape[1]//4:]) / total_energy
    
    # Structure
    field_variance = torch.var(field).item()
    field_entropy = -torch.sum(torch.abs(field) * torch.log(torch.abs(field) + 1e-10)).item()
    
    grad_y, grad_x = torch.gradient(field)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    avg_gradient = torch.mean(grad_magnitude).item()
    
    knowledge_summary = {
        "attractor_relationships": attractor_relationships,
        "frequency_characteristics": {
            "low_freq_ratio": low_freq.item(),
            "high_freq_ratio": high_freq.item()
        },
        "field_structure": {
            "variance": field_variance,
            "entropy": field_entropy,
            "avg_gradient": avg_gradient
        },
        "timestamp": self.time_step.item() if hasattr(self, 'time_step') else 0
    }
    
    self.consolidation_history.append(knowledge_summary)
    
    return knowledge_summary

def apply_field_persistence(self, current_field, previous_field, persistence_strength=0.05):
    """
    Apply inertial memory to field updates because fields tend to maintain 
    some of their previous state during updates.
    
    Args:
        current_field: Current field tensor
        previous_field: Previous field tensor
        persistence_strength: How strongly previous state influences current (0-1)
        
    Returns:
        Modified field with persistence applied
    """
    if hasattr(self, 'field_history') and len(self.field_history) > 0:
        # More heavy
        weights = torch.tensor([0.8**i for i in range(len(self.field_history))], 
                              device=self.device)
        weights = weights / weights.sum()
        
        historical_field = torch.zeros_like(current_field)
        for i, hist_field in enumerate(reversed(self.field_history[-10:])):
            if i < len(weights):
                historical_field += hist_field * weights[i]
    else:
        historical_field = previous_field
    
    persistent_field = (1 - persistence_strength) * current_field + persistence_strength * historical_field
    
    persistence_effect = torch.norm(persistent_field - current_field).item()
    
    return persistent_field, persistence_effect

def calculate_novelty_resistance_map(self):
    """
    Create a map showing areas of the field that should resist novelty
    based on established patterns.
    
    Returns:
        Tensor with values 0-1 where higher values indicate more resistance to novelty
    """
    resistance_map = torch.zeros(self.field_dim, device=self.device)

    if not hasattr(self, 'field_history') or len(self.field_history) < 5:
        return resistance_map
    
    recent_history = self.field_history[-10:]
    if len(recent_history) < 2:
        return resistance_map
    
    stacked_history = torch.stack(recent_history)
    point_variance = torch.var(stacked_history, dim=0)
    
    # Low variance = high consistency = high resistance to novelty
    max_variance = torch.max(point_variance).item()
    if max_variance > 0:
        consistency = 1.0 - (point_variance / max_variance)
        
        # Need non-linear scaling, higher consistency creates way higher resistance
        resistance_map = torch.pow(consistency, 3)
    
    return resistance_map

def apply_novelty_resistance(self, base_field, novelty_field, resistance_map=None):
    """
    To areas with established patterns resist change.
    
    Args:
        base_field: Original field tensor
        novelty_field: New field tensor introducing novelty
        resistance_map: Optional pre-calculated resistance map
        
    Returns:
        Modified field with novelty resistance applied
    """
    if resistance_map is None:
        resistance_map = self.calculate_novelty_resistance_map()
    
    # Change being introduced
    field_delta = novelty_field - base_field
    
    resistant_delta = field_delta * (1.0 - resistance_map)
    
    # New field, Resistant changes
    resistant_field = base_field + resistant_delta
    
    return resistant_field

def analyze_energy_history(self):
    """
    Checking past energy distributions to find successful patterns.
    
    Returns:
        Dict with energy pattern analysis
    """
    if not hasattr(self, 'energy_history') or len(self.energy_history) < 5:
        return {"pattern_detected": False}
    
    recent_history = self.energy_history[-20:]
    
    efficiency_scores = [] #  how much stability per unit energy
    for entry in recent_history:
        if 'stability' in entry and 'energy_level' in entry and entry['energy_level'] > 0:
            efficiency = entry['stability'] / entry['energy_level']
            efficiency_scores.append(efficiency)
    
    # Find the best distribution patterns
    if efficiency_scores:
        max_index = efficiency_scores.index(max(efficiency_scores))
        optimal_distribution = recent_history[max_index].get('energy_distribution', {})
        
        return {
            "pattern_detected": True,
            "optimal_distribution": optimal_distribution,
            "max_efficiency": max(efficiency_scores),
            "avg_efficiency": sum(efficiency_scores) / len(efficiency_scores)
        }
    
    return {"pattern_detected": False}

def apply_contextual_energy_favoring(self, field, energy_distribution):
    """
    Adjust energy distribution to favor past successful patterns.
    
    Args:
        field: Current field tensor
        energy_distribution: Current energy distribution
        
    Returns:
        Modified energy distribution with historical bias
    """
    analysis = self.analyze_energy_history()
    
    if not analysis.get("pattern_detected", False):
        return energy_distribution
    
    optimal = analysis.get("optimal_distribution", {})
    
    # Just how similar is current field to past efficient fields
    similarity_score = 0.7  # moderate similarity
    
    if hasattr(self, 'field_history') and len(self.field_history) > 0:
        historical_fields = self.field_history[-10:]
        
        max_similarity = 0.0
        for hist_field in historical_fields:
            sim = F.cosine_similarity(
                field.view(1, -1),
                hist_field.view(1, -1)
            ).item()
            max_similarity = max(max_similarity, sim)
        
        similarity_score = max_similarity
    
    # Adjust current toward optimal based on similarity
    blend_factor = 0.2 * similarity_score  # Higher similarity is stronger favoring
    
    if 'maintenance' in energy_distribution and 'maintenance' in optimal:
        energy_distribution['maintenance'] = ((1 - blend_factor) * energy_distribution['maintenance'] +
                                           blend_factor * optimal['maintenance'])
    
    if 'growth' in energy_distribution and 'growth' in optimal:
        energy_distribution['growth'] = ((1 - blend_factor) * energy_distribution['growth'] +
                                      blend_factor * optimal['growth'])
    
    if 'adaptation' in energy_distribution and 'adaptation' in optimal:
        energy_distribution['adaptation'] = ((1 - blend_factor) * energy_distribution['adaptation'] +
                                          blend_factor * optimal['adaptation'])
    
    return energy_distribution

def apply_flow_based_modulation(self, field, curvature_data, alignment_factor):
    """
    Go with the flow rather than against contradictions.
    
    Args:
        field: Current field tensor
        curvature_data: Dict from detect_field_curvature
        alignment_factor: How much to align with field flow (0-1)
        
    Returns:
        Modified field tensor
    """

    modified_field = field.clone()

    flow_x = curvature_data["flow_x"]
    flow_y = curvature_data["flow_y"]
    flow_magnitude = math.sqrt(flow_x**2 + flow_y**2 + 1e-10)
    
    if flow_magnitude < 1e-6:
        return modified_field  # No significant flow
    
    flow_x /= flow_magnitude
    flow_y /= flow_magnitude
    
    # Need flow vector for efficient computation
    flow_vector = torch.tensor([flow_y, flow_x], device=self.device)
    
    h, w = field.shape
    y_coords = torch.linspace(-1, 1, h, device=self.device).view(-1, 1).expand(-1, w)
    x_coords = torch.linspace(-1, 1, w, device=self.device).view(1, -1).expand(h, -1)
    
    # Dot product with flow direction
    position_vectors = torch.stack([y_coords, x_coords], dim=2)
    alignment = torch.matmul(position_vectors, flow_vector)
    
    modulation = alignment * alignment_factor * 0.1 # smooth falloff
    modified_field = modified_field + modulation
    
    modified_field = F.normalize(modified_field.view(-1), p=2, dim=0).view(h, w)
    
    return modified_field

def curvature_aware_forward(self, input_tensor=None, input_fingerprint=None, 
                            input_pos=None, goal_data=None, prefrontal_control=True):
    """Enhanced forward method with curvature riding and fatigue awareness."""
    
    if goal_data and hasattr(self, 'prefrontal'):
        priority = goal_data.get('priority', 1.0) if isinstance(goal_data, dict) else 1.0
        self.prefrontal.set_goal(goal_data, priority)

    try:
        result = super(type(self), self).forward(input_tensor, input_fingerprint, input_pos)
    except Exception as e:
        print("[WARNING] Forward pass failed:", e)
        result = self.last_valid_result.copy() if hasattr(self, 'last_valid_result') else {
            'energy_distribution': {},
            'stability': 0.5,
            'fatigue_level': 0.0,
            'cognitive_metrics': {},
        }

    else:
        self.last_valid_result = result.copy()

    if prefrontal_control and hasattr(self, 'prefrontal'):
        energy_distribution = result.get('energy_distribution', {})
        signal_metrics = result.get('signal_metrics', {})
        fatigue_level = getattr(self.prefrontal, 'fatigue_level', 0.0)
        result['fatigue_level'] = fatigue_level

        curvature_data = self.detect_field_curvature()
        result['curvature_data'] = curvature_data
        stability = result.get('stability', 0.5)
        previous_field = self.field.detach().clone()

        # Consolidation time. Let method determine threshold logic
        modulated_field, modified_distribution, consolidation_active, consolidation_recovery = (
            self.implement_consolidation_phase(
                self.field, energy_distribution, fatigue_level, curvature_data)
        )

        if consolidation_active:
            result['consolidation_active'] = True
            result['consolidation_recovery'] = consolidation_recovery

            if hasattr(self.prefrontal, 'fatigue_level'):
                self.prefrontal.fatigue_level = max(0.0, self.prefrontal.fatigue_level - consolidation_recovery)
                print(f"[CONSOLIDATION] Fatigue reduced by {consolidation_recovery:.2f} → New level: {self.prefrontal.fatigue_level:.2f}")
        else:
            if curvature_data["magnitude"] > 0.015:
                modulated_field, modified_distribution = self.prefrontal.apply_curvature_based_control(
                    self.field, energy_distribution, curvature_data, stability, signal_metrics)
                result['curvature_riding_active'] = True
            else:
                modulated_field, modified_distribution = self.prefrontal(
                    self.field, energy_distribution, signal_metrics)
                result['curvature_riding_active'] = False

        # Field persistence and novelty resistance
        persistence_strength = min(0.2, 0.05 + (fatigue_level / 100.0)) if fatigue_level > 10.0 else 0.05
        if hasattr(self, 'apply_field_persistence'):
            modulated_field, persistence_effect = self.apply_field_persistence(
                modulated_field, previous_field, persistence_strength)
            result['persistence_strength'] = persistence_strength

        if hasattr(self, 'apply_novelty_resistance'):
            resistance_factor = 1.0 + min(0.5, (fatigue_level - 15.0) / 30.0) if fatigue_level > 15.0 else 1.0
            resistance_map = self.calculate_novelty_resistance_map()
            enhanced_map = torch.pow(resistance_map, 1.0 / resistance_factor)
            modulated_field = self.apply_novelty_resistance(previous_field, modulated_field, enhanced_map)
            result['novelty_resistance_factor'] = resistance_factor

        if hasattr(self, 'apply_contextual_energy_favoring'):
            favoring_strength = min(0.4, 0.2 + (fatigue_level - 20.0) / 50.0) if fatigue_level > 20.0 else 0.2
            modified_distribution = self.apply_contextual_energy_favoring(modified_distribution, favoring_strength)
            result['energy_favoring_strength'] = favoring_strength

        # Working memory update
        if input_tensor is not None:
            if fatigue_level > 25.0:
                importance_threshold = 0.3 + min(0.4, (fatigue_level - 25.0) / 50.0)
                if hasattr(input_tensor, 'norm'):
                    importance = min(1.0, input_tensor.norm().item() / 2.0)
                    if importance > importance_threshold:
                        self.prefrontal.update_working_memory(input_tensor, importance)
                else:
                    self.prefrontal.update_working_memory(input_tensor)
            else:
                self.prefrontal.update_working_memory(input_tensor)

        result['prefrontal_status'] = self.prefrontal.get_status()
        if 'cognitive_metrics' in result['prefrontal_status']:
            result['prefrontal_status']['cognitive_metrics']['curvature_magnitude'] = curvature_data["magnitude"]
        result['prefrontal_status']['fatigue_level'] = self.prefrontal.fatigue_level

        outcome_metrics = {
            "stability": result.get('stability', 0.5),
            "energy_level": result.get('energy_level', 0.0),
            "goal_progress": 0.05,
            "curvature_magnitude": curvature_data["magnitude"],
            "fatigue_level": fatigue_level
        }
        self.prefrontal.evaluate_outcome(outcome_metrics)

        if hasattr(self, '_update_cognitive_metrics'):
            self._update_cognitive_metrics(result)


        # Final field update
        self.field.data = modulated_field
        result['energy_distribution'] = modified_distribution
        
        # Force proper energy visualization
        if hasattr(self, 'energy_current'):
            # field-aligned energy distribution
            energy_current = torch.zeros_like(self.field)
                
            # Base energy on field strength but with smooth gradient
            h, w = self.field.shape
                
            y_indices = torch.arange(h, device=self.device).float()
            x_indices = torch.arange(w, device=self.device).float()
            y_grid = y_indices.view(-1, 1).repeat(1, w)
            x_grid = x_indices.view(1, -1).repeat(h, 1)
                
            center_y, center_x = h/2, w/2
            dist_from_center = torch.sqrt(((y_grid - center_y)/h)**2 + ((x_grid - center_x)/w)**2)
                
            # Base distribution on field value with smooth falloff
            energy_current = torch.abs(self.field) + (1.0 - dist_from_center) * 0.3

            if hasattr(self, 'calculate_novelty_resistance_map'):
                resistance_map = self.calculate_novelty_resistance_map()
                    
                # Create visualization for resistance. Don't worry it's different color
                plt.figure(figsize=(8, 8))
                plt.imshow(resistance_map.detach().cpu().numpy(), cmap='plasma')
                plt.colorbar(label='Novelty Resistance')
                plt.title('Novelty Resistance Map')
                plt.savefig(os.path.join(self.output_dir, "resistance_map.png"))
                plt.close()
                
            # Apply energy pathway influences
            if 'energy_distribution' in result:
                maintenance = result['energy_distribution'].get('maintenance', 0.2)
                growth = result['energy_distribution'].get('growth', 0.2)
                adaptation = result['energy_distribution'].get('adaptation', 0.2)
                    
                center_mask = torch.zeros_like(dist_from_center)
                growth_mask = torch.zeros_like(dist_from_center)
                adapt_mask = torch.zeros_like(dist_from_center)
                    
                center_mask[dist_from_center < 0.3] = 1.0
                growth_mask[(dist_from_center >= 0.3) & (dist_from_center < 0.6)] = 1.0
                adapt_mask[dist_from_center >= 0.6] = 1.0
                    
                energy_current += center_mask * maintenance * 0.5
                energy_current += growth_mask * growth * 0.5
                energy_current += adapt_mask * adaptation * 0.5
                
            energy_min = torch.min(energy_current)
            energy_max = torch.max(energy_current)
            if energy_max > energy_min:
                energy_current = (energy_current - energy_min) / (energy_max - energy_min)
                
            self.energy_current = energy_current
        
    return result



 

# Integration with EcliphraFieldWithPrefrontal class

def integrate_curvature_riding(model):
    """
    Integrate curvature riding into an existing Ecliphra model.
    
    Args:
        model: An instance of EcliphraFieldWithPrefrontal
        
    Returns:
        The modified model with curvature riding capabilities
    """
    model.detect_field_curvature = detect_field_curvature.__get__(model)
    
    if hasattr(model, 'prefrontal'):
        model.prefrontal.calculate_adaptive_response = calculate_adaptive_response.__get__(model.prefrontal)
        model.prefrontal.apply_curvature_based_control = apply_curvature_based_control.__get__(model.prefrontal)
        model.prefrontal.adjust_memory_retention = adjust_memory_retention.__get__(model.prefrontal)
        model.prefrontal.update_goals_adaptively = update_goals_adaptively.__get__(model.prefrontal)
        model.prefrontal.apply_flow_based_modulation = apply_flow_based_modulation.__get__(model.prefrontal)
        
        model.prefrontal.last_curvature_response = None

    model.implement_consolidation_phase = implement_consolidation_phase.__get__(model)
    model.track_consolidated_knowledge = track_consolidated_knowledge.__get__(model)
    
    # use curvature-based control
    original_forward = model.forward
    return model
    
    

from types import MethodType
 
    # Use these functions when integrating curvature riding
def integrate_curvature_riding_with_fatigue(model):
    """
    Integrate curvature riding capabilities with fatigue awareness into an existing Ecliphra model.
    
    Args:
        model: An instance of EcliphraFieldWithPrefrontal
        
    Returns:
        The modified model with curvature riding and fatigue capabilities
    """
    base_model = integrate_curvature_riding(model)

    if base_model is None:
        raise ValueError("Model was None after integrate_curvature_riding")

    # Proceed with fatigue-aware updates
    if hasattr(base_model, 'prefrontal'):
        base_model.prefrontal.calculate_adaptive_response = calculate_adaptive_response.__get__(base_model.prefrontal)
        base_model.prefrontal.apply_curvature_based_control = apply_curvature_based_control.__get__(base_model.prefrontal)

    base_model.implement_consolidation_phase = implement_consolidation_phase.__get__(base_model)
    base_model.forward = curvature_aware_forward.__get__(base_model)

    return base_model
    

# Usage example:
'''
# Initialize your model
model = EcliphraFieldWithPrefrontal(
    field_dim=(32, 32),
    device='cpu',
    memory_capacity=5,
    working_memory_size=5,
    max_goals=3
)

# Integrate curvature riding capabilities
model = integrate_curvature_riding(model)

# Now use the model with enhanced capabilities
result = model(input_tensor=some_input, prefrontal_control=True)

# Check if curvature riding was active
if result.get('curvature_riding_active', False):
    print(f"Curvature riding active with magnitude: {result['curvature_data']['magnitude']:.4f}")
'''
