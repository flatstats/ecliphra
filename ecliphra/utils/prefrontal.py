import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import time

class PrefrontalModule(nn.Module):
    """
    Simulates basic prefrontal behaviors like memory retention, goal selection, and inhibition.
    It’s the executive center of Ecliphra, even if it doesn’t know that yet.
    """
    def __init__(self, field_dim=(32, 32), device='cpu', 
                 working_memory_size=5, max_goals=3, fatigue_controller=None):
        super().__init__()
        self.device = device
        self.field_dim = field_dim
        self.fatigue_controller = fatigue_controller if fatigue_controller is not None else FatigueAwareController()

        
        self.working_memory_size = working_memory_size
        self.working_memory = []  # tensors/patterns
        self.context_embedding = torch.zeros(64, device=device)  # context representation
        self.last_context_embedding = self.context_embedding.clone()
        
        self.max_goals = max_goals
        self.current_goals = []  # goal, priority, progress, activation
        self.goal_embeddings = nn.Parameter(torch.randn(max_goals, 64, device=device))
        
        self.inhibitory_strength = nn.Parameter(torch.tensor(0.6, device=device))
        self.inhibition_targets = set()
        self.inhibition_field = torch.zeros(field_dim, device=device)

        self.attention_mask = nn.Parameter(torch.ones(field_dim, device=device))
        self.attention_bias = nn.Parameter(torch.zeros(3, device=device))  # For 3 pathways
        
        self.resource_weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.3], device=device))  # Initial pathway weights
        self.priority_sensitivity = nn.Parameter(torch.tensor(0.7, device=device))
        
        self.learning_rate = nn.Parameter(torch.tensor(0.05, device=device))
        self.exploration_rate = nn.Parameter(torch.tensor(0.2, device=device))
        self.decay_rate = nn.Parameter(torch.tensor(0.95, device=device))

        self.debug_mode = True 

        
        self.decision_history = []
        self.outcome_history = []
        
        # Integration with energy system
        self.pathway_adjustments = torch.zeros(3, device=device)  # For maintenance, growth, adaptation
        
        self.cognitive_metrics = {
            "goal_alignment": 0.0,
            "inhibition_efficiency": 0.0,
            "resource_efficiency": 0.0
        }
        
        self._initialize_cognitive_metrics() # with small non-zero values for visuals
    
    def _initialize_cognitive_metrics(self):
        self.cognitive_metrics["goal_alignment"] = 0.1
        self.cognitive_metrics["inhibition_efficiency"] = 0.1
        self.cognitive_metrics["resource_efficiency"] = 0.1

    def integrate_enhanced_fatigue(self):

        fatigue_controller = EnhancedFatigueController(
            novelty_threshold=0.2,
            drift_threshold=0.5,
            fatigue_steps=5,  
            recovery_factor=0.4,
            decay_rate=0.98
        )
        
        fatigue_level = 0.0
    
    def set_goal(self, goal_data, priority=1.0):
        """
        Set a new goal with priority
        
        Args:
            goal_data: Goal representations can be tensor, pattern, or structured data
            priority: Importance level (0.0-1.0)
            
        Returns:
            Goal ID (index)
        """
        if not isinstance(goal_data, torch.Tensor):
            if isinstance(goal_data, dict):
                # For structured goals an embedding from components is needed
                goal_embedding = self._encode_structured_goal(goal_data)
            else:
                # All other types take a simple embedding
                goal_embedding = torch.randn(64, device=self.device)
        else:
            goal_embedding = goal_data
            
        goal_entry = {
            "embedding": goal_embedding,
            "data": goal_data,
            "priority": priority,
            "progress": 0.0,
            "activation": 1.0,
            "creation_time": len(self.decision_history),
            "last_active": len(self.decision_history),
            "type": goal_data.get("type", "generic") if isinstance(goal_data, dict) else "generic"
        }
        
        print(f"Setting goal: {goal_entry['type']}, priority: {priority}")
        
        self.current_goals.append(goal_entry)
        
        # Keep only top max_goals by priority
        if len(self.current_goals) > self.max_goals:
            self.current_goals.sort(key=lambda x: x["priority"], reverse=True)
            self.current_goals = self.current_goals[:self.max_goals]
            
        self._update_goal_embeddings()
        
        return len(self.current_goals) - 1  # Return goal ID
    
    def _encode_structured_goal(self, goal_data):
        # Simple implementation, might be expanded based on goal data structure
        embedding = torch.zeros(64, device=self.device)
        
        # This if for components based on available keys
        if "target_pattern" in goal_data and isinstance(goal_data["target_pattern"], torch.Tensor):
            # Making a pattern to embedding space
            pattern_proj = F.adaptive_avg_pool2d(
                goal_data["target_pattern"].unsqueeze(0).unsqueeze(0), (8, 8)
            ).view(-1)
            embedding[:pattern_proj.shape[0]] = pattern_proj
            
        if "target_value" in goal_data:
            value_idx = min(int(goal_data["target_value"] * 10), 9)
            embedding[50 + value_idx] = 1.0
            
        if "type" in goal_data:
            type_mapping = {"stability": 60, "energy": 61, "novelty": 62, "balanced": 63}
            type_idx = type_mapping.get(goal_data["type"], 63)
            embedding[type_idx] = 1.0
        
   
        embedding = F.normalize(embedding, p=2, dim=0)
        return embedding
    
    def _update_goal_embeddings(self):
        # based on current goals
        with torch.no_grad():
            for i, goal in enumerate(self.current_goals):
                if i < self.max_goals:
                    self.goal_embeddings[i] = goal["embedding"]
    
    def update_goal_progress(self, goal_id, progress_delta):
        # towards a specifc goal
        if 0 <= goal_id < len(self.current_goals):
            self.current_goals[goal_id]["progress"] += progress_delta
            
            # Cap progress at 100%
            self.current_goals[goal_id]["progress"] = min(1.0, max(0.0, self.current_goals[goal_id]["progress"]))
            self.current_goals[goal_id]["last_active"] = len(self.decision_history)

    def calculate_goal_specific_progress(self, goal):
        # specfic goals based on their type
        goal_type = goal.get("type", "generic")
        
        if goal_type == "stability":
            # measure field stability
            return min(1.0, self.get_current_stability())
        elif goal_type == "energy":
            # measure energy level
            return min(1.0, self.get_current_energy_level())
        elif goal_type == "novelty":
            # measure adaptation 
            return min(1.0, self.get_current_adaptation())
        else:

            return 0.5
    
    def inhibit_process(self, process_id=None, field_region=None):
        """
        Args:
            process_id: ID of process to inhibit, this could be noise
            field_region: Tuple of y1, y2, x1, x2 will define the inhibit region
        """
        if process_id is not None:
            self.inhibition_targets.add(process_id)
            
        if field_region is not None:
            y1, y2, x1, x2 = field_region
            
            # specified region
            with torch.no_grad():
                self.inhibition_field[y1:y2, x1:x2] = self.inhibitory_strength
    
    def focus_attention(self, field_region=None, pathway_bias=None):
        """
        Focus attention on a specific field region or pathway
        
        Args:
            field_region: Tuple of (y1, y2, x1, x2) defining region to focus on
            pathway_bias: Tensor of 3 values for biasing the pathways
        """
        with torch.no_grad():
            self.attention_mask.fill_(0.2)  # Baseline 
            
            # focused attention if specified
            if field_region is not None:
                y1, y2, x1, x2 = field_region
                self.attention_mask[y1:y2, x1:x2] = 1.0
            
            if pathway_bias is not None:
                self.attention_bias.copy_(torch.tensor(pathway_bias, device=self.device))
    
    def update_working_memory(self, new_item, importance=1.0):
        """
        Update working memory with new information

        Args:
            new_item: New item to add to memory (pattern tensor or structured data)
            importance: Importance of this item (affects retention)
        """
        if not isinstance(new_item, torch.Tensor):
            if isinstance(new_item, dict) and "pattern" in new_item:
                memory_tensor = new_item["pattern"].detach().clone()
            else:
                memory_tensor = torch.zeros(self.field_dim, device=self.device)
        else:
            memory_tensor = new_item.detach().clone()

        self.working_memory.append({
            "content": memory_tensor,
            "importance": importance,
            "age": 0,
            "raw_data": new_item
        })

        if len(self.working_memory) > self.working_memory_size:
            self.working_memory.sort(key=lambda x: x["importance"] * (0.9 ** x["age"]), reverse=True)
            self.working_memory = self.working_memory[:self.working_memory_size]

        self._update_context_embedding()

    def _update_context_embedding(self):
        if not self.working_memory:
            self.context_embedding = torch.zeros(64, device=self.device)
            return

        embeddings = []
        importance_weights = []

        for item in self.working_memory:
            if isinstance(item["content"], torch.Tensor):
                if item["content"].dim() >= 2:
                    projection = F.adaptive_avg_pool2d(
                        item["content"].unsqueeze(0).unsqueeze(0), (8, 8)
                    ).view(-1)[:64]

                    if projection.shape[0] < 64:
                        padding = torch.zeros(64 - projection.shape[0], device=self.device)
                        projection = torch.cat([projection, padding])

                    embeddings.append(projection)
                    importance_weights.append(item["importance"] * (0.9 ** item["age"]))

        if embeddings:
            embeddings_tensor = torch.stack(embeddings)
            weights = torch.tensor(importance_weights, device=self.device)
            weights = weights / weights.sum()

            self.last_context_embedding = self.context_embedding.clone()
            self.context_embedding = torch.sum(embeddings_tensor * weights.unsqueeze(1), dim=0)
            self.context_embedding = F.normalize(self.context_embedding, p=2, dim=0)

    def get_working_memory_score(self):
        if not self.working_memory:
            return 0.0
        score = sum(item["importance"] * (0.9 ** item["age"]) for item in self.working_memory)
        return min(score / self.working_memory_size, 1.0)

    def get_context_drift(self):
        return torch.norm(self.context_embedding - self.last_context_embedding, p=2).item()

    
    def allocate_resources(self, available_energy, energy_distribution=None, signal_metrics=None):
        """
        Modify resource allocation based on goals and executive control
        
        Args:
            available_energy: Total available energy
            energy_distribution: Current energy distribution
            signal_metrics: Signal metrics (coherence, intensity, complexity)
            
        Returns:
            Modified energy distribution dict
        """
        if signal_metrics is None:
             signal_metrics = {}
        signal_metrics["fatigue_level"] = getattr(self, "fatigue_level", 0.0)

        # Start with incoming
        if energy_distribution is None:
            # based on internal weights
            distribution = {
                "maintenance": available_energy * self.resource_weights[0].item(),
                "growth": available_energy * self.resource_weights[1].item(),
                "adaptation": available_energy * self.resource_weights[2].item(),
                "total": available_energy
            }
        else:
            distribution = energy_distribution.copy()
        
        pathway_adjustments = self._calculate_pathway_adjustments(signal_metrics)
        
        for target in self.inhibition_targets:
            if target in distribution:
                distribution[target] *= (1.0 - self.inhibitory_strength.item())

        distribution["maintenance"] += distribution["total"] * self.attention_bias[0].item() * 0.1
        distribution["growth"] += distribution["total"] * self.attention_bias[1].item() * 0.1
        distribution["adaptation"] += distribution["total"] * self.attention_bias[2].item() * 0.1

        distribution["maintenance"] += pathway_adjustments[0].item() * distribution["total"] * 0.2
        distribution["growth"] += pathway_adjustments[1].item() * distribution["total"] * 0.2
        distribution["adaptation"] += pathway_adjustments[2].item() * distribution["total"] * 0.2
        
        for key in ["maintenance", "growth", "adaptation"]:
            distribution[key] = max(0.0, distribution[key]) # Make sure there is no negative values
        
        self.decision_history.append({
            "original_distribution": energy_distribution,
            "modified_distribution": distribution,
            "pathway_adjustments": pathway_adjustments.detach().cpu().numpy().tolist(),
            "attention_bias": self.attention_bias.detach().cpu().numpy().tolist(),
            "active_goals": len(self.current_goals)
        })
        
        for item in self.working_memory:  # Age working memory
            item["age"] += 1

        # Added logging for introspection
        distribution["wm_score"] = self.get_working_memory_score()
        distribution["context_drift"] = self.get_context_drift()

        return distribution
    
    def _calculate_pathway_adjustments(self, signal_metrics=None):
        # based on goals and context
        adjustments = torch.zeros(3, device=self.device)
        
        if self.current_goals:
            for goal in self.current_goals:
                if isinstance(goal["data"], dict) and "pathway_needs" in goal["data"]:
                    # Explicit pathway needs
                    pathway_needs = torch.tensor(goal["data"]["pathway_needs"], device=self.device)
                else:
                    # might make this more sophisticated, okay I definitely will who am I kidding
                    goal_type = goal.get("type", "")
                    if goal_type == "stability":
                        pathway_needs = torch.tensor([0.6, 0.2, 0.2], device=self.device)
                    elif goal_type == "energy":
                        pathway_needs = torch.tensor([0.2, 0.6, 0.2], device=self.device)
                    elif goal_type == "novelty":
                        pathway_needs = torch.tensor([0.1, 0.3, 0.6], device=self.device)
                    else:
                        pathway_needs = torch.tensor([0.33, 0.33, 0.34], device=self.device)
                
                goal_weight = goal["priority"] * goal["activation"] * (1.0 - goal["progress"])
                adjustments += pathway_needs * goal_weight
        
        if signal_metrics:
            coherence = signal_metrics.get("coherence", 0.5)
            intensity = signal_metrics.get("intensity", 0.5)
            complexity = signal_metrics.get("complexity", 0.5)
            
            metric_adjustment = torch.tensor(
                [coherence, intensity, complexity],
                device=self.device
            )
            # To sum to zero, it's purely an adjustment
            metric_adjustment = metric_adjustment - metric_adjustment.mean()
            
            adjustments = adjustments + metric_adjustment * 0.3
        
        if adjustments.abs().sum() > 0:
            adjustments = adjustments - adjustments.mean()
        
        return adjustments
    
    def evaluate_outcome(self, outcome_metrics):
        """
        Learn from outcomes to adjust future resource allocation
        
        Args:
            outcome_metrics: Dict with outcome evaluations
        """
        self.outcome_history.append(outcome_metrics)
        
        if len(self.outcome_history) < 2:
            return
        
        prev_decision = self.decision_history[-2] if len(self.decision_history) >= 2 else None
        
        if prev_decision:
            # Based on the outcome...
            stability = outcome_metrics.get("stability", 0.5)
            field_energy = outcome_metrics.get("energy_level", 0.5)
            goal_progress = outcome_metrics.get("goal_progress", 0.0)
            
            # Overall outcome score needs upgrading in the future
            outcome_score = stability * 0.3 + field_energy * 0.3 + goal_progress * 0.4
            
            # Reinforce those decisions when positive
            if outcome_score > 0.6:
                pathway_adjustments = torch.tensor(prev_decision["pathway_adjustments"], device=self.device)
                self.pathway_adjustments = (
                    self.pathway_adjustments * (1 - self.learning_rate) + 
                    pathway_adjustments * self.learning_rate
                )
                
                # Needed to reduce exploration rate
                self.exploration_rate.data *= 0.95
            
            # If negative outcome though...try different way
            elif outcome_score < 0.4:
                # Now gotta increase exploration 
                self.exploration_rate.data *= 1.05
                self.exploration_rate.data = torch.clamp(self.exploration_rate.data, 0.05, 0.5)
                
                # Decay previous 
                self.pathway_adjustments *= 0.8
                
        self._update_goals(outcome_metrics)
        
        self._update_cognitive_metrics(outcome_metrics)

    def update_novelty_progress(self, field, previous_field, curvature_data):
        """
        Calculate novelty progress based on field changes and curvature.
        
        Returns:
            float: Novelty progress is 0-1
        """
        field_change = 0.0
        curvature_factor = 0.0
        attractor_change = 0.0
        
        if previous_field is not None:
            field_diff = torch.norm(field - previous_field).item()
            # making it a reasonable range
            field_change = min(1.0, field_diff * 10)
        
        # higher curvature means higher novelty potential
        curvature_factor = min(1.0, curvature_data["magnitude"] * 20)
        
        if hasattr(self, 'previous_attractors') and hasattr(self, 'attractors'):
            if len(self.attractors) != len(self.previous_attractors):
                attractor_change = 0.2  # Significant change
            else:
                # Check for position changes
                for i, att in enumerate(self.attractors):
                    if i < len(self.previous_attractors):
                        pos1 = att[0]
                        pos2 = self.previous_attractors[i][0]
                        if abs(pos1[0] - pos2[0]) > 2 or abs(pos1[1] - pos2[1]) > 2:
                            attractor_change = 0.1  # Moderate change
                            break
        
        novelty_increment = (field_change * 0.5 + 
                            curvature_factor * 0.3 + 
                            attractor_change * 0.2)
            
        # Apply decay to novelty
        novelty_progress = 0.0
        for goal in self.current_goals:
            if goal.get("type", "") == "novelty":
                novelty_progress = goal.get("progress", 0.0)
                break
        
        # just a little decay
        decay_factor = 0.0
        if novelty_progress > 0.4:
            decay_factor = (novelty_progress - 0.4) * (1.25)  
            decay_factor = min(decay_factor, 1.0)  
        
        final_increment = novelty_increment * (1.0 - decay_factor * 0.7) + 0.05 * decay_factor

        if hasattr(self, 'attractors'):   # for next comparison
            self.previous_attractors = [att.copy() if isinstance(att, list) else att for att in self.attractors]
        
        return novelty_increment
    
    # Updated version of update goals to work with enhanced_fatigue_controller
    def _update_goals(self, outcome_metrics):
            """Update goals based on outcome with enhanced fatigue handling"""
            energy_used = 1.0
            if hasattr(self, 'decision_history') and self.decision_history:
                latest_decision = self.decision_history[-1]
                if 'modified_distribution' in latest_decision:
                    energy_used = latest_decision['modified_distribution'].get('total', 1.0)
                elif 'energy_distribution' in latest_decision:
                    energy_used = latest_decision['energy_distribution'].get('total', 1.0)
            
            novelty_progress = 0.0
            for g in self.current_goals:
                if g.get("type") == "novelty":
                    novelty_progress = g.get("progress", 0.0)
                    break
            
            context_drift = self.get_context_drift()
            
            steps_active = len(self.decision_history)
            
            # Includes fatigue aware goal correction with enhanced controller
            self.current_goals, self.fatigue_level = self.fatigue_controller.step(
                self.current_goals,
                energy_used,
                novelty_progress,
                context_drift,
                steps_active
            )
            
            # progress now dependent on fatigue
            default_progress = 0.01 * max(0.5, 1.0 - self.fatigue_level/20.0)
            
            for i, goal in enumerate(self.current_goals):
                goal_id = i
                goal_key = f"goal_{goal_id}_progress"
                
                if goal_key in outcome_metrics:
                    progress_delta = outcome_metrics[goal_key]
                else:
                    stability = outcome_metrics.get("stability", 0.5)
                    energy = outcome_metrics.get("energy_level", 0.5)
                    
                    # Different progress calculation 
                    if goal.get("type") == "stability":
                        progress_delta = default_progress * stability * 2  # Favor stability for stability 
                    elif goal.get("type") == "energy":
                        progress_delta = default_progress * energy * 2  # Favor energy for energy
                    elif goal.get("type") == "novelty":
                        # Must get more progress when unstable for novelty goals
                        progress_delta = default_progress * (1 - stability) * 2
                        
                        # Needs a boost if fatigue is high or drift is rising
                        if self.fatigue_level > 10 or context_drift > 0.5:
                            progress_delta += 0.01  # a small surge
                    else:
                        progress_delta = default_progress
                
                self.update_goal_progress(i, progress_delta)
            
            self.current_goals = [
                goal for goal in self.current_goals
                if goal.get("progress", 0.0) < 0.95 and goal.get("activation", 1.0) > 0.05
            ]
    
    def _update_cognitive_metrics(self, outcome_metrics):
        if self.current_goals:
            progress_sum = sum(g["progress"] * g["priority"] for g in self.current_goals)
            priority_sum = sum(g["priority"] for g in self.current_goals)
            if priority_sum > 0:
                self.cognitive_metrics["goal_alignment"] = progress_sum / priority_sum
   
        if self.inhibition_targets:
            self.cognitive_metrics["inhibition_efficiency"] = outcome_metrics.get("stability", 0.5)

        if "energy_level" in outcome_metrics:
            energy_level = outcome_metrics["energy_level"]
            stability = outcome_metrics.get("stability", 0.5)
            self.cognitive_metrics["resource_efficiency"] = energy_level * stability
        
        # Adding fatigue tracking
        if hasattr(self, "fatigue_level"):
            self.cognitive_metrics["fatigue_level"] = self.fatigue_level
            
            # fatigue impact
            fatigue_factor = min(0.3, self.fatigue_level / 30.0)
            
            # Now high fatigue reduces resource abilities
            if "resource_efficiency" in self.cognitive_metrics:
                self.cognitive_metrics["resource_efficiency"] *= (1.0 - fatigue_factor)
            
            # can also reduce goal alignment
            if "goal_alignment" in self.cognitive_metrics and self.fatigue_level > 20.0:
                self.cognitive_metrics["goal_alignment"] *= (1.0 - fatigue_factor * 0.5)
        
            # Add fatigue tracking (turned off for now)
       # if hasattr(self, "fatigue_level"):
           # self.cognitive_metrics["fatigue_level"] = self.fatigue_level
    
    def get_fatigue_status(self):
        """Get detailed status of the fatigue controller"""
        if hasattr(self, 'fatigue_controller'):
            return self.fatigue_controller.get_status()
        return {"fatigue_level": 0.0}
    
    def forward(self, field, energy_distribution=None, signal_metrics=None):
        """
        Process the field through prefrontal controls
        
        Args:
            field: The current field tensor
            energy_distribution: Current energy distribution
            signal_metrics: Signal metrics from processing
            
        Returns:
            Tuple of (modified field, modified energy distribution)
        """
        print(f"DEBUG: Goals in get_status: {len(self.get_status().get('goals', []))}")
        print(f"DEBUG: Goals in current_goals: {len(self.current_goals)}")
        attended_field = field * self.attention_mask
        
        inhibited_field = attended_field * (1.0 - self.inhibition_field)
        
        available_energy = energy_distribution.get("total", 1.0) if energy_distribution else 1.0
        modified_distribution = self.allocate_resources(
            available_energy, energy_distribution, signal_metrics
        )
    
        print(f"Prefrontal forward: goals={len(self.current_goals)}, distribution={modified_distribution}") # can probably be removed
        
        # With energy distribution
        return inhibited_field, modified_distribution
    
    def get_status(self):
        # with actual values
        if self.cognitive_metrics["goal_alignment"] < 0.01:
            self._initialize_cognitive_metrics()
        
        return {
            "goals": [
                {
                    "priority": g["priority"],
                    "progress": g["progress"],
                    "activation": g["activation"],
                    "type": g.get("type", "generic")
                } for g in self.current_goals
            ],
            "working_memory": [
                {
                    "importance": wm["importance"],
                    "age": wm["age"]
                } for wm in self.working_memory
            ],
            "attention_bias": self.attention_bias.detach().cpu().numpy().tolist(),
            "resource_weights": self.resource_weights.detach().cpu().numpy().tolist(),
            "inhibition_targets": list(self.inhibition_targets),
            "cognitive_metrics": self.cognitive_metrics
        }
        # fatigue information
        if hasattr(self, 'fatigue_level'):
            status['fatigue_level'] = self.fatigue_level
        
        # fatigue debug 
        if hasattr(self, 'fatigue_controller') and hasattr(self.fatigue_controller, 'debug_info'):
            status['fatigue_details'] = self.fatigue_controller.debug_info
        
        return status
    
    def reset(self):
        """Reset the prefrontal module state"""
        self.working_memory = []
        self.current_goals = []
        self.inhibition_targets = set()
        
        with torch.no_grad():
            self.inhibition_field.zero_()
            self.attention_mask.fill_(1.0)
            self.attention_bias.zero_()
            self.pathway_adjustments.zero_()
            self.context_embedding.zero_()
        
        self.decision_history = []
        self.outcome_history = []
        
        # Reset cognitive metrics
        self._initialize_cognitive_metrics()

        if hasattr(self, 'fatigue_controller'):
            self.fatigue_controller.reset()
        
        self.fatigue_level = 0.0
        print(f"RESET: Clearing {len(self.current_goals)} goals") # checking goal carryover

import random

class SignalEnvironment(nn.Module):
    """
    Generates dynamic signals for emergent prefrontal experiments.
    Supports drift and novelty variation to test goal modulation.
    """
    def __init__(self, field_dim=(32, 32), device='cpu', drift_frequency=10, seed=None):
        self.time_step = 0
        self.drift_frequency = drift_frequency
        self.current_focus = random.choice(['stability', 'energy', 'novelty'])
        self.seed = seed
        self.field_dim = field_dim
        self.device = device

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def generate_signal(self):
        """
        Produces a synthetic 2D tensor simulating environmental stimulus.
        """
        # Base value that varies over time
        base = torch.sin(torch.tensor(self.time_step / 5.0)) + torch.randn(1) * 0.1
        
        # 2D tensor with field dimensions
        signal = torch.zeros(self.field_dim, device=self.device)
        
        # Fill with pattern based on the base value
        center_i, center_j = self.field_dim[0]//2, self.field_dim[1]//2
        for i in range(self.field_dim[0]):
            for j in range(self.field_dim[1]):
                # gaussian pattern
                dist = ((i - center_i)/center_i)**2 + ((j - center_j)/center_j)**2
                signal[i, j] = base.item() * np.exp(-3.0 * dist) + torch.randn(1).item() * 0.05
        
        # Normalize
        signal = signal / torch.norm(signal)
        
        self.time_step += 1
        return signal
        
    def trigger_drift(self):
        """
        Randomly change environmental goal focus.
        Used to simulate unexpected shifts in reward structure or priority.
        """
        old_focus = self.current_focus
        available = ['stability', 'energy', 'novelty']
        available.remove(old_focus)
        self.current_focus = random.choice(available)
        print(f"[Environment Drift] Focus changed from {old_focus} to {self.current_focus}")

import torch

class FatigueAwareController(nn.Module):
    def __init__(self, novelty_threshold=0.2, drift_threshold=0.5, fatigue_steps=10):
        self.novelty_threshold = novelty_threshold
        self.drift_threshold = drift_threshold
        self.fatigue_steps = fatigue_steps
        self.fatigue_counter = 0

    def compute_fatigue(self, energy_used, novelty_progress, steps_active):
        if novelty_progress < self.novelty_threshold and steps_active > self.fatigue_steps:
            return energy_used / max(novelty_progress, 1e-3)
        return 0.0

    def detect_unproductive_drift(self, drift_value, novelty_progress):
        return drift_value > self.drift_threshold and novelty_progress < self.novelty_threshold

    def apply_rebalancing(self, goals):
        updated_goals = []
        for goal in goals:
            new_goal = goal.copy()
            if goal["type"] == "stability" and goal["progress"] > 0.75:
                new_goal["activation"] *= 0.5  # decay stability's grip
            if goal["type"] == "novelty":
                new_goal["priority"] += 0.2    # boost novelty during fatigue
            updated_goals.append(new_goal)
        return updated_goals

    def step(self, goals, energy_used, novelty_progress, context_drift, steps_active):
        fatigue = self.compute_fatigue(energy_used, novelty_progress, steps_active)
        unproductive = self.detect_unproductive_drift(context_drift, novelty_progress)

        if fatigue > 10.0 or unproductive:
            self.fatigue_counter += 1
            goals = self.apply_rebalancing(goals)
        else:
            self.fatigue_counter = max(0, self.fatigue_counter - 1)

        return goals, self.fatigue_counter

