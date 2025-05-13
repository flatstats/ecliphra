import torch
import numpy as np
import torch.nn as nn

class EnhancedFatigueController(nn.Module):
    """
    Enhanced fatigue controller that implements the formula:
    Ψ (Fatigue) = (G × ΔS) + Rₘ
    
    G is the effort, ΔS is how much has changed, and Rₘ is what’s left emotionally after a goal.
    Hopefully this helps reduce the feeling that goals always make things worse.
    """
    def __init__(self, 
                 novelty_threshold=0.2, 
                 drift_threshold=0.5, 
                 fatigue_steps=10,
                 recovery_factor=0.4,
                 decay_rate=0.98):
        """
        Initialize the fatigue controller with configurable parameters.
        
        Args:
            novelty_threshold: Minimum novelty progress expected to avoid fatigue
            drift_threshold: Maximum acceptable context drift before considered unproductive
            fatigue_steps: Minimum steps active before fatigue can accumulate
            recovery_factor: How much goal completion reduces fatigue (higher = more recovery)
            decay_rate: Natural decay rate of fatigue over time (0-1)
        """
        self.novelty_threshold = novelty_threshold
        self.drift_threshold = drift_threshold
        self.fatigue_steps = fatigue_steps
        self.recovery_factor = recovery_factor
        self.decay_rate = decay_rate
        
        self.fatigue_level = 0.0
        self.previous_goal_progress = {}
        self.goal_completion_history = []
        self.fatigue_history = []
        self.debug_info = {}
    
    def compute_fatigue_increment(self, goals, energy_used, context_drift, novelty_progress, steps_active):
        """
        Finding fatigue using the formula: Ψ = (G × ΔS) + Rₘ
        
        Args:
            goals: List of the current goals and their states or what they are holding onto
            energy_used: How much energy or effort roughly for each step
            context_drift: How far the current state is to recent memory
            novelty_progress: Did it find something new or just stabilizing
            steps_active: How long the system has been holding a goal
            
        Returns:
            Fatigue increment for the current step
        """
        if steps_active < self.fatigue_steps:
            return 0.0
            
        # Calculate goal intensity (G) that would mean higher energy use and lower novelty are equal to higher intensity

        # to represent diminishing returns
        if novelty_progress < self.novelty_threshold:
            # High effort with little progress means high intensity
            goal_intensity = min(1.0, energy_used / max(novelty_progress, 0.05))
        else:
            # Normal effort with good progress means lower intensity
            goal_intensity = min(1.0, energy_used * 0.5)
        
        # Here is the shift in cognitive state (ΔS) based on context drift
        # Higher drift with low novelty progress indicates inefficient cognitive shifts
        shift_magnitude = min(1.0, context_drift * (2.0 - novelty_progress))
        
        residual_meaning = 0.0
        completed_goals = []
        active_goals = []
        
        for goal_id, goal in enumerate(goals):
            goal_type = goal.get("type", "generic")
            current_progress = goal.get("progress", 0.0)
            previous_progress = self.previous_goal_progress.get(goal_id, 0.0)
            priority = goal.get("priority", 0.5)
            
            if current_progress < 0.95:
                active_goals.append(goal_id)
            
            progress_delta = current_progress - previous_progress
            
            if current_progress > 0.95 and previous_progress <= 0.95:
                completed_goals.append(goal_id)
                
                # Positive residual meaning which is restorative means that higher priority goals provide more recovery
                restoration = priority * self.recovery_factor
                residual_meaning -= restoration
                
                self.goal_completion_history.append({
                    "goal_id": goal_id,
                    "goal_type": goal_type,
                    "priority": priority,
                    "restoration": restoration
                })
            
            # Small continuous recovery from making progress on goals
            elif progress_delta > 0.01:
                # Small recovery from incremental progress which would be 10% of completion recovery
                small_recovery = progress_delta * priority * self.recovery_factor * 0.1
                residual_meaning -= small_recovery
            
            # Giving it a slight increase in fatigue when making no progress on active goals, see how that goes
            elif current_progress < 0.9 and progress_delta < 0.001:
                residual_meaning += 0.01 * priority # Would be from stagnation
            
            self.previous_goal_progress[goal_id] = current_progress
        
        for goal_id in list(self.previous_goal_progress.keys()):
            if goal_id not in active_goals and goal_id not in completed_goals:
                del self.previous_goal_progress[goal_id]
        
        # Using formula Ψ = (G × ΔS) + Rₘ
        fatigue_increment = (goal_intensity * shift_magnitude) + residual_meaning
        
        self.debug_info = {
            "goal_intensity": goal_intensity,
            "shift_magnitude": shift_magnitude,
            "residual_meaning": residual_meaning,
            "completed_goals": completed_goals,
            "fatigue_increment": fatigue_increment
        }
        
        return fatigue_increment
    
    def compute_fatigue_factor(self):
        """
         Finding normalized fatigue from fatigue_level.
        Scales and caps the fatigue influence used across modules.
        """
        return min(0.8, self.fatigue_level / 20.0)

    
    def detect_unproductive_drift(self, drift_value, novelty_progress):
        """
        Discovering whether cognitive drift is unproductive.
        
        Args:
            drift_value: Context 
            novelty_progress: Progress towards the goal
            
        Returns:
            Boolean indicating whether drift is unproductive
        """
        return drift_value > self.drift_threshold and novelty_progress < self.novelty_threshold
    
    def apply_rebalancing(self, goals):
        """
        When fatigue is high, rebalance to give more attention to novelty and lower the stability pressure.
        
        Args:
            goals: List of goals currently
            
        Returns:
            Updated list of goals
        """
        updated_goals = []
        for goal in goals:
            new_goal = goal.copy()
            
            # Scaling the adjustments
            fatigue_factor = self.compute_fatigue_factor()
            
            if goal["type"] == "stability" and goal["progress"] > 0.5:
                # Going to reduce stability pressure as fatigue gets higher
                new_goal["activation"] *= max(0.3, 1.0 - fatigue_factor)
                new_goal["priority"] *= max(0.6, 1.0 - fatigue_factor * 0.5)
            
            elif goal["type"] == "novelty":
                # Give a boost to novelty and activation during fatigue
                new_goal["priority"] = min(1.0, new_goal["priority"] + fatigue_factor * 0.3)
                new_goal["activation"] = min(1.0, new_goal["activation"] + fatigue_factor * 0.2)
            
            elif goal["type"] == "energy" and self.fatigue_level > 15.0:
                # If there is severe fatigue, energy recovery gets more attention
                new_goal["priority"] = min(1.0, new_goal["priority"] + fatigue_factor * 0.2)
            
            updated_goals.append(new_goal)
        
        return updated_goals
    
    def step(self, goals, energy_used, novelty_progress, context_drift, steps_active):
        """
        Process one step of fatigue calculation, goal rebalancing, and return updated state.
        
        Args:
            goals: Current list of goals
            energy_used: Whats consumed in current step
            novelty_progress: Towards novelty goals
            context_drift: How much the cognitive context has changed
            steps_active: How many steps the system has been active
            
        Returns:
            Tuple of (updated_goals, fatigue_level)
        """
        # What has changed
        fatigue_increment = self.compute_fatigue_increment(
            goals, energy_used, context_drift, novelty_progress, steps_active
        )
        
        # Check for unproductive drift, because it's the problem
        unproductive = self.detect_unproductive_drift(context_drift, novelty_progress)
        if unproductive:
            fatigue_increment += 0.5  # penalty for unproductives (sorry)
        
        # Update with natural decay. Decay first, new increments later
        self.fatigue_level = self.fatigue_level * self.decay_rate + fatigue_increment
        
        self.fatigue_level = max(0.0, self.fatigue_level)  # No go below 0
        
        self.fatigue_history.append(self.fatigue_level)
        
        # When fatigue is high
        updated_goals = goals
        if self.fatigue_level > 10.0:
            updated_goals = self.apply_rebalancing(goals)
        
        return updated_goals, self.fatigue_level
    
    def get_status(self):
        """
        For the fatigue controller.
        
        Returns:
            Dict with fatigue status information
        """
        return {
            "fatigue_level": self.fatigue_level,
            "fatigue_history": self.fatigue_history[-10:] if self.fatigue_history else [],
            "completed_goals": len(self.goal_completion_history),
            "debug_info": self.debug_info
        }
    
    def reset(self):
        """Reset the fatigue controller state."""
        self.fatigue_level = 0.0
        self.previous_goal_progress = {}
        self.goal_completion_history = []
        self.fatigue_history = []
        self.debug_info = {}