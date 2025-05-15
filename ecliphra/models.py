"""
Ecliphra Models Module

This module contains all model implementations for the Ecliphra system:
- Base EcliphraField
- EcliphraFieldWithEcho
- EcliphraFieldWithSemantics
- EcliphraWithEnhancedFingerprinting
- EcliphraWithPersistentMemory
- EcliphraFieldWithPhotoSynthesis
- EcliphraFieldWithEnergySystem
- EcliphraFieldWithPrefrontal
- Legacy comparison models

Models are organized hierarchically, with each new capability building on previous ones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import time
from ecliphra.utils.enhanced_fingerprinting import (
    EnhancedFingerprinting
)
from ecliphra.utils.photofield import SignalRouter, FieldRewritingProtocol, IntegrationThresholdLayer, MemoryAugmentedClassifier, EcliphraPhotoField
from ecliphra.utils.energy_system import SignalAnalyzer, FieldModulator, EnergyDistributor, PhotosynthesisEnergySystem
from ecliphra.utils.prefrontal import PrefrontalModule
from ecliphra.utils.curvature_riding import integrate_curvature_riding, integrate_curvature_riding_with_fatigue, implement_consolidation_phase, curvature_aware_forward

class EcliphraField(nn.Module):
    """
    Base physics-based tensor field with emergent attractors.

    This is the foundation model that implements the core field dynamics.
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim
        self.field = nn.Parameter(torch.zeros(field_dim, device=device))
        self.register_buffer('velocity', torch.zeros(field_dim, device=device))
        self.register_buffer('adaptivity', torch.zeros(field_dim, device=device))
        

        self.stability = nn.Parameter(torch.tensor(0.85, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.2, device=device))
        self.excitation = nn.Parameter(torch.tensor(0.5, device=device))
        self.adaptivity_growth_rate = 0.2
        self.adaptivity_decay_rate = 0.05
        self.adaptivity_maximum = 3.0
        self.disruption_threshold = 0.1

        
        self.attractors = []
        self.field_history = []
        self.field_diff_history = []
        self.adaptivity_summary = []
        self.step = 0

       
        self.signal_log = []            
        self.curiosity_log = []         
        self.signal_threshold = 0.3
        self.curiosity_threshold = 0.7

      


    def forward(self, input_tensor=None, input_pos=None):
        self.step += 1
        prev = self.field.detach().clone()

        # The occasional check and address of classification slown down
        if len(self.signal_processing_history) % 10 == 0 and len(self.signal_processing_history) > 0:
            self.inject_entropy()

        self.apply_weight_neutrality_decay(steps=10, decay_factor=0.99)

        if input_tensor is not None:
            pos = input_pos or (self.field_dim[0]//2, self.field_dim[1]//2)
            self.apply_input(input_tensor, pos)

        if len(self.field_history)>=10: self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())

        self.update_field_physics()
        self.detect_attractors()

        diff = torch.norm(self.field - prev).item()
        self.field_diff_history.append(diff)
        if len(self.field_diff_history)>10: self.field_diff_history.pop(0)

        self.update_adaptivity(prev, self.field)

        stability = self.calculate_stability()
        novelty = 1.0 - stability

        if stability > 0.6:
            resonance += 0.05

        # Logging 
        if stability < self.signal_threshold:
            self.signal_log.append({'step':self.step,'stability':stability})
        if novelty > self.curiosity_threshold:
            self.curiosity_log.append({'step':self.step,'novelty':novelty})

        return {'field':self.field,'attractors':self.attractors,'stability':stability,'novelty':novelty}

        # Adding a new method to update adaptivity to fix logs
    def update_adaptivity(self, prev_field, current_field):

        field_change = torch.abs(current_field - prev_field)

        disruption_mask = field_change > self.disruption_threshold
        growth = field_change * self.adaptivity_growth_rate

        self.adaptivity[disruption_mask] += growth[disruption_mask]

        self.adaptivity = torch.clamp(self.adaptivity, max=self.adaptivity_maximum)

        self.adaptivity = self.adaptivity * (1.0 - self.adaptivity_decay_rate)

    def apply_input(self, input_tensor, pos):
        """Apply input as force to field"""
        i, j = pos
        radius = 3  

        
        if input_tensor.dim() > 1:
            input_tensor = input_tensor.mean(dim=0)

        
        magnitude = torch.norm(input_tensor) * self.excitation

        # This will apply gaussian distribution of force around the position
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                ni, nj = i+di, j+dj
                if 0 <= ni < self.field_dim[0] and 0 <= nj < self.field_dim[1]:
                    # Gaussian falloff from center
                    dist = torch.sqrt(torch.tensor(float(di**2 + dj**2)))
                    factor = torch.exp(-0.5 * (dist / 2)**2)
                    self.velocity[ni, nj] += magnitude * factor

    def update_field_physics(self):
        """Update field based on physical model"""
        # Calculate Laplacian (∇²F) measures curvature
        laplacian = self.compute_laplacian()

        # The wave equation: acceleration = c²∇²F
        self.velocity += self.propagation * laplacian

        self.field.data += self.velocity
        # (damping)
        self.velocity *= self.stability

    def compute_laplacian(self):
        """Compute the Laplacian of the field"""
        # Had to use manual implementation of Laplacian using basic indexing due to mismatches before
        laplacian = torch.zeros_like(self.field)

        # Inner region that excludes the boundaries
        h, w = self.field.shape
        for i in range(1, h-1):
            for j in range(1, w-1):
                # 5 point stencil
                center = self.field[i, j]
                up = self.field[i-1, j]
                down = self.field[i+1, j]
                left = self.field[i, j-1]
                right = self.field[i, j+1]

                laplacian[i, j] = up + down + left + right - 4 * center

        # Boundaries simplified
        # Top edge
        laplacian[0, 1:-1] = self.field[1, 1:-1] + self.field[0, :-2] + self.field[0, 2:] - 3 * self.field[0, 1:-1]
        # Bottom edge
        laplacian[-1, 1:-1] = self.field[-2, 1:-1] + self.field[-1, :-2] + self.field[-1, 2:] - 3 * self.field[-1, 1:-1]
        # Left edge
        laplacian[1:-1, 0] = self.field[:-2, 0] + self.field[2:, 0] + self.field[1:-1, 1] - 3 * self.field[1:-1, 0]
        # Right edge
        laplacian[1:-1, -1] = self.field[:-2, -1] + self.field[2:, -1] + self.field[1:-1, -2] - 3 * self.field[1:-1, -1]

        # Corners simplified
        laplacian[0, 0] = self.field[1, 0] + self.field[0, 1] - 2 * self.field[0, 0]
        laplacian[0, -1] = self.field[1, -1] + self.field[0, -2] - 2 * self.field[0, -1]
        laplacian[-1, 0] = self.field[-2, 0] + self.field[-1, 1] - 2 * self.field[-1, 0]
        laplacian[-1, -1] = self.field[-2, -1] + self.field[-1, -2] - 2 * self.field[-1, -1]

        return laplacian

    def detect_attractors(self):
        """Find emergent attractors in the field with adaptive basin sizing"""
        h, w = self.field.shape
        smoothed = F.avg_pool2d(
            self.field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()

        field_mean = smoothed.mean().item()
        field_std = smoothed.std().item()
        threshold = field_mean + 0.3 * field_std  # More sensitive threshold
        
        candidates = []
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = smoothed[i, j].item()
                if center < threshold:
                    continue
                
                neighborhood = smoothed[i-1:i+2, j-1:j+2]
                if center >= torch.max(neighborhood).item():
                    # Modifying basin size for more sensitivity to pattern variation
                    base_basin_size = self.calculate_basin(i, j, sensitivity=1.5)  # Increased
                    
                    energy_multiplier = 1.0
                    if hasattr(self, 'energy_current'):
                        energy_multiplier = 1.0 + self.energy_current[i, j].item() * 0.7  
                    
                    adaptive_basin_size = base_basin_size * energy_multiplier
                    
                    # Looking for more pattern-specific weighting
                    pattern_weight = 1.0
                    if hasattr(self, 'last_pattern_type'):
                        pattern_weight = 1.2 if self.last_pattern_type == "spiral" else 1.0
                    
                    strength = center * adaptive_basin_size * pattern_weight
                    
                    candidates.append(((i, j), strength, base_basin_size, adaptive_basin_size))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Modifying to store more attractor information
        self.attractors = [(pos, strength, base_size, adaptive_size)
                        for (pos, strength, base_size, adaptive_size) in candidates[:3]]

    def calculate_basin(self, i, j, sensitivity=1.0):
        """Calculate basin of attraction width"""
        h, w = self.field.shape
        grad_sum = 0.0
        count = 0

        # neighborhood gradient
        for ni in range(max(0, i-2), min(h, i+3)):
            for nj in range(max(0, j-2), min(w, j+3)):
                if ni == i and nj == j:
                    continue

                grad = abs(self.field[ni, nj].item() - self.field[i, j].item())
                grad_sum += grad
                count += 1

        avg_grad = grad_sum / count if count > 0 else 1.0 # note to self: smaller means wider basin

        # smaller gradient = wider basin
        if avg_grad < 0.001:
            return 5.0  # Cap on basin size
        return min(5.0, 1.0 / avg_grad)

    def calculate_stability(self):
        """Calculate overall field stability"""
        if len(self.field_diff_history) < 2:
            return 0.5  # Default neutral stability

        recent_diffs = self.field_diff_history[-5:]
        weights = torch.linspace(0.5, 1.0, len(recent_diffs))
        weighted_diff = sum(d * w for d, w in zip(recent_diffs, weights)) / sum(weights)

        stability = 1.0 / (1.0 + weighted_diff * 4.0)  # Increased sensitivity multiplier
        
        # Adding field entropy to calculation in stability (oops)
        field_norm = F.normalize(self.field.view(-1), p=2, dim=0)
        field_squared = field_norm * field_norm
        entropy = -torch.sum(field_squared * torch.log2(field_squared + 1e-10))
        entropy_factor = torch.clamp(entropy / 10.0, 0.0, 1.0).item()

        stability = stability * 0.8 + (1.0 - entropy_factor) * 0.2

        return stability

    def process_with_metrics(self, pattern, step, noise_level):
        """Process a pattern and collect detailed metrics"""
        self.reset()
        
        result = self(input_tensor=pattern)
        
        if hasattr(self, 'create_robust_fingerprint'):
            fingerprint, pattern_type = self.create_robust_fingerprint(pattern)
        else:
            fingerprint, pattern_type = None, "unknown"
        
        metrics = {
            'step': step,
            'noise_level': noise_level,
            'stability': result.get('stability', 0.0),
            'attractor_count': len(result.get('attractors', [])),
            'pattern_type': pattern_type,
            'field_norm': torch.norm(self.field).item(),
            'field_mean': self.field.mean().item(),
            'field_std': self.field.std().item(),
            'field_max': self.field.max().item(),
            'field_min': self.field.min().item(),
        }
        
        if hasattr(self, 'get_energy_statistics'):
            energy_stats = self.get_energy_statistics()
            metrics.update(energy_stats)
        
        return metrics
        return stability

    def reset(self):
        """Reset field state"""
        with torch.no_grad():
            self.field.zero_()
            self.velocity.zero_()
            self.field_history = []
            self.field_diff_history = []
            self.attractors = []

    def associate_content(self, content):
        """Associate content with strongest attractor"""
        if self.attractors:
            pos, strength = self.attractors[0]
            self.attractors[0] = (pos, strength, content[:50])  # store summary


class EcliphraFieldWithEcho(EcliphraField):
    """
    Enhanced EcliphraField with Echo Resonance Cycles.

    This model adds self-reinforcement through echo resonance,
    allowing the field to maintain patterns over time.
    """
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5):
        super().__init__(field_dim, device)

        # Override physical parameters for better echo performance
        self.stability = nn.Parameter(torch.tensor(0.92, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.08, device=device))
        self.excitation = nn.Parameter(torch.tensor(0.3, device=device))

        # Echo resonance parameters
        self.echo_strength = nn.Parameter(torch.tensor(0.6, device=device))
        self.echo_decay = nn.Parameter(torch.tensor(0.95, device=device))
        self.quiet_threshold = 0.15
        

        self.attractor_memory = []  # Will store (position, strength, field_state) tuples
        self.memory_capacity = memory_capacity
        self.memory_counter = 0  

        self.velocity_norm_history = []
        self.echo_history = []  

        self.register_buffer('time_step', torch.tensor(0.0))

        self.step = 0

        self.signal_threshold = 0.7  
        self.curiosity_threshold = 0.5  
        self.signal_log = []
        self.curiosity_log = []

        self._initialize_field()

    def _initialize_field(self):
        """Initialize with stable pattern"""
        h, w = self.field_dim
        cx, cy = w // 2, h // 2

        with torch.no_grad():
            # Create a smooth radial gradient
            for i in range(h):
                for j in range(w):
                    # Distance from center (normalized)
                    dist = torch.sqrt(torch.tensor(((i-cy)/cy)**2 + ((j-cx)/cx)**2))
                    # Smooth falloff
                    self.field[i, j] = torch.exp(-3.0 * dist)

            self.field.data = self.field.data / torch.norm(self.field.data)

    def forward(self, input_tensor=None, input_pos=None):
        """Process an input through the field with echo resonance"""
        # storing for comparison
        previous_field = self.field.detach().clone()

        self.step += 1

        input_applied = False
        if input_tensor is not None:
            if input_pos is None:
                input_pos = (self.field_dim[0]//2, self.field_dim[1]//2)

            self.apply_input(input_tensor, input_pos)
            input_applied = True

        if len(self.field_history) >= 10:
            self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())

        self.update_field_physics()

        with torch.no_grad():
            field_norm = torch.norm(self.field.data)
            if field_norm < 0.5 or field_norm > 2.0:
                self.field.data = self.field.data / field_norm

        self.detect_attractors() # For emergent attractors

        self.update_attractor_memory() # Store the strongest attractor

        # Here it decides whether to apply echo resonance
        echo_applied = False
        if not input_applied:  
            velocity_norm = torch.norm(self.velocity).item()
            self.velocity_norm_history.append(velocity_norm)

            # If the field is relatively quiet (low activity)
            if velocity_norm < self.quiet_threshold:
                echo_applied = self.apply_echo_resonance()

        field_diff = torch.norm(self.field - previous_field).item()
        if len(self.field_diff_history) >= 10:
            self.field_diff_history.pop(0)
        self.field_diff_history.append(field_diff)

        stability = self.calculate_stability()

        novelty = 1.0 - stability

        if stability < self.signal_threshold:
            self.signal_log.append({'step': self.step, 'stability': stability})
        if novelty > self.curiosity_threshold:
            self.curiosity_log.append({'step': self.step, 'novelty': novelty})

        resonance = 0.5  # Default
        if len(self.field_history) >= 2:
            latest = self.field_history[-1].view(-1)
            previous = self.field_history[-2].view(-1)
            resonance = F.cosine_similarity(latest, previous, dim=0).item()

        self.echo_history.append(1.0 if echo_applied else 0.0)


        self.time_step += 1.0

        return {
            'field': self.field,
            'attractors': self.attractors,
            'stability': stability,
            'novelty': novelty,  
            'resonance': resonance,
            'echo_applied': echo_applied,
            'attractor_memory': self.attractor_memory[:]
        }
    
    def update_attractor_memory(self):
        """Update the attractor memory with current strong attractors"""
        if not self.attractors:
            return

        # Get strongest attractor
        if len(self.attractors[0]) == 2:
            pos, strength = self.attractors[0]
        elif len(self.attractors[0]) >= 3:
            pos, strength = self.attractors[0][:2]  # Just using the first two elements
        else:
            return

        # Only storing if it's a significant attractor
        if strength > 0.3:
            i, j = pos  # Extract a patch around the attractor
            radius = 5  # Size of stored field patch

            # Create patch extraction indices with bounds checking
            i_min = max(0, i - radius)
            i_max = min(self.field_dim[0], i + radius + 1)
            j_min = max(0, j - radius)
            j_max = min(self.field_dim[1], j + radius + 1)

            patch = self.field[i_min:i_max, j_min:j_max].detach().clone()

            memory_entry = {
                'position': pos,
                'strength': strength,
                'patch': patch,
                'patch_coords': (i_min, i_max, j_min, j_max),
                'time': self.time_step.item(),
                'echo_count': 0  
            }

            # Adding to memory
            self.attractor_memory.append(memory_entry)
            if len(self.attractor_memory) > self.memory_capacity:
                # Remove weakest or oldest entry
                self.attractor_memory.sort(key=lambda x: x['strength'], reverse=True)
                self.attractor_memory = self.attractor_memory[:self.memory_capacity]

    def apply_echo_resonance(self):
        """This will apply an echo from memory to create self-reinforcing patterns"""
        if not self.attractor_memory:
            return False

        self.memory_counter += 1

        sorted_memory = sorted(
            self.attractor_memory,
            key=lambda x: x['strength'],
            reverse=True
        )

        if np.random.random() < 0.2 and len(sorted_memory) > 1:
            echo_idx = np.random.randint(1, len(sorted_memory))
            memory_to_echo = sorted_memory[echo_idx]
        else:
            memory_to_echo = sorted_memory[0]

        memory_strength = memory_to_echo['strength']
        pos = memory_to_echo['position']
        patch = memory_to_echo['patch']
        i_min, i_max, j_min, j_max = memory_to_echo['patch_coords']

        echo_count = memory_to_echo['echo_count']
        echo_factor = self.echo_strength * (self.echo_decay ** echo_count)

        # A memory patch as a gentle field influence
        with torch.no_grad():
            current_patch = self.field[i_min:i_max, j_min:j_max]
            self.field[i_min:i_max, j_min:j_max] = (1 - echo_factor) * current_patch + echo_factor * patch

            self.velocity[i_min:i_max, j_min:j_max] += echo_factor * 0.1 * patch

        memory_to_echo['echo_count'] += 1

        # If echoed too many times, decay its strength
        if memory_to_echo['echo_count'] > 3:
            memory_to_echo['strength'] *= 0.9

        return True

    def reset(self):
        """Reset field state"""
        with torch.no_grad():
            self._initialize_field()
            self.velocity.zero_()
            self.field_history = []
            self.field_diff_history = []
            self.attractors = []
            self.attractor_memory = []
            self.echo_history = []
            self.time_step.zero_()
            self.step = 0
            self.signal_log = []
            self.curiosity_log = []
            self.velocity_norm_history = []
       


class EcliphraFieldWithSemantics(EcliphraFieldWithEcho):
    """
    This model adds the ability to recognize semantically similar patterns
    and associate them with existing attractors.
    """

    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5, fingerprint_dim=768):
        super().__init__(field_dim, device, memory_capacity)

        self.semantic_threshold = 0.65

        self.adaptive_threshold = True  
        self.min_threshold = 0.5  

        self.semantic_decay = 0.98 # how quickly semantics fade without reinforcement

        # Store when we last matched an input semantically
        self.last_semantic_match = None

        # Try to use Enhancedfingerprinting
        if 'EnhancedFingerprinting' in globals():
            self.fingerprinter = EnhancedFingerprinting(field_dim=field_dim, device=device)
            self.has_enhanced_fingerprinting = True
        else:
            self.has_enhanced_fingerprinting = False

    def create_robust_fingerprint(self, input_tensor):
        """ A noise-resistant fingerprint from input tensor"""
        if hasattr(self, 'has_enhanced_fingerprinting') and self.has_enhanced_fingerprinting:
            result = self.fingerprinter.create_robust_fingerprint(input_tensor)
            
            if isinstance(result, tuple):
                return result[0]  # Just return the fingerprint part
            return result

        # Fallback to basic fingerprinting
        smoothed = F.avg_pool2d(
            input_tensor.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()

        h, w = smoothed.shape
        features = []

        # Global statistics (resistant to noise)
        features.append(torch.mean(smoothed))
        features.append(torch.std(smoothed))

        # Center of mass coordinates
        total_mass = torch.sum(smoothed)
        if total_mass > 0:
            y_indices = torch.arange(h, device=self.device).float()
            x_indices = torch.arange(w, device=self.device).float()

            y_center = torch.sum(torch.outer(y_indices, torch.ones(w, device=self.device)) * smoothed) / total_mass
            x_center = torch.sum(torch.outer(torch.ones(h, device=self.device), x_indices) * smoothed) / total_mass

            features.append(y_center / h)  # Normalized position
            features.append(x_center / w)  
        else:
            features.append(0.5)
            features.append(0.5)

        # Radial features (scale-invariant)
        center_y, center_x = h // 2, w // 2
        for radius in [0.25, 0.5, 0.75]:
            mask = torch.zeros_like(smoothed)
            for i in range(h):
                for j in range(w):
                    # Normalized distance from center 
                    dist_squared = ((i - center_y) / h)**2 + ((j - center_x) / w)**2
                    dist = torch.sqrt(torch.tensor(dist_squared, device=self.device))
                    if dist <= radius:
                        mask[i, j] = 1.0

            # average value within this radius
            masked_mean = torch.sum(smoothed * mask) / max(torch.sum(mask), 1)
            features.append(masked_mean)

        # Low-frequency components (more robust to noise)
        grid_features = F.avg_pool2d(
            smoothed.unsqueeze(0).unsqueeze(0),
            kernel_size=h//4, stride=h//4
        ).squeeze().flatten()

        fingerprint = torch.cat([torch.tensor(features, device=self.device), grid_features])

        return F.normalize(fingerprint, p=2, dim=0)  # for cosine similarity

    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None):
        """
        Process input with semantic fingerprinting

        Args:
            input_tensor: The field influence
            input_fingerprint: Optional semantic fingerprint vector of the input
            input_pos: Position which is default center
        """
        if input_fingerprint is None and input_tensor is not None:
            input_fingerprint = self.create_robust_fingerprint(input_tensor)

        if isinstance(input_fingerprint, tuple):
            input_fingerprint = input_fingerprint[0]

        # Looking for semantic matches with existing memories
        semantic_match_applied = False
        match_similarity = 0.0
        if input_fingerprint is not None:
            semantic_match_applied, match_similarity = self.apply_semantic_matching(input_fingerprint)

       
        result = super().forward(input_tensor, input_pos)

        # Capture semantics of strong attractors
        if input_fingerprint is not None and len(self.attractors) > 0:
            self.associate_fingerprints(input_fingerprint)

        result['semantic_match_applied'] = semantic_match_applied
        result['match_similarity'] = match_similarity

        return result

    def associate_fingerprints(self, input_fingerprint):
        """Associate with current attractors"""
        for attractor in self.attractors:
            # Handle both old and new 
            if len(attractor) == 2:
                pos, strength = attractor
            elif len(attractor) >= 4:
                pos, strength = attractor[:2]  
            else:
                continue  

            if strength > 0.3:  # Only with significant attractors
                memory_entry = None
                for entry in self.attractor_memory:
                    if entry['position'] == pos:
                        memory_entry = entry
                        break

                if memory_entry is None:
                    memory_entry = {
                        'position': pos,
                        'strength': strength,
                        'patch': self.extract_patch(pos),
                        'patch_coords': self.get_patch_coords(pos),
                        'time': self.time_step.item(),
                        'echo_count': 0,
                        'fingerprint': input_fingerprint.clone().detach(),
                        'semantic_hits': 1
                    }
                    self.attractor_memory.append(memory_entry)
                else:
                    # Update existing 
                    if 'fingerprint' in memory_entry:
                        # Gradually blend 70% old, 30% new
                        old_fp = memory_entry['fingerprint']
                        memory_entry['fingerprint'] = F.normalize(
                            0.7 * old_fp + 0.3 * input_fingerprint, p=2, dim=0)
                        memory_entry['semantic_hits'] = memory_entry.get('semantic_hits', 0) + 1
                    else:
                        memory_entry['fingerprint'] = input_fingerprint.clone().detach()
                        memory_entry['semantic_hits'] = 1

                # To maintain capacity limit
                if len(self.attractor_memory) > self.memory_capacity:
                    # Remove weakest or oldest 
                    self.attractor_memory.sort(key=lambda x: x['strength'], reverse=True)
                self.attractor_memory = self.attractor_memory[:self.memory_capacity]

    def extract_patch(self, pos):
        """Extract field around position"""
        i, j = pos
        radius = 5  # stored field patch

        # The patch extraction indices with bounds checking
        i_min = max(0, i - radius)
        i_max = min(self.field_dim[0], i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.field_dim[1], j + radius + 1)

        # Extract 
        return self.field[i_min:i_max, j_min:j_max].detach().clone()

    def get_patch_coords(self, pos):
        """Get location for extraction"""
        i, j = pos
        radius = 5

        i_min = max(0, i - radius)
        i_max = min(self.field_dim[0], i + radius + 1)
        j_min = max(0, j - radius)
        j_max = min(self.field_dim[1], j + radius + 1)

        return (i_min, i_max, j_min, j_max)

    def apply_semantic_matching(self, input_fingerprint):
        if not self.attractor_memory:
            return False, 0.0

        similarities = []
        for entry in self.attractor_memory:
            if 'fingerprint' not in entry:
                continue

            similarity = F.cosine_similarity(
                input_fingerprint.view(1, -1),
                entry['fingerprint'].view(1, -1)
            ).item()

            similarities.append((entry, similarity))

        if not similarities:
            return False, 0.0

        similarities.sort(key=lambda x: x[1], reverse=True)
        best_entry, best_similarity = similarities[0]

        threshold = self.semantic_threshold
        if self.adaptive_threshold:
            if best_similarity > 0.9:  # Clean signal
                threshold = self.semantic_threshold
            elif best_similarity > 0.8:  # Minor noise
                threshold = max(self.semantic_threshold - 0.05, self.min_threshold)
            elif best_similarity > 0.7:  # Moderate noise
                threshold = max(self.semantic_threshold - 0.1, self.min_threshold)
            else:  # Significant noise
                threshold = max(self.semantic_threshold - 0.15, self.min_threshold)

        matches = [(entry, sim) for entry, sim in similarities if sim >= threshold]

        if not matches:
            return False, best_similarity  # even if below threshold

        with torch.no_grad():
            for entry, similarity in matches[:2]:
                pos = entry['position']
                i, j = pos

                # Finding reinforcement based on semantics
                strength = similarity * 0.2  # Controlled reinforcement

                # Apply gaussian influence around the position
                radius = int(5 * similarity)  # The more similar the wider the influence
                for di in range(-radius, radius+1):
                    for dj in range(-radius, radius+1):
                        ni, nj = i+di, j+dj
                        if 0 <= ni < self.field_dim[0] and 0 <= nj < self.field_dim[1]:
                            # Distance from center
                            dist = torch.sqrt(torch.tensor(float(di**2 + dj**2)))
                            # Gaussian falloff
                            factor = torch.exp(-0.5 * (dist / radius)**2)
                            # Apply influence
                            self.field[ni, nj] += factor * strength * self.field[i, j]

                # Matches
                entry['semantic_hits'] = entry.get('semantic_hits', 0) + 1
                self.last_semantic_match = {
                    'position': pos,
                    'similarity': similarity,
                    'time': self.time_step.item()
                }

        return True, best_similarity

    def apply_semantic_decay(self):
        """Apply decay to memories"""
        current_time = self.time_step.item()

        for entry in self.attractor_memory:
            if 'fingerprint' in entry:
                last_hit_time = entry.get('last_semantic_hit_time', entry.get('time', 0))
                time_since_hit = current_time - last_hit_time

                # Apply decay 
                if time_since_hit > 10:  # If not used in a while
                    # Add small noise
                    noise_scale = 0.01 * min(time_since_hit / 20, 0.2)  # Cap at 20% noise
                    noise = torch.randn_like(entry['fingerprint']) * noise_scale
                    entry['fingerprint'] = F.normalize(entry['fingerprint'] + noise, p=2, dim=0)

    def reset(self):
        """Including semantic information"""
        super().reset()
        self.last_semantic_match = None


from ecliphra.utils.enhanced_fingerprinting import EnhancedFingerprinting

class EcliphraWithEnhancedFingerprinting(torch.nn.Module):
    """
    Extends the Ecliphra model with enhanced fingerprinting.

    This is a simplified version for demonstrating the integration.
    I will need to fully add it to EcliphraFieldWithSemantics class at
    a later date.
    """
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5):
        super().__init__()
        self.device = device
        self.field_dim = field_dim

        self.fingerprinter = EnhancedFingerprinting(field_dim, device)

        # simplified for demonstration
        self.field = torch.nn.Parameter(torch.zeros(field_dim, device=device))
        self.velocity = torch.zeros(field_dim, device=device)
        self.stability = torch.nn.Parameter(torch.tensor(0.85, device=device))
        self.propagation = torch.nn.Parameter(torch.tensor(0.2, device=device))
        self.excitation = torch.nn.Parameter(torch.tensor(0.3, device=device))

        # Echo simplified
        self.echo_strength = torch.nn.Parameter(torch.tensor(0.35, device=device))
        self.memory_capacity = memory_capacity
        self.attractor_memory = []
        self.time_step = torch.tensor(0.0, device=device)

        # Matching parameters
        self.semantic_threshold = 0.65

    def forward(self, input_tensor=None, input_pos=None):
        """Process input through the field with enhanced fingerprinting."""
        input_fingerprint = None
        if input_tensor is not None:
            input_fingerprint = self.fingerprinter.create_robust_fingerprint(input_tensor)

        semantic_match_applied = False
        match_similarity = 0.0

        if input_fingerprint is not None:
            semantic_match_applied, match_similarity = self.apply_semantic_matching(input_fingerprint)

        if input_tensor is not None:
            if input_pos is None:
                input_pos = (self.field_dim[0]//2, self.field_dim[1]//2)
            self.apply_input(input_tensor, input_pos)

        self.update_field_physics()

        self.time_step += 1.0

        return {
            'field': self.field,
            'semantic_match_applied': semantic_match_applied,
            'match_similarity': match_similarity
        }

    def apply_input(self, input_tensor, pos):
        """ Applied as force to field."""
        i, j = pos
        radius = 3  # Affect a radius around the position

        # Apply gaussian distribution of force around the position
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                ni, nj = i+di, j+dj
                if 0 <= ni < self.field_dim[0] and 0 <= nj < self.field_dim[1]:
                    # Gaussian falloff from center
                    dist = torch.sqrt(torch.tensor(float(di**2 + dj**2)))
                    factor = torch.exp(-0.5 * (dist / 2)**2)
                    self.velocity[ni, nj] += input_tensor[ni, nj] * factor * self.excitation

    def update_field_physics(self):

        self.field.data += self.velocity

        self.velocity *= self.stability

    def apply_semantic_matching(self, input_fingerprint):
        """
        Apply influence to memories.

        This version uses the enhanced fingerprints for improved pattern differentiation.
        """
        if not self.attractor_memory:
            return False

class EcliphraWithPersistentMemory(EcliphraFieldWithSemantics):
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5,
                 consolidated_capacity=20, persistence_file=None):
        super().__init__(field_dim, device, memory_capacity)

        # Long-term (updated- more stable)
        self.consolidated_memories = []
        self.consolidated_capacity = consolidated_capacity

        self.consolidation_threshold = 8  # before consolidation
        self.memory_lifetime = 1000  # before decay begins
        self.consolidation_cooldown = 50  # this is between consolidation attempts
        self.last_consolidation = 0

        self.persistence_file = persistence_file
        if persistence_file and os.path.exists(persistence_file):
            self.load_persistent_memories()

    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None):
        """Process input with potential matches from consolidated memory"""
        # Check consolidated first for matches
        consolidated_match = False
        if input_fingerprint is not None and self.consolidated_memories:
            consolidated_match = self.check_consolidated_memories(input_fingerprint)

        result = super().forward(input_tensor, input_fingerprint, input_pos)

        if (self.time_step.item() - self.last_consolidation > self.consolidation_cooldown):
            self.consolidate_memories()
            self.last_consolidation = self.time_step.item()

        # Occasionally save memories 
        if self.persistence_file and self.time_step.item() % 200 == 0:
            self.save_persistent_memories()

        result['consolidated_match'] = consolidated_match

        return result

    def check_consolidated_memories(self, input_fingerprint):
        best_match = None
        best_similarity = 0

        for memory in self.consolidated_memories:
            if 'fingerprint' not in memory:
                continue

            similarity = F.cosine_similarity(
                input_fingerprint.view(1, -1),
                memory['fingerprint'].view(1, -1)
            ).item()

            # Going to try to use a slightly lower threshold for consolidated memories
            if similarity > max(self.semantic_threshold - 0.05, 0.6) and similarity > best_similarity:
                best_match = memory
                best_similarity = similarity

        if best_match:
            # Apply influence 
            with torch.no_grad():
                pos = best_match['position']
                i, j = pos

                # Recall the pattern 
                i_min, i_max, j_min, j_max = best_match['patch_coords']
                self.field[i_min:i_max, j_min:j_max] += best_match['patch'] * 0.15

                # To create momentum
                self.velocity[i_min:i_max, j_min:j_max] += best_match['patch'] * 0.05

                # To track memory usage
                best_match['last_activation'] = self.time_step.item()
                best_match['activation_count'] = best_match.get('activation_count', 0) + 1

            return True

        return False

    def consolidate_memories(self):
        """Move frequently accessed memories to be stored"""
        candidates = []

        for entry in self.attractor_memory:
            # Does it qualify
            if entry.get('semantic_hits', 0) >= self.consolidation_threshold:
                consolidated = entry.copy()
                consolidated['consolidated_time'] = self.time_step.item()
                consolidated['decay_rate'] = 0.005  # need slower decay
                consolidated['activation_count'] = 0
                consolidated['importance'] = entry.get('strength', 0.5) * (1 + entry.get('semantic_hits', 0) / 10)

                # duplicates?
                if not any(self.is_similar_memory(consolidated, existing)
                          for existing in self.consolidated_memories):
                    candidates.append(consolidated)

        # Add new candidates 
        if candidates:
            self.consolidated_memories.extend(candidates)
            print(f"Consolidated {len(candidates)} new memories.")

            # removing least important memories
            if len(self.consolidated_memories) > self.consolidated_capacity:
                self.consolidated_memories.sort(key=lambda x: x.get('importance', 0), reverse=True)
                self.consolidated_memories = self.consolidated_memories[:self.consolidated_capacity]

    def is_similar_memory(self, memory1, memory2, threshold=0.85):
        """Check if two memories are similar enough to be considered duplicates"""
        if 'fingerprint' not in memory1 or 'fingerprint' not in memory2:
            return False

        similarity = F.cosine_similarity(
            memory1['fingerprint'].view(1, -1),
            memory2['fingerprint'].view(1, -1)
        ).item()

        return similarity > threshold

    def save_persistent_memories(self):
        """Save consolidated memories to disk using PyTorch serialization"""
        if not self.persistence_file:
            return

        try:
            # Making a dictionary to hold all the memory data
            save_data = {
                'timestamp': time.time(),
                'memory_count': len(self.consolidated_memories),
                'memories': self.consolidated_memories
            }

            torch.save(save_data, self.persistence_file)
            print(f"Saved {len(self.consolidated_memories)} memories to {self.persistence_file}")
        except Exception as e:
            print(f"Error saving memories: {e}")

    def load_persistent_memories(self):
        """Load consolidated memories from disk using PyTorch deserialization"""
        if not self.persistence_file or not os.path.exists(self.persistence_file):
            return

        try:
            save_data = torch.load(self.persistence_file)

            # Extract memories
            self.consolidated_memories = save_data.get('memories', [])

            # Gotta make sure tensors are on the correct device
            for memory in self.consolidated_memories:
                for key, value in memory.items():
                    if isinstance(value, torch.Tensor):
                        memory[key] = value.to(self.device)

            print(f"Loaded {len(self.consolidated_memories)} memories from {self.persistence_file}")
        except Exception as e:
            print(f"Error loading memories: {e}")

class EcliphraWithAdvancedMemory(EcliphraWithPersistentMemory):
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5,
                 consolidated_capacity=20, persistence_file=None):
        super().__init__(field_dim, device, memory_capacity, consolidated_capacity, persistence_file)

        # Memory contextualization
        self.context_window = []  # recent inputs/states form context
        self.context_window_size = 5
        self.context_embedding_dim = 16  # low dim embedding 

        # A small network for generating context embeddings
        self.context_encoder = nn.Sequential(
            nn.Linear(field_dim[0] * field_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, self.context_embedding_dim)
        ).to(device)

        # Memory pruning 
        self.pruning_interval = 200  
        self.relevance_threshold = 0.4  # Minimum relevance to keep
        self.last_pruning = 0

        # Memory integration 
        self.integration_interval = 300  
        self.integration_threshold = 0.8  # Similarity threshold
        self.last_integration = 0

        # Graphs hold relationships between memories
        self.memory_graph = {}  # Dictionary mapping memory IDs to related memories

        self.memory_id_counter = 0

    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None):
        # Update with current field state
        self.update_context_window()

        current_context = self.generate_context_embedding()

        result = super().forward(input_tensor, input_fingerprint, input_pos)

        current_time = self.time_step.item()

        # Pruning will run periodically
        if current_time - self.last_pruning > self.pruning_interval:
            pruned_count = self.prune_memories()
            self.last_pruning = current_time
            if pruned_count > 0:
                result['pruned_memories'] = pruned_count

        if current_time - self.last_integration > self.integration_interval:
            integrated_count = self.integrate_memories()
            self.last_integration = current_time
            if integrated_count > 0:
                result['integrated_memories'] = integrated_count

        # Attach any newly formed memories
        if self.attractors and input_tensor is not None:
            self.contextualize_memories(current_context)

        return result

    def update_context_window(self):
        if len(self.context_window) >= self.context_window_size:
            self.context_window.pop(0)

        self.context_window.append(self.field.detach().clone())

    def generate_context_embedding(self):
        # If context window is empty, return zeros
        if not self.context_window:
            return torch.zeros(self.context_embedding_dim, device=self.device)

        avg_field = sum(self.context_window) / len(self.context_window)

        with torch.no_grad():
            embedding = self.context_encoder(avg_field.flatten())
            return F.normalize(embedding, p=2, dim=0)

    def contextualize_memories(self, context_embedding):
        """Attach to new memories"""
        for entry in self.attractor_memory:
            if 'context' in entry:
                continue

            entry['context'] = context_embedding.clone().detach()

            entry['formation_time'] = self.time_step.item()

            # A unique ID for graph relationships
            entry['memory_id'] = self.memory_id_counter
            self.memory_id_counter += 1

            # Start up graph connections
            self.memory_graph[entry['memory_id']] = []

    def prune_memories(self):
        """Remove outdated or irrelevant memories"""
        if not self.consolidated_memories:
            return 0

        current_context = self.generate_context_embedding()

        # Finding the relevance scores for all memories
        memories_to_remove = []

        for memory in self.consolidated_memories:
            if 'context' not in memory:
                continue

            # Finding temporal relevance, newer being more relevant
            time_factor = max(0.2, min(1.0,
                              np.exp(-(self.time_step.item() - memory.get('formation_time', 0)) / self.memory_lifetime)))

            context_similarity = F.cosine_similarity(
                current_context.view(1, -1),
                memory['context'].view(1, -1)
            ).item() if 'context' in memory else 0.5

            # Usage relevance
            last_activation = memory.get('last_activation', 0)
            recency_factor = max(0.1, min(1.0,
                                np.exp(-(self.time_step.item() - last_activation) / 500)))

            relevance = 0.3 * time_factor + 0.4 * context_similarity + 0.3 * recency_factor

            # Mark for removal if below threshold
            if relevance < self.relevance_threshold:
                memories_to_remove.append(memory)
                if 'memory_id' in memory:
                    self.memory_graph.pop(memory['memory_id'], None)

        for memory in memories_to_remove:
            self.consolidated_memories.remove(memory)

        return len(memories_to_remove)

    def integrate_memories(self):
        """Merge similar memories to form more general concepts"""
        if len(self.consolidated_memories) < 2:
            return 0

        # Find pairs of similar memories
        integration_pairs = []

        for i, mem1 in enumerate(self.consolidated_memories[:-1]):
            for j, mem2 in enumerate(self.consolidated_memories[i+1:], i+1):
                # Skip if either memory lacks fingerprint
                if 'fingerprint' not in mem1 or 'fingerprint' not in mem2:
                    continue

                similarity = F.cosine_similarity(
                    mem1['fingerprint'].view(1, -1),
                    mem2['fingerprint'].view(1, -1)
                ).item()

                context_match = False
                if 'context' in mem1 and 'context' in mem2:
                    context_similarity = F.cosine_similarity(
                        mem1['context'].view(1, -1),
                        mem2['context'].view(1, -1)
                    ).item()
                    context_match = context_similarity > 0.7

                if similarity > self.integration_threshold or context_match:
                    integration_pairs.append((i, j, similarity))

        # Sort by similarity (highest first)
        integration_pairs.sort(key=lambda x: x[2], reverse=True)

        integrated = set()
        integration_count = 0

        # Process integration pairs
        for i, j, similarity in integration_pairs:
            if i in integrated or j in integrated:
                continue

            mem1 = self.consolidated_memories[i]
            mem2 = self.consolidated_memories[j]

            integrated_memory = self.create_integrated_memory(mem1, mem2)

            # Update graph relationships
            if 'memory_id' in mem1 and 'memory_id' in mem2:
                integrated_memory['memory_id'] = self.memory_id_counter
                self.memory_id_counter += 1

                # Merge 
                connections = set()
                connections.update(self.memory_graph.get(mem1['memory_id'], []))
                connections.update(self.memory_graph.get(mem2['memory_id'], []))
                self.memory_graph[integrated_memory['memory_id']] = list(connections)

                for node_id in self.memory_graph:
                    if node_id == integrated_memory['memory_id']:
                        continue

                    connections = self.memory_graph[node_id]
                    if mem1['memory_id'] in connections or mem2['memory_id'] in connections:
                        # Replacing old connections
                        new_connections = [c for c in connections
                                          if c != mem1['memory_id'] and c != mem2['memory_id']]
                        new_connections.append(integrated_memory['memory_id'])
                        self.memory_graph[node_id] = new_connections

            self.consolidated_memories.append(integrated_memory)

            integrated.add(i)
            integrated.add(j)
            integration_count += 1

        self.consolidated_memories = [mem for i, mem in enumerate(self.consolidated_memories)
                                    if i not in integrated]

        return integration_count

    def create_integrated_memory(self, mem1, mem2):
        """Combine two memories into a new integrated memory"""
        base_memory = mem1 if mem1.get('strength', 0) > mem2.get('strength', 0) else mem2
        integrated = base_memory.copy()

        # Blended fingerprint with more weight to stronger memory
        if 'fingerprint' in mem1 and 'fingerprint' in mem2:
            weight1 = mem1.get('strength', 0.5) / (mem1.get('strength', 0.5) + mem2.get('strength', 0.5))
            weight2 = 1 - weight1

            integrated['fingerprint'] = F.normalize(
                weight1 * mem1['fingerprint'] + weight2 * mem2['fingerprint'],
                p=2, dim=0
            )

        integrated['strength'] = max(mem1.get('strength', 0), mem2.get('strength', 0)) * 1.2
        integrated['importance'] = max(mem1.get('importance', 0), mem2.get('importance', 0)) * 1.1

        if 'context' in mem1 and 'context' in mem2:
            integrated['context'] = F.normalize(
                0.5 * mem1['context'] + 0.5 * mem2['context'],
                p=2, dim=0
            )

        # metadata
        integrated['integrated'] = True
        integrated['parent_ids'] = [
            mem1.get('memory_id', -1),
            mem2.get('memory_id', -1)
        ]
        integrated['integration_time'] = self.time_step.item()

        integrated['decay_rate'] = min(mem1.get('decay_rate', 0.01), mem2.get('decay_rate', 0.01)) * 0.8    # Now it should be less prone to decay

        # Semantic generalization which will increase basin size for integrated memories
        if len(mem1.get('attractors', [])) > 0 and len(mem2.get('attractors', [])) > 0:
            integrated['basin_size_multiplier'] = 1.5  # Need a larger basin of attraction for generalized concepts

        return integrated

    def find_related_memories(self, memory_id, depth=1):
        """Find memories related to a given memory based on graph connections"""
        if memory_id not in self.memory_graph:
            return []

        if depth <= 0:
            return []

        direct_connections = self.memory_graph[memory_id]

        all_connections = set(direct_connections)
        if depth > 1:
            for connected_id in direct_connections:
                deeper_connections = self.find_related_memories(connected_id, depth - 1)
                all_connections.update(deeper_connections)

        # Return memory objects instead of just IDs
        related_memories = []
        for memory in self.consolidated_memories:
            if 'memory_id' in memory and memory['memory_id'] in all_connections:
                related_memories.append(memory)

        return related_memories

    def update_memory_relationships(self):
        """Update relationships between memories based on co-activation"""
        recently_activated = []
        for memory in self.consolidated_memories:
            last_activation = memory.get('last_activation', 0)
            if self.time_step.item() - last_activation < 10: 
                if 'memory_id' in memory:
                    recently_activated.append(memory['memory_id'])

        # Connections between co-activated memories
        for i, id1 in enumerate(recently_activated[:-1]):
            for id2 in recently_activated[i+1:]:
                # bidirectional connections
                if id2 not in self.memory_graph[id1]:
                    self.memory_graph[id1].append(id2)
                if id1 not in self.memory_graph[id2]:
                    self.memory_graph[id2].append(id1)


class EcliphraFieldWithPhotoSynthesis(EcliphraFieldWithSemantics):
    """
    Model with photosynthesis-like signal processing.

    This combines the semantic capabilities of EcliphraFieldWithSemantics
    with the photosynthesis-inspired signal processing components.
    """

    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5, fingerprint_dim=768):
        super().__init__(field_dim, device, memory_capacity, fingerprint_dim)

        # photosynthesis components
        self.signal_router = SignalRouter(field_dim, device)
        self.threshold_layer = IntegrationThresholdLayer(device)
        self.rewriting_protocol = FieldRewritingProtocol(field_dim, device)

        self.memory_classifier = MemoryAugmentedClassifier(
            memory_capacity=memory_capacity * 4,  # Use 4x the attractor memory capacity
            feature_dim=fingerprint_dim,
            device=device
        )

        self.signal_processing_history = []

        # Need to override some physical parameters for better photosynthesis performance
        self.stability = nn.Parameter(torch.tensor(0.88, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.15, device=device))
        self.excitation = nn.Parameter(torch.tensor(0.4, device=device))


    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None):
        """
        Process input through photosynthesis field with semantic capabilities.

        Enhances the standard forward pass with signal classification and
        adaptive processing based on signal characteristics.
        """
        if input_fingerprint is None and input_tensor is not None:
            input_fingerprint = self.create_robust_fingerprint(input_tensor)

        # Store for comparison
        previous_field = self.field.detach().clone()

        if input_tensor is not None:
            signal_info = self.signal_router(
                input_tensor,
                self.field,
                self.velocity,
                self.attractors
            )

            signal_metrics = {
                'novelty': signal_info['novelty'],
                'resonance': signal_info['resonance'],
                'disruption': signal_info['disruption'],
                'stability': self.calculate_stability()
            }

            classification_result = self.memory_classifier(
                input_fingerprint,  # features
                signal_metrics,     
                signal_info['signal_type']  # original classification
            )

            signal_info['original_signal_type'] = signal_info['signal_type']
            signal_info['signal_type'] = classification_result['signal_type']
            signal_info['classification_confidence'] = classification_result['confidence']
            signal_info['was_revised'] = classification_result['was_revised']


            # threshold gating
            stability = self.calculate_stability()
            gated_signal, gate_info = self.threshold_layer(
                input_tensor,
                signal_info,
                stability,
                self.adaptivity if hasattr(self, 'adaptivity') else None
            )

            processing_summary = {
                'signal_type': signal_info['signal_type'],
                'original_type': signal_info.get('original_signal_type'),
                'was_revised': signal_info.get('was_revised', False),
                'confidence': signal_info.get('classification_confidence', 1.0),
                'pass_ratio': gate_info['pass_ratio'],
                'novelty': signal_info['novelty']
            }
            self.signal_processing_history.append(processing_summary)


            if signal_info['signal_type'] == "noise" and gate_info['pass_ratio'] < 0.3:
                semantic_match_applied, match_similarity = self.apply_semantic_matching(input_fingerprint)

                if semantic_match_applied:

                    self.update_field_physics()

                    # Track after update
                    if len(self.field_history) >= 10:
                        self.field_history.pop(0)
                    self.field_history.append(self.field.detach().clone())

                    # emergent attractors
                    self.detect_attractors()

                    field_diff = torch.norm(self.field - previous_field).item()
                    if len(self.field_diff_history) >= 10:
                        self.field_diff_history.pop(0)
                    self.field_diff_history.append(field_diff)

                    stability = self.calculate_stability()

                    return {
                        'field': self.field,
                        'attractors': self.attractors,
                        'stability': stability,
                        'signal_info': signal_info,
                        'gate_info': gate_info,
                        'classification_result': classification_result,
                        'semantic_match_applied': semantic_match_applied,
                        'match_similarity': match_similarity
                    }

            updated_field, rewrite_info = self.rewriting_protocol(
                self.field,
                gated_signal,
                signal_info,
                self.attractors
            )

            # Now with the rewritten result
            self.field.data = updated_field


            self.update_field_physics()

            # After update
            if len(self.field_history) >= 10:
                self.field_history.pop(0)
            self.field_history.append(self.field.detach().clone())

            self.detect_attractors()

            # For semantics of strong attractors
            if input_fingerprint is not None and len(self.attractors) > 0:
                self.associate_fingerprints(input_fingerprint)

            field_diff = torch.norm(self.field - previous_field).item()
            if len(self.field_diff_history) >= 10:
                self.field_diff_history.pop(0)
            self.field_diff_history.append(field_diff)

            stability = self.calculate_stability()

            return {
                'field': self.field,
                'attractors': self.attractors,
                'stability': stability,
                'signal_info': signal_info,
                'gate_info': gate_info,
                'rewrite_info': rewrite_info,
                'classification_result': classification_result,
                'semantic_match_applied': False
            }
        else:
            return super().forward(input_tensor, input_fingerprint, input_pos)


    def get_classification_stats(self):
        """Get statistics about the memory-augmented classification system"""
        revision_stats = self.memory_classifier.get_revision_stats()

        recent_types = [entry['signal_type'] for entry in self.signal_processing_history[-20:]]
        type_counts = {t: recent_types.count(t) for t in self.memory_classifier.signal_types}

        recent_revisions = [1 if entry.get('was_revised', False) else 0
                          for entry in self.signal_processing_history[-20:]]
        recent_revision_rate = sum(recent_revisions) / max(1, len(recent_revisions))

        avg_confidence = sum(entry.get('confidence', 0.0)
                            for entry in self.signal_processing_history[-20:]) / max(1, len(self.signal_processing_history[-20:]))

        return {
            'revision_stats': revision_stats,
            'recent_distribution': type_counts,
            'recent_revision_rate': recent_revision_rate,
            'avg_confidence': avg_confidence,
            'total_signals_processed': len(self.signal_processing_history),
            'memory_weight': self.memory_classifier.memory_weight.item(),
            'feature_weight': self.memory_classifier.feature_weight.item()
        }

    def apply_input(self, input_tensor, pos):
        """
        Override the apply_input method to use photosynthesis routing.
        This ensures the signal processing happens even when using the standard input application.
        """
        signal_info = self.signal_router(
            input_tensor,
            self.field,
            self.velocity,
            self.attractors
        )

        signal_metrics = {
            'novelty': signal_info['novelty'],
            'resonance': signal_info['resonance'],
            'disruption': signal_info['disruption'],
            'stability': self.calculate_stability()
        }

        classification_result = self.memory_classifier(
            input_fingerprint,  
            signal_metrics,     
            signal_info['signal_type'] 
        )

        signal_info['original_signal_type'] = signal_info['signal_type']
        signal_info['signal_type'] = classification_result['signal_type']
        signal_info['classification_confidence'] = classification_result['confidence']
        signal_info['was_revised'] = classification_result['was_revised']

         
        stability = self.calculate_stability()
        gated_signal, gate_info = self.threshold_layer(
            input_tensor,
            signal_info,
            stability,
            self.adaptivity if hasattr(self, 'adaptivity') else None
        )


        updated_field, rewrite_info = self.rewriting_protocol(
            self.field,
            gated_signal,
            signal_info,
            self.attractors
        )

        self.field.data = updated_field

        processing_summary = {
            'signal_type': signal_info['signal_type'],
            'pass_ratio': gate_info['pass_ratio'],
            'avg_influence': rewrite_info['avg_influence']
        }
        self.signal_processing_history.append(processing_summary)

    def get_photosynthesis_stats(self):
        """Return statistics about the photosynthesis process"""
        total_signals = torch.sum(self.signal_router.signal_stats).item()
        if total_signals == 0:
            signal_distribution = [0, 0, 0]
        else:
            signal_distribution = [
                self.signal_router.signal_stats[0].item() / total_signals,
                self.signal_router.signal_stats[1].item() / total_signals,
                self.signal_router.signal_stats[2].item() / total_signals
            ]

        # pass/block rates
        pass_rates = []
        for i in range(3):
            total = self.threshold_layer.pass_counts[i] + self.threshold_layer.block_counts[i]
            if total > 0:
                pass_rates.append(self.threshold_layer.pass_counts[i].item() / total)
            else:
                pass_rates.append(0)

        # recent trend
        if len(self.signal_processing_history) > 0:
            recent_types = [entry['signal_type'] for entry in self.signal_processing_history[-10:]]
            type_counts = {
                'noise': recent_types.count('noise'),
                'seed': recent_types.count('seed'),
                'challenge': recent_types.count('challenge')
            }
            dominant_type = max(type_counts, key=type_counts.get)
        else:
            dominant_type = "unknown"

        return {
            'signal_distribution': signal_distribution,  # noise, seed, challenge
            'pass_rates': pass_rates,  # noise, seed, challenge
            'dominant_recent_type': dominant_type,
            'total_processed': total_signals
        }


class EcliphraFieldWithEnergySystem(EcliphraFieldWithSemantics):
    """
    Model with energy-based photosynthesis principles.
    
    This model replaces classification-based signal processing with an
    energy allocation paradigm that manages energy flow, distribution,
    and field evolution through three key pathways: maintenance, 
    growth, and adaptation.
    """
    
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5, fingerprint_dim=768):
        super().__init__(field_dim, device, memory_capacity, fingerprint_dim)
        
        # Energy system components
        self.signal_analyzer = SignalAnalyzer(field_dim, device)
        self.energy_distributor = EnergyDistributor(device)
        self.field_modulator = FieldModulator(field_dim, device)
        
        self.energy_capacity = nn.Parameter(torch.ones(field_dim, device=device))
        self.energy_current = torch.zeros(field_dim, device=device)
        self.energy_flow_rates = torch.ones(3, device=device)  # For 3 pathways
        
        # Need to override some physical parameters for better energy-based performance
        self.stability = nn.Parameter(torch.tensor(0.88, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.15, device=device))
        self.excitation = nn.Parameter(torch.tensor(0.4, device=device))

        self.energy_history = []
        

    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None):
        """
        Process input through energy-based field with semantic capabilities.
        
        This combines the semantics of EcliphraFieldWithSemantics with
        the energy-based processing paradigm.
        """
        if input_fingerprint is None and input_tensor is not None:
            input_fingerprint = self.create_robust_fingerprint(input_tensor)
        
        previous_field = self.field.detach().clone()
        
        if input_tensor is not None:
            # Going to process through the energy system now
            
            signal_metrics = self.signal_analyzer(input_tensor, self.field)
    
            available_energy = self.calculate_available_energy()
            
            energy_distribution = self.energy_distributor(signal_metrics, available_energy)
            
            modulation_effects = self.field_modulator(self.field, energy_distribution)
            
            self.field.data = self.field.data + modulation_effects["field_delta"]
            
            self.update_energy_state(energy_distribution)
            
            semantic_match_applied = False
            match_similarity = 0.0
            
            if input_fingerprint is not None:
                semantic_match_applied, match_similarity = self.apply_semantic_matching(input_fingerprint)
                
                if len(self.attractors) > 0:
                    self.associate_fingerprints(input_fingerprint)
        else:
            # just check for echo resonance
            semantic_match_applied = False
            match_similarity = 0.0
            signal_metrics = None
            energy_distribution = None
            modulation_effects = None
            
            # if field is relatively quiet
            echo_applied = self.apply_echo_resonance()
            
        self.update_field_physics()
        
        if len(self.field_history) >= 10:
            self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())
        
        self.detect_attractors()
        
        field_diff = torch.norm(self.field - previous_field).item()
        if len(self.field_diff_history) >= 10:
            self.field_diff_history.pop(0)
        self.field_diff_history.append(field_diff)
  
        stability = self.calculate_stability()
        
        if input_tensor is not None:
            self.energy_history.append({
                'signal_metrics': signal_metrics,
                'energy_distribution': energy_distribution,
                'available_energy': available_energy,
                'modulation_effects': {k: v for k, v in modulation_effects.items() 
                                     if not isinstance(v, torch.Tensor)}
            })
        
        return {
            'field': self.field,
            'attractors': self.attractors,
            'stability': stability,
            'semantic_match_applied': semantic_match_applied,
            'match_similarity': match_similarity,
            'signal_metrics': signal_metrics,
            'energy_distribution': energy_distribution,
            'modulation_effects': modulation_effects if input_tensor is not None else None,
            'energy_level': torch.mean(self.energy_current).item()
        }
        
    def calculate_available_energy(self):
        """Calculate how much energy is available for allocation"""
        base_energy = torch.mean(self.energy_capacity).item()
        
        current_state = torch.mean(self.energy_current).item()
        
        available = max(0.1, base_energy - (current_state * 0.5))
        
        return available
    
    def update_energy_state(self, distribution):
        """Update internal energy state based on allocation"""
        total_allocated = sum(v for v in distribution.values() if isinstance(v, (int, float)))  # Energy consumption, safer version
        
        # Using a simple model where energy decreases based on usage and recovers slowly
        self.energy_current = self.energy_current * 0.9 + torch.ones_like(self.energy_current) * total_allocated * 0.3
        
        # To the lower bound
        self.energy_current = torch.clamp(self.energy_current, min=0.0)
        
        # To the upper bound 
        self.energy_current = torch.minimum(self.energy_current, self.energy_capacity)
        
    def detect_attractors(self):
        """Find emergent attractors in the field with energy-aware adaptive basin sizing"""
        h, w = self.field.shape
        smoothed = F.avg_pool2d(
            self.field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
        # Looking for local maxima 
        threshold = 0.15
        candidates = []
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = smoothed[i, j].item()
                if center < threshold:
                    continue
                
                neighborhood = smoothed[i-1:i+2, j-1:j+2]
                if center >= torch.max(neighborhood).item():
                    base_basin_size = self.calculate_basin(i, j)
                    
                    energy_multiplier = 1.0 + self.energy_current[i, j].item() * 0.5
                    
                    adaptive_basin_size = base_basin_size * energy_multiplier
                   
                    strength = center * adaptive_basin_size
                    
                    candidates.append(((i, j), strength, base_basin_size, adaptive_basin_size))
        
        # Keep top 3
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        self.attractors = [(pos, strength, base_size, adaptive_size)
                          for (pos, strength, base_size, adaptive_size) in candidates[:3]]
    
    def get_energy_statistics(self):
        if not self.energy_history:
            return {
                "avg_energy_level": 0.0,
                "pathway_distribution": {"maintenance": 0.33, "growth": 0.33, "adaptation": 0.34},
                "energy_stability": 0.5
            }
        
        recent_history = self.energy_history[-10:]
   
        energy_levels = [torch.mean(self.energy_current).item()]
        avg_energy = sum(energy_levels) / len(energy_levels)
        
        energy_stability = 1.0 - min(1.0, torch.std(torch.tensor(energy_levels)).item()) # how consistent energy levels are
        
        maintenance_vals = [h['energy_distribution'].get('maintenance', 0.0) for h in recent_history]
        growth_vals = [h['energy_distribution'].get('growth', 0.0) for h in recent_history]
        adaptation_vals = [h['energy_distribution'].get('adaptation', 0.0) for h in recent_history]
        
        total_energy = sum(maintenance_vals) + sum(growth_vals) + sum(adaptation_vals)
        if total_energy > 0:
            pathway_distribution = {
                "maintenance": sum(maintenance_vals) / total_energy,
                "growth": sum(growth_vals) / total_energy,
                "adaptation": sum(adaptation_vals) / total_energy
            }
        else:
            pathway_distribution = {"maintenance": 0.33, "growth": 0.33, "adaptation": 0.34}
        
        if recent_history:
            recent_metrics = {
                "avg_intensity": sum(h['signal_metrics'].get('intensity', 0.0) for h in recent_history) / len(recent_history),
                "avg_coherence": sum(h['signal_metrics'].get('coherence', 0.0) for h in recent_history) / len(recent_history),
                "avg_complexity": sum(h['signal_metrics'].get('complexity', 0.0) for h in recent_history) / len(recent_history)
            }
        else:
            recent_metrics = {"avg_intensity": 0.0, "avg_coherence": 0.0, "avg_complexity": 0.0}
        
        return {
            "avg_energy_level": avg_energy,
            "energy_stability": energy_stability,
            "pathway_distribution": pathway_distribution,
            "recent_metrics": recent_metrics,
            "most_recent_distribution": recent_history[-1]['energy_distribution'] if recent_history else {"maintenance": 0.0, "growth": 0.0, "adaptation": 0.0}
        }
    
    def visualize_energy_flow(self, output_dir=None):
        """
        Visualize energy flow and allocation through the system.
        
        Args:
            output_dir: Directory to save visualization (optional)
        
        Returns:
            Dict with paths to saved visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            return {"error": "Matplotlib not installed"}
        
        if not self.energy_history:
            return {"error": "No energy history to visualize"}
        
        # if needed
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # energy allocation over time
        plt.figure(figsize=(12, 6))
        
        steps = range(len(self.energy_history))
        maintenance = [h['energy_distribution'].get('maintenance', 0.0) for h in self.energy_history]
        growth = [h['energy_distribution'].get('growth', 0.0) for h in self.energy_history]
        adaptation = [h['energy_distribution'].get('adaptation', 0.0) for h in self.energy_history]
        
        plt.plot(steps, maintenance, 'b-', label='Maintenance Energy', linewidth=2)
        plt.plot(steps, growth, 'g-', label='Growth Energy', linewidth=2)
        plt.plot(steps, adaptation, 'r-', label='Adaptation Energy', linewidth=2)
        
        plt.xlabel('Steps')
        plt.ylabel('Energy Allocation')
        plt.title('Energy Allocation Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        allocation_plot_path = os.path.join(output_dir, 'energy_allocation.png') if output_dir else None
        if allocation_plot_path:
            plt.savefig(allocation_plot_path)
        plt.close()
        
        # energy distribution in field
        plt.figure(figsize=(10, 8))
        
        energy_cmap = LinearSegmentedColormap.from_list('Energy', [
            (0, 'darkblue'), (0.4, 'royalblue'),
            (0.6, 'orange'), (0.8, 'crimson'), (1.0, 'gold')
        ])
        
        im = plt.imshow(self.energy_current.detach().cpu().numpy(), cmap=energy_cmap)
        plt.colorbar(im, label='Energy Level')
 
        for attractor in self.attractors:
            if len(attractor) >= 2:
                pos = attractor[0]
                strength = attractor[1]
                plt.scatter(pos[1], pos[0], c='white', s=100*strength + 50, marker='*',
                           edgecolors='black', linewidths=1)
        
        plt.title('Energy Distribution in Field')
        
        energy_map_path = os.path.join(output_dir, 'energy_map.png') if output_dir else None
        if energy_map_path:
            plt.savefig(energy_map_path)
        plt.close()
        
        # signal metrics
        plt.figure(figsize=(12, 6))

        intensity = [h['signal_metrics'].get('intensity', 0.0) for h in self.energy_history]
        coherence = [h['signal_metrics'].get('coherence', 0.0) for h in self.energy_history]
        complexity = [h['signal_metrics'].get('complexity', 0.0) for h in self.energy_history]
        
        plt.plot(steps, intensity, 'r-', label='Signal Intensity', linewidth=2)
        plt.plot(steps, coherence, 'g-', label='Signal Coherence', linewidth=2)
        plt.plot(steps, complexity, 'b-', label='Signal Complexity', linewidth=2)
        
        plt.xlabel('Steps')
        plt.ylabel('Metric Value')
        plt.title('Signal Energy Metrics Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        metrics_plot_path = os.path.join(output_dir, 'signal_metrics.png') if output_dir else None
        if metrics_plot_path:
            plt.savefig(metrics_plot_path)
        plt.close()
        
        return {
            "allocation_plot": allocation_plot_path,
            "energy_map": energy_map_path,
            "metrics_plot": metrics_plot_path
        }
    
    def reset(self):
        """Reset field state"""
        super().reset()  
        self.energy_current.zero_()
        self.energy_history = []

class EcliphraFieldWithPrefrontal(EcliphraFieldWithEnergySystem):
    """
    Model mixed with energy-based photosynthesis and prefrontal control.
    
    This model adds prefrontal cortex-inspired executive function capabilities 
    to the energy-based architecture.
    """
    
    def __init__(self, field_dim=(32, 32), device='cpu', memory_capacity=5, 
                 working_memory_size=5, max_goals=3):
        super().__init__(field_dim, device, memory_capacity)
        
        self.prefrontal = PrefrontalModule(
            field_dim=field_dim,
            device=device,
            working_memory_size=working_memory_size,
            max_goals=max_goals
        )

        def set_goal(self, goal_data, priority=1.0):
            """Set a goal in the prefrontal module"""
            print(f"EcliphraFieldWithPrefrontal attempting to set goal: {goal_data}")
            goal_id = self.prefrontal.set_goal(goal_data, priority)
            print(f"Goal set with ID {goal_id}, total goals: {len(self.prefrontal.current_goals)}")
            return goal_id
            
        self.stability = nn.Parameter(torch.tensor(0.85, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.12, device=device))

        self.cognitive_metrics = {
            "goal_alignment": [],
            "inhibition_efficiency": [],
            "resource_efficiency": []
        }

        self.debug_mode = True  # make it configurable?

        
    
    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None, 
            goal_data=None, prefrontal_control=True):
        """
        Process input through energy-based field with prefrontal control.
        
        Args:
            input_tensor: Optional input 
            input_fingerprint: Optional semantics
            input_pos: Optional to apply input
            goal_data: Optional goal to set
            prefrontal_control: Whether to apply prefrontal control
            
        Returns:
            Dict with field state and metrics
        """
        # Needs to be through energy system first
        result = super().forward(input_tensor, input_fingerprint, input_pos)

        if prefrontal_control and hasattr(self, 'prefrontal'):
            if hasattr(self.prefrontal, 'fatigue_level'):
                result['fatigue_level'] = self.prefrontal.fatigue_level
                
            if hasattr(self.prefrontal, 'get_fatigue_status'):
                fatigue_status = self.prefrontal.get_fatigue_status()
                if 'debug_info' in fatigue_status:
                    debug_info = fatigue_status['debug_info'] # checking fatigue integration
                    result['fatigue_components'] = {
                        'goal_intensity': debug_info.get('goal_intensity', 0.0),
                        'shift_magnitude': debug_info.get('shift_magnitude', 0.0),
                        'residual_meaning': debug_info.get('residual_meaning', 0.0)
                    }
        
        if prefrontal_control and hasattr(self, 'prefrontal'):
            energy_distribution = result.get('energy_distribution', None)
            signal_metrics = result.get('signal_metrics', None)

            modulated_field, modified_distribution = self.prefrontal(
                self.field, energy_distribution, signal_metrics
            )
            
            self.field.data = modulated_field
            
            if 'energy_distribution' in result:
                result['energy_distribution'] = modified_distribution
                # Diagnostic print
                print(f"[ENERGY SCALAR APPLIED] {modified_distribution.get('total', 1.0):.4f}")
                
            if input_tensor is not None:
                self.prefrontal.update_working_memory(input_tensor)
                
            pf_status = self.prefrontal.get_status()
            result['prefrontal_status'] = pf_status
            
            outcome_metrics = {
                "stability": result.get('stability', 0.5),
                "energy_level": result.get('energy_level', 0.0),
                "goal_progress": 0.1  # Default, could be more sophisticated
            }
            self.prefrontal.evaluate_outcome(outcome_metrics)
            
            self._update_cognitive_metrics(result)
        
        return result

    def apply_energy_weights(self, distribution):
        if hasattr(self, 'energy_current'):
            if self.energy_current.dim() == 2:
                # Fallback to scalar modulation if field is 2D
                scalar = sum([
                    distribution.get('maintenance', 0.0),
                    distribution.get('growth', 0.0),
                    distribution.get('adaptation', 0.0)
                ])
                print(f"[ENERGY SCALAR APPLIED] {scalar:.4f}")
                self.energy_current *= scalar
            else:
                weights = torch.tensor([
                    distribution.get('maintenance', 1.0),
                    distribution.get('growth', 1.0),
                    distribution.get('adaptation', 1.0)
                ], device=self.energy_current.device)
                self.energy_current *= weights.view(-1, 1, 1)
                print(f"[ENERGY WEIGHTS APPLIED] {weights.cpu().numpy()}")

    def get_cognitive_metrics(self):
        if hasattr(self, 'prefrontal'):
            status = self.prefrontal.get_status()
            if 'cognitive_metrics' in status:
                return status['cognitive_metrics']
        
        return {        # Fallback to tracked metrics if prefrontal doesn't have them
            "goal_alignment": 0.05,
            "inhibition_efficiency": 0.05,
            "resource_efficiency": 0.05
        }

    # goal alignment 
    def _update_cognitive_metrics(self, result):
        """Directly update cognitive metrics in prefrontal module"""
        if not hasattr(self, 'prefrontal'):
            return
        
        if not hasattr(self.prefrontal, 'cognitive_metrics'):
            self.prefrontal.cognitive_metrics = {
                "goal_alignment": 0.3,
                "inhibition_efficiency": 0.4,
                "resource_efficiency": 0.5
            }
        
      
        pf_status = self.prefrontal.get_status()  # based on goals
        goals = pf_status.get('goals', [])
        
        if goals:
            goal_types = [g.get('type', 'generic') for g in goals]
 
            if 'stability' in goal_types and result.get('stability', 0) > 0.7:
                type_alignment = 0.8  # High for stability 
            elif 'energy' in goal_types and result.get('energy_level', 0) > 0.6:
                type_alignment = 0.7  # Good for energy 
            else:
                type_alignment = 0.4  # Default 
            
  
            alignment = (sum(g.get('priority', 0.0) * g.get('progress', 0.1) for g in goals) / 
                        sum(g.get('priority', 1.0) for g in goals)) * type_alignment
            
            if g['progress'] < 0.3 and g['priority'] > 0.7:
                 log("Goal with high priority is lagging.")

            
            
            self.prefrontal.cognitive_metrics["goal_alignment"] = max(0.1, alignment)    # allowing it to vary above the minimum
        
        stability = result.get('stability', 0.5)
        self.prefrontal.cognitive_metrics["inhibition_efficiency"] = stability * 0.6
        
        energy_level = result.get('energy_level', 0.5)
        self.prefrontal.cognitive_metrics["resource_efficiency"] = energy_level * 0.8
    
    def reset(self):
        """Reset field state and prefrontal module"""
        if hasattr(self, 'prefrontal') and hasattr(self.prefrontal, 'current_goals'):
            print(f"RESET: Clearing {len(self.prefrontal.current_goals)} goals")
        
        super().reset()
        
        if hasattr(self, 'prefrontal'):
            self.prefrontal.reset()
            
    def set_goal(self, goal_data, priority=1.0):
        return self.prefrontal.set_goal(goal_data, priority)
    
    def inhibit_process(self, process_id=None, field_region=None):
        self.prefrontal.inhibit_process(process_id, field_region)
    
    def focus_attention(self, field_region=None, pathway_bias=None):
        self.prefrontal.focus_attention(field_region, pathway_bias)
    
    def get_cognitive_metrics(self):
        return {
            key: np.mean(values) if values else 0.0
            for key, values in self.cognitive_metrics.items()
        }
    
    """
    This is a direct fix  to ensure cognitive metrics and energy distribution work properly.
    Should condense when project finishes.
    """
    def debug_print(self, message):
        print(f"[DEBUG] {message}")

    def forward(self, input_tensor=None, input_fingerprint=None, input_pos=None, # fixed
                goal_data=None, prefrontal_control=True):
        """
        Process input through energy-based field with prefrontal control.
        
        Args:
            input_tensor: Optional input 
            input_fingerprint: Optional semantics
            input_pos: Optional apply input
            goal_data: Optional goal to set
            prefrontal_control: Whether to apply prefrontal control
            
        Returns:
            Dict with field state and metrics
        """
        print(f"Prefrontal module has {len(self.prefrontal.current_goals)} goals at start of forward")

        self.debug_print(f"Starting forward pass, prefrontal_control={prefrontal_control}")
        
        if goal_data is not None and hasattr(self, 'set_goal'):
            priority = goal_data.get('priority', 1.0) if isinstance(goal_data, dict) else 1.0
            self.set_goal(goal_data, priority)
            self.debug_print(f"Set goal: {goal_data}")
        
        result = super().forward(input_tensor, input_fingerprint, input_pos)
        self.debug_print(f"Energy system forward complete, result keys: {list(result.keys())}")

        if 'energy_distribution' not in result: # More aggressive
            result['energy_distribution'] = {
                "maintenance": 0.2,
                "growth": 0.15,
                "adaptation": 0.15,
                "total": 0.5
            }
        
        if 'signal_metrics' not in result:
            result['signal_metrics'] = {
                "coherence": 0.6,
                "intensity": 0.5,
                "complexity": 0.4
            }
        
        if prefrontal_control and hasattr(self, 'prefrontal'):
            self.debug_print("Applying prefrontal control")
            
            energy_distribution = result.get('energy_distribution')
            signal_metrics = result.get('signal_metrics')
            
            try:
                modulated_field, modified_distribution = self.prefrontal(
                    self.field, energy_distribution, signal_metrics
                )
                
                self.debug_print(f"Prefrontal returned field diff: {torch.norm(modulated_field - self.field).item()}")
                self.debug_print(f"Modified distribution: {modified_distribution}")
                
                self.field.data = modulated_field

                if 'energy_distribution' in result:
                    result['energy_distribution'] = modified_distribution
                    
                if input_tensor is not None:
                    self.prefrontal.update_working_memory(input_tensor)
                    
                self._update_cognitive_metrics(result)
                
                result['prefrontal_status'] = self.prefrontal.get_status()
                
                outcome_metrics = {
                    "stability": result.get('stability', 0.5),
                    "energy_level": result.get('energy_level', 0.0),
                    "goal_progress": min(1.0, max(0.0, 
                          result.get('stability', 0.5) * 0.5 + 
                          result.get('energy_level', 0.0) * 0.5))
                }
                self.prefrontal.evaluate_outcome(outcome_metrics)
                
            except Exception as e:
                self.debug_print(f"Error in prefrontal control: {e}")
                # Continue with regular processing if prefrontal fails
        
        # FORCE UPDATE
        if hasattr(self, 'energy_current'):
            energy_current = self.field.detach().clone().abs() * 0.8 + 0.2
            energy_current = (energy_current - energy_current.min()) / (energy_current.max() - energy_current.min() + 1e-8)
            energy_current = energy_current + torch.randn_like(energy_current) * 0.05
            self.energy_current = energy_current
        
        result['cognitive_metrics'] = self.get_cognitive_metrics()
        
        return result
    
    # fatigue controller
    def get_cognitive_stats(self):
        """Get statistics including fatigue"""
        stats = super().get_cognitive_stats()
        
        if hasattr(self, 'prefrontal') and hasattr(self.prefrontal, 'fatigue_level'):
            stats['fatigue_level'] = self.prefrontal.fatigue_level

            if 'cognitive_metrics' in stats:
                fatigue_factor = min(0.5, self.prefrontal.fatigue_level / 20.0)

                stats['cognitive_metrics']['fatigue_level'] = self.prefrontal.fatigue_level
      
                if 'resource_efficiency' in stats['cognitive_metrics']:
                    stats['cognitive_metrics']['resource_efficiency'] *= (1.0 - fatigue_factor)
                
                if 'goal_alignment' in stats['cognitive_metrics'] and self.prefrontal.fatigue_level > 15.0:  # High fatigue can make goal achievement harder
                    stats['cognitive_metrics']['goal_alignment'] *= (1.0 - fatigue_factor * 0.3) 
            
            if hasattr(self.prefrontal, 'fatigue_controller') and hasattr(self.prefrontal.fatigue_controller, 'debug_info'):
                debug_info = self.prefrontal.fatigue_controller.debug_info
                stats['fatigue_components'] = {
                    'goal_intensity': debug_info.get('goal_intensity', 0.0),
                    'shift_magnitude': debug_info.get('shift_magnitude', 0.0),
                    'residual_meaning': debug_info.get('residual_meaning', 0.0),
                    'completed_goals': len(self.prefrontal.fatigue_controller.goal_completion_history)
                }
        
        return stats

    def get_cognitive_metrics(self):
        """Get with forced fallback values"""
        metrics = {
            "goal_alignment": 0.3,
            "inhibition_efficiency": 0.4,
            "resource_efficiency": 0.5
        }
        
        if hasattr(self, 'prefrontal'):
            pf_status = self.prefrontal.get_status()
            pf_metrics = pf_status.get('cognitive_metrics', {})

            for key, value in pf_metrics.items():
                if value > 0.001: 
                    metrics[key] = value
        
        self.debug_print(f"Returning cognitive metrics: {metrics}")
        return metrics

    def _update_cognitive_metrics(self, result):
        if not hasattr(self, 'prefrontal'):
            return
        
        if not hasattr(self.prefrontal, 'cognitive_metrics'):
            self.prefrontal.cognitive_metrics = {
                "goal_alignment": 0.3,
                "inhibition_efficiency": 0.4,
                "resource_efficiency": 0.5
            }
      
        pf_status = self.prefrontal.get_status()
        goals = pf_status.get('goals', [])
        if goals:
            alignment = sum(g.get('priority', 0.0) * g.get('progress', 0.1) 
                        for g in goals) / sum(g.get('priority', 1.0) for g in goals)
            self.prefrontal.cognitive_metrics["goal_alignment"] = max(alignment, 0.1)
        
        stability = result.get('stability', 0.5)
        self.prefrontal.cognitive_metrics["inhibition_efficiency"] = stability * 0.6

        energy_level = result.get('energy_level', 0.5)
        self.prefrontal.cognitive_metrics["resource_efficiency"] = energy_level * 0.8


   

from ecliphra.utils.enhanced_fatigue_controller import EnhancedFatigueController

# model factory function
def create_ecliphra_model(model_type='semantics', field_dim=(32, 32), device='cpu', 
                         memory_capacity=5, prefrontal=False, working_memory_size=5, max_goals=3,
                         enhanced_fatigue=False, fatigue_recovery=0.4, fatigue_decay=0.98):
    """
    Factory function to create Ecliphra models with specific capabilities.

    Args:
        model_type: Type of model to create ('base', 'echo', 'semantics', 'photo', 'energy')
        field_dim: Dimensions of the field (height, width)
        device: Device to run on ('cpu' or 'cuda')
        memory_capacity: Memory capacity for echo/semantic models
        prefrontal: Whether to add prefrontal capabilities
        working_memory_size: Size for prefrontal module
        max_goals: Maximum number of concurrent goals
        enhanced_fatigue: Whether to use enhanced fatigue controller
        fatigue_recovery: Recovery factor for goal completion (0-1)
        fatigue_decay: Natural decay rate of fatigue (0-1)

    Returns:
        Instantiated model of the requested type
    """
    if prefrontal and model_type == 'energy':
        fatigue_controller = None
        if enhanced_fatigue:
            print("Using enhanced fatigue controller...")
            fatigue_controller = EnhancedFatigueController(
                novelty_threshold=0.2,
                drift_threshold=0.5,
                fatigue_steps=5,
                recovery_factor=fatigue_recovery,
                decay_rate=fatigue_decay
            )

        # prefrontal module with fatigue controller
        prefrontal_module = PrefrontalModule(
            field_dim=field_dim,
            device=device,
            working_memory_size=working_memory_size,
            max_goals=max_goals,
            fatigue_controller=fatigue_controller
        )

        # model with attached prefrontal
        model = EcliphraFieldWithPrefrontal(
            field_dim=field_dim,
            device=device,
            memory_capacity=memory_capacity,
            working_memory_size=working_memory_size,
            max_goals=max_goals,
            enhanced_fatigue=enhanced_fatigue,
            fatigue_recovery=fatigue_recovery,
            fatigue_decay=fatigue_decay
        )
        model.prefrontal = prefrontal_module
        return model

    if model_type == 'base':
        return EcliphraField(field_dim=field_dim, device=device)
    elif model_type == 'echo':
        return EcliphraFieldWithEcho(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'semantics':
        return EcliphraFieldWithSemantics(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'energy':
        return EcliphraFieldWithEnergySystem(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'photo':
        model = EcliphraFieldWithPhotoSynthesis(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
        model.memory_classifier.memory_capacity = memory_capacity * 4  # 4x larger for more history
        return model
    elif model_type == 'legacy':
        return LegacyIdentityField(field_size=field_dim[0], device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class LegacyIdentityField(nn.Module):
    """Legacy implementation for comparison experiments"""
    def __init__(self, field_size=32, device='cpu'):
        super().__init__()
        self.device = device
        self.field_size = field_size

        # Core field tensor
        self.field = nn.Parameter(torch.zeros((field_size, field_size), device=device))
        self._initialize_field()

        # Parameters
        self.drift_factor = nn.Parameter(torch.tensor(0.1, device=device))
        self.resonance_scale = nn.Parameter(torch.tensor(0.2, device=device))

        # Attractor tracking (fixed count)
        self.register_buffer('attractor_centers', torch.zeros(3, 2, device=device))
        self.register_buffer('attractor_strengths', torch.zeros(3, device=device))

        # History for metacognitive
        self.field_history = []

    def _initialize_field(self):
        """Initialize with spiral pattern"""
        for i in range(self.field_size):
            for j in range(self.field_size):
                # The spiral pattern with phase relationships
                r = np.sqrt((i - self.field_size/2)**2 + (j - self.field_size/2)**2) / self.field_size
                theta = np.arctan2(j - self.field_size/2, i - self.field_size/2)
                self.field.data[i, j] = r * np.sin(theta + r * np.pi * 4)

        self.field.data /= torch.norm(self.field.data) + 1e-9

    def forward(self, input_tensor):
        # simplified
        input_projection = input_tensor.view(-1, self.field_size, self.field_size)

        previous_field = self.field.clone()

        resonance = input_projection.mean(dim=0) * self.resonance_scale

        # The gradients for drift
        grad_x = torch.zeros_like(self.field)
        grad_y = torch.zeros_like(self.field)

        # Approximate gradients
        grad_x[:, :-1] = self.field[:, 1:] - self.field[:, :-1]
        grad_y[:-1, :] = self.field[1:, :] - self.field[:-1, :]

        drift = self.drift_factor * (grad_x + grad_y)

        self.field.data = self.field.data + resonance + drift

        self.field.data = self.field.data / (torch.norm(self.field.data) + 1e-9)  # to prevent explosion

        if len(self.field_history) >= 10:
            self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())

        self._update_attractors(previous_field)

        return {
            'field': self.field,
            'stability': self.calculate_stability(previous_field)
        }

    def _update_attractors(self, previous_field):
        """Track attractors in the field"""
        field = self.field.detach()

        smoothed = F.avg_pool2d(
            field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()

        # Find peaks
        threshold = 0.05
        maxima = []

        for i in range(1, self.field_size-1):
            for j in range(1, self.field_size-1):
                center = smoothed[i, j].item()
                if center < threshold:
                    continue

                neighborhood = smoothed[i-1:i+2, j-1:j+2]
                if center >= torch.max(neighborhood).item():
                    maxima.append((i, j, field[i, j].item()))

        maxima.sort(key=lambda x: x[2], reverse=True)

        for idx, (i, j, strength) in enumerate(maxima[:3]):
            if idx >= len(self.attractor_centers):
                break
            self.attractor_centers[idx, 0] = i
            self.attractor_centers[idx, 1] = j
            self.attractor_strengths[idx] = strength

    def calculate_stability(self, previous_field):
        """Finding inverse of change magnitude"""
        diff = torch.norm(self.field - previous_field).item()
        return 1.0 / (1.0 + diff)


# Version tracking
VERSIONS = {
    'base': '1.0',          # Original EcliphraField
    'echo': '1.1',          # Added echo resonance
    'semantics': '1.2',     # Added semantic fingerprinting
    'enhanced fingerprinting': '1.3',     # Added enhanced version of semantic fingerprinting
    'memory': '1.4',        # Added memory capabilities
    'photo': '1.5',         # Added photosynthesis inspired class
    'energy': '1.6',        # Added energy system inspired by photosynthesis 
    'prefrontal': '1.7'     # Added goal oriented model
}

# Model factory function for easy instantiation
def create_ecliphra_model(model_type='semantics', field_dim=(32, 32), device='cpu', 
                         memory_capacity=5, prefrontal=False, working_memory_size=5, max_goals=3,
                         enhanced_fatigue=False, fatigue_recovery=0.4, fatigue_decay=0.98):
    """
    Factory function to create Ecliphra models with specific capabilities.

    Args:
        model_type: Type of model to create ('base', 'echo', 'semantics')
        field_dim: Dimensions of the field (height, width)
        device: Device to run on ('cpu' or 'cuda')
        memory_capacity: Memory capacity for echo/semantic models

    Returns:
        Instantiated model of the requested type
    """
 # If prefrontal is requested and model_type is 'energy', will create a prefrontal version
    if prefrontal and model_type == 'energy':
        # If fatigue controller if requested
        fatigue_controller = None
        if enhanced_fatigue:
            print("Using enhanced fatigue controller...")
            fatigue_controller = EnhancedFatigueController(
                novelty_threshold=0.2,
                drift_threshold=0.5,
                fatigue_steps=5,
                recovery_factor=fatigue_recovery,
                decay_rate=fatigue_decay
            )

        # with fatigue controller
        prefrontal_module = PrefrontalModule(
            field_dim=field_dim,
            device=device,
            working_memory_size=working_memory_size,
            max_goals=max_goals,
            fatigue_controller=fatigue_controller
        )

        #  model with attached prefrontal
        model = EcliphraFieldWithPrefrontal(
            field_dim=field_dim,
            device=device,
            memory_capacity=memory_capacity,
            working_memory_size=working_memory_size,
            max_goals=max_goals
        )
        model.prefrontal = prefrontal_module
        return model

    if model_type == 'base':
        return EcliphraField(field_dim=field_dim, device=device)
    elif model_type == 'echo':
        return EcliphraFieldWithEcho(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'semantics':
        return EcliphraFieldWithSemantics(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'energy':
        return EcliphraFieldWithEnergySystem(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
    elif model_type == 'photo':
        model = EcliphraFieldWithPhotoSynthesis(field_dim=field_dim, device=device, memory_capacity=memory_capacity)
        model.memory_classifier.memory_capacity = memory_capacity * 4  # 4x larger for more history
        return model
    elif model_type == 'legacy':
        return LegacyIdentityField(field_size=field_dim[0], device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
