"""
Photosynthesis Inspired Energy System

This version replaces EcliphraPhotoField with a new model 
that handles internal dynamics through energy flow, not classification.
Its meant to simulate how limited internal resources affect memory, growth, and modulation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from ecliphra.prefrontal import PrefrontalModule


class SignalAnalyzer(nn.Module):
    """
    Looks to see how strong, messy, or unusual a signal is without caring what its supposed to be.
    This helps the model react to energy shifts like a nervous system would, not a classifier.
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim

        # Signal energy characteristics
        self.intensity_weight = nn.Parameter(torch.tensor(0.4, device=device))
        self.coherence_weight = nn.Parameter(torch.tensor(0.3, device=device))
        self.complexity_weight = nn.Parameter(torch.tensor(0.3, device=device))

        # Spectral analysis 
        self.frequency_bands = nn.Parameter(torch.tensor([0.2, 0.5, 0.8], device=device))
        
        # For the energy patterns
        self.signal_history = []
        self.history_capacity = 10
        
        # Energy signal using 3 pathways for now. Should I use more?
        self.register_buffer('energy_distribution', torch.zeros(3, device=device))  # maintenance, growth, adaptation

    def forward(self, input_signal: torch.Tensor, field_state: torch.Tensor) -> Dict:
        """
        Analyze energy characteristics of an incoming signal.

        Args:
            input_signal: The incoming signal tensor
            field_state: The current state 

        Returns:
            Dict containing energy metrics and characteristics
        """
        intensity = self.calculate_intensity(input_signal)
        coherence = self.calculate_coherence(input_signal, field_state)
        complexity = self.calculate_complexity(input_signal)
        
        spectral_info = self.analyze_spectral_components(input_signal)
        
        energy_potential = (
            self.intensity_weight * intensity +
            self.coherence_weight * coherence +
            self.complexity_weight * complexity
        )
        
        # To find affinity for different energy pathways
        maintenance_affinity = coherence * 0.7 + intensity * 0.3
        growth_affinity = intensity * 0.6 + complexity * 0.4
        adaptation_affinity = complexity * 0.7 + (1 - coherence) * 0.3

        self.update_history(input_signal, {
            'intensity': intensity.item(),
            'coherence': coherence.item(),
            'complexity': complexity.item()
        })
        
        # Getting energy analysis
        return {
            "energy_potential": energy_potential.item(),
            "intensity": intensity.item(),
            "coherence": coherence.item(),
            "complexity": complexity.item(),
            "spectral_info": spectral_info,
            "pathway_affinities": {
                "maintenance": maintenance_affinity.item(),
                "growth": growth_affinity.item(),
                "adaptation": adaptation_affinity.item()
            }
        }

    def calculate_intensity(self, input_signal: torch.Tensor) -> torch.Tensor:
        # Measure overall magnitude and variance
        magnitude = torch.norm(input_signal)
        variance = torch.var(input_signal)
        
        # 0-1 range with non-linear scaling
        intensity = torch.tanh(magnitude * 0.1) * 0.7 + torch.tanh(variance * 10) * 0.3
        return intensity

    def calculate_coherence(self, input_signal: torch.Tensor, field_state: torch.Tensor) -> torch.Tensor:
        """Calculate how coherent the signal is with the current field state"""

        alignment = F.cosine_similarity(
            input_signal.view(-1).unsqueeze(0),
            field_state.view(-1).unsqueeze(0)
        )
        

        # Simplifing the structural similarity or pattern coherence using spatial gradients
        input_grad_x = torch.abs(input_signal[:, 1:] - input_signal[:, :-1]).mean()
        input_grad_y = torch.abs(input_signal[1:, :] - input_signal[:-1, :]).mean()
        
        field_grad_x = torch.abs(field_state[:, 1:] - field_state[:, :-1]).mean()
        field_grad_y = torch.abs(field_state[1:, :] - field_state[:-1, :]).mean()
        
        # Lower difference means stronger similiarty 
        grad_diff_x = torch.abs(input_grad_x - field_grad_x)
        grad_diff_y = torch.abs(input_grad_y - field_grad_y)
        structural_similarity = 1.0 - torch.tanh((grad_diff_x + grad_diff_y) * 5.0)
        
        coherence = alignment * 0.6 + structural_similarity * 0.4
        return coherence

    def calculate_complexity(self, input_signal: torch.Tensor) -> torch.Tensor:
        # Spatial frequency analysis, well...approximately
        # When theres higher spatial frequency it means more complexity
        avg_local_variance = 0.0
        kernel_size = 3
        
        # Not sure this is the best way to measure instability, but it gives decent signal turbulence
        padded = F.pad(input_signal.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='reflect')
        for i in range(kernel_size):
            for j in range(kernel_size):
                window = padded[0, 0, i:i+self.field_dim[0], j:j+self.field_dim[1]]
                avg_local_variance += torch.var(window)
        
        avg_local_variance /= (kernel_size * kernel_size)
        
        # For entropy
        patches = input_signal.unfold(0, 2, 1).unfold(1, 2, 1).reshape(-1, 4)
        patch_means = torch.mean(patches, dim=1)
        patch_vars = torch.var(patches, dim=1)
        entropy_approx = torch.mean(patch_vars / (patch_means + 1e-6))
        
        complexity = torch.tanh(avg_local_variance * 20) * 0.5 + torch.tanh(entropy_approx * 10) * 0.5
        return complexity
    
    def analyze_spectral_components(self, input_signal: torch.Tensor) -> Dict:
        # 2D FFT 
        fft = torch.fft.rfft2(input_signal)
        fft_mag = torch.abs(fft)
        
        total_energy = torch.sum(fft_mag)
        low_freq = torch.sum(fft_mag[:int(self.field_dim[0]*0.2), :int(self.field_dim[1]*0.2)])
        mid_freq = torch.sum(fft_mag[int(self.field_dim[0]*0.2):int(self.field_dim[0]*0.6), 
                                     int(self.field_dim[1]*0.2):int(self.field_dim[1]*0.6)])
        high_freq = total_energy - low_freq - mid_freq
        
        if total_energy > 0:
            low_freq_ratio = (low_freq / total_energy).item()
            mid_freq_ratio = (mid_freq / total_energy).item()
            high_freq_ratio = (high_freq / total_energy).item()
        else:
            low_freq_ratio, mid_freq_ratio, high_freq_ratio = 0.33, 0.33, 0.34
            
        return {
            "low_frequency": low_freq_ratio,
            "mid_frequency": mid_freq_ratio,
            "high_frequency": high_freq_ratio,
            "dominant_frequency": "low" if low_freq_ratio > max(mid_freq_ratio, high_freq_ratio) else
                                 "mid" if mid_freq_ratio > high_freq_ratio else "high"
        }

    def update_history(self, signal: torch.Tensor, energy_metrics: Dict):
        self.signal_history.append((signal.detach().clone(), energy_metrics))
        
        if len(self.signal_history) > self.history_capacity:
            self.signal_history.pop(0)

# Updated version to include contradiction signals V2.0
class EnergyDistributor(nn.Module):
    """
    Distributes available energy across maintenance, growth, and adaptation pathways.
    Includes dynamic modulation based on system instability and contradiction signals.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Base allocation ratios
        self.maintenance_ratio = nn.Parameter(torch.tensor(0.5, device=device))
        self.growth_ratio = nn.Parameter(torch.tensor(0.3, device=device))
        self.adaptation_ratio = nn.Parameter(torch.tensor(0.2, device=device))

        # Dynamic modulation
        self.allocation_flexibility = nn.Parameter(torch.tensor(0.4, device=device))
        self.feedback_sensitivity = nn.Parameter(torch.tensor(0.3, device=device))

        # Energy tracking
        self.register_buffer('allocation_history', torch.zeros(3, 10, device=device))
        self.history_position = 0

    def forward(self, signal_metrics: Dict, available_energy: float) -> Dict:
        affinities = signal_metrics.get("pathway_affinities", {})
        maintenance_affinity = affinities.get("maintenance", 0.33)
        growth_affinity = affinities.get("growth", 0.33)
        adaptation_affinity = affinities.get("adaptation", 0.34)

        # Spectral input
        spectral_info = signal_metrics.get("spectral_info", {})
        spectral_adjustment = {
            "maintenance": spectral_info.get("low_frequency", 0.33) * 0.2,
            "growth": spectral_info.get("mid_frequency", 0.33) * 0.2,
            "adaptation": spectral_info.get("high_frequency", 0.34) * 0.2
        }

        # Instability/contradiction modulation
        instability = max(0.0, 1.0 - signal_metrics.get("stability", 1.0))
        fatigue = signal_metrics.get("fatigue_level", 0.0)  
        contradiction = instability + min(1.0, fatigue / 20.0)

        contradiction_boost = contradiction * self.feedback_sensitivity.item()

        base_dist = torch.tensor([
            self.maintenance_ratio.item(),
            self.growth_ratio.item(),
            self.adaptation_ratio.item() + contradiction_boost
        ], device=self.device)

        signal_influence = torch.tensor([
            maintenance_affinity,
            growth_affinity,
            adaptation_affinity
        ], device=self.device)

        spectral_influence = torch.tensor([
            spectral_adjustment["maintenance"],
            spectral_adjustment["growth"],
            spectral_adjustment["adaptation"]
        ], device=self.device)

        flexibility = self.allocation_flexibility.item()
        distribution = base_dist * (1 - flexibility) + (signal_influence + spectral_influence) * flexibility

        distribution = distribution / torch.sum(distribution)

        # Strategic hesitation...it must reserve unspent energy during low confidence
        reserve = 0.0
        if instability > 0.6:
            reserve = 0.1 * available_energy
            available_energy *= 0.9

        energy_allocation = {
            "maintenance": (distribution[0] * available_energy).item(),
            "growth": (distribution[1] * available_energy).item(),
            "adaptation": (distribution[2] * available_energy).item(),
            "reserved": reserve,
            "total": available_energy
        }

        self.allocation_history *= 0.95
        self.allocation_history[:, self.history_position] = distribution
        self.history_position = (self.history_position + 1) % 10

        reasoning_footprint = {
            "base_ratios": base_dist.tolist(),
            "signal_affinities": signal_influence.tolist(),
            "spectral_influence": spectral_influence.tolist(),
            "contradiction_boost": contradiction_boost,
            "flexibility": flexibility,
            "final_distribution": distribution.tolist()
        }

        energy_allocation["reasoning_footprint"] = reasoning_footprint

        return energy_allocation


class FieldModulator(nn.Module):
    """
    Modulates the field based on energy allocation.
    Instead of rewriting protocols, it implements energy-based effects.
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim
        
        # Field modulation 
        self.maintenance_stability = nn.Parameter(torch.tensor(0.8, device=device))  # Higher the better...more stable
        self.growth_expansion = nn.Parameter(torch.tensor(0.5, device=device))      
        self.adaptation_plasticity = nn.Parameter(torch.tensor(0.6, device=device))  # How easily field structure changes
        
        # Spatial influence
        self.global_influence = nn.Parameter(torch.tensor(0.3, device=device))      # Global vs local
        self.pattern_weight = nn.Parameter(torch.tensor(0.7, device=device))        
        
        # Energy efficacy 
        self.efficiency_curve = nn.Parameter(torch.tensor([0.2, 0.8, 1.0, 0.9, 0.7], device=device))  # Non-linear 
        
        self.register_buffer('recent_modulation', torch.zeros(field_dim, device=device))
        self.register_buffer('modulation_count', torch.tensor(0, device=device))

    def forward(self, field_state: torch.Tensor, energy_allocation: Dict) -> Dict:
        """
        Apply field modulation based on energy allocation.

        Args:
            field_state: Current field tensor
            energy_allocation: Dict with energy allocated to different pathways

        Returns:
            Dict with modulation effects to apply to the field
        """
        maintenance_energy = energy_allocation.get("maintenance", 0.0)
        growth_energy = energy_allocation.get("growth", 0.0)
        adaptation_energy = energy_allocation.get("adaptation", 0.0)
        
        # field stabilization
        maintenance_effect = self.calculate_maintenance_effect(field_state, maintenance_energy)
        
        # field expansion or intensification
        growth_effect = self.calculate_growth_effect(field_state, growth_energy)
        
        # field restructuring
        adaptation_effect = self.calculate_adaptation_effect(field_state, adaptation_energy)
        
        field_modulation = maintenance_effect + growth_effect + adaptation_effect
        
        # The efficiency of energy usage isn't linear
        field_modulation = self.apply_efficiency_curve(field_modulation)
        
        self.recent_modulation = field_modulation.detach()
        self.modulation_count += 1
        
        avg_magnitude = torch.mean(torch.abs(field_modulation)).item()
        max_magnitude = torch.max(torch.abs(field_modulation)).item()
        
        return {
            "field_delta": field_modulation,
            "maintenance_effect": torch.mean(torch.abs(maintenance_effect)).item(),
            "growth_effect": torch.mean(torch.abs(growth_effect)).item(),
            "adaptation_effect": torch.mean(torch.abs(adaptation_effect)).item(),
            "avg_magnitude": avg_magnitude,
            "max_magnitude": max_magnitude
        }
        
    def calculate_maintenance_effect(self, field: torch.Tensor, energy: float) -> torch.Tensor:
        # Maintenance energy preserves existing patterns and dampens fluctuations
        
        # baseline
        h, w = self.field_dim
        stability_mask = torch.ones((h, w), device=self.device) * self.maintenance_stability
        
        energy_factor = torch.tanh(torch.tensor(energy * 2.0, device=self.device))
        stabilization = stability_mask * energy_factor
        
        # small stabilizing nudge toward attractor points
        field_smoothed = F.avg_pool2d(
            field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
        # stabilization pulls slightly toward smoothed version
        delta = (field_smoothed - field) * stabilization * 0.05
        
        return delta
        
    def calculate_growth_effect(self, field: torch.Tensor, energy: float) -> torch.Tensor:
        # Growth energy intensifies existing patterns and expands their influence
        
        # wheres the local maxima?
        field_abs = torch.abs(field)
        threshold = torch.mean(field_abs) + torch.std(field_abs) * 0.5
        active_regions = (field_abs > threshold).float()
        
        # Should be stronger near active regions
        growth_mask = F.max_pool2d(
            active_regions.unsqueeze(0).unsqueeze(0),
            kernel_size=5, stride=1, padding=2
        ).squeeze()
        
        # Growing by available energy
        energy_factor = torch.tanh(torch.tensor(energy * 3.0, device=self.device))
        growth_strength = self.growth_expansion * energy_factor
        
        # INTENSIFY existing patterns
        delta = field * growth_mask * growth_strength * 0.15
        
        return delta
        
    def calculate_adaptation_effect(self, field: torch.Tensor, energy: float) -> torch.Tensor:
        # Adaptation energy introduces new patterns and reorganizes the field
        
        # The structual transformation pattern
        h, w = self.field_dim
        
        # This uses a combination of gradients and fft-based processing...this probably needs work
        
        grad_y, grad_x = torch.gradient(field)
        curl = grad_x - grad_y
        
        # Emergent pattern
        fft = torch.fft.rfft2(field)
        phase = torch.angle(fft)
        structure = torch.fft.irfft2(torch.exp(1j * phase), s=field.shape)
        
        # Rotate
        adaptation_pattern = curl * 0.3 + structure * 0.7
        adaptation_pattern = adaptation_pattern / (torch.norm(adaptation_pattern) + 1e-8)
        
        energy_factor = torch.tanh(torch.tensor(energy * 2.5, device=self.device))
        adaptation_strength = self.adaptation_plasticity * energy_factor
        
        delta = adaptation_pattern * adaptation_strength * 0.2
        
        return delta
        
    def apply_efficiency_curve(self, modulation: torch.Tensor) -> torch.Tensor:
        # Energy utilization isn't perfectly efficient
        # This applies a non-linear curve to the modulation magnitude
        modulation_magnitude = torch.abs(modulation)
        modulation_sign = torch.sign(modulation)
        
        # For interpolation
        norm_magnitude = modulation_magnitude / (torch.max(modulation_magnitude) + 1e-8)
        
        curve = self.efficiency_curve
        
        # Trying a 5 point curve
        adjusted_magnitude = torch.zeros_like(modulation_magnitude)
        
        for i in range(len(curve) - 1):
            mask = ((norm_magnitude >= i / (len(curve) - 1)) & 
                   (norm_magnitude < (i + 1) / (len(curve) - 1)))
            
            t = (norm_magnitude - i / (len(curve) - 1)) * (len(curve) - 1)
            adjusted = curve[i] * (1 - t) + curve[i + 1] * t
            adjusted_magnitude[mask] = modulation_magnitude[mask] * adjusted[mask]
            
        mask = (norm_magnitude >= (len(curve) - 1) / (len(curve) - 1))
        adjusted_magnitude[mask] = modulation_magnitude[mask] * curve[-1]
        
        adjusted_modulation = adjusted_magnitude * modulation_sign
        
        return adjusted_modulation


class PhotosynthesisEnergySystem(nn.Module):
    """
    Energy allocation system inspired by photosynthesis.
    Focuses on energy capture, distribution and allocation
    rather than classification.
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim
        
        self.energy_capacity = nn.Parameter(torch.ones(field_dim, device=device))
        self.energy_current = torch.zeros(field_dim, device=device)
        self.energy_flow_rates = torch.ones(3, device=device)  # For 3 pathways
        
        # renamed from router
        self.signal_analyzer = SignalAnalyzer(field_dim, device)
        
        # renamed from threshold layer
        self.energy_distributor = EnergyDistributor(device)
        
        # renamed from rewriting protocol
        self.field_modulator = FieldModulator(field_dim, device)
        
        self.field = nn.Parameter(torch.zeros(field_dim, device=device))
        self.register_buffer('velocity', torch.zeros(field_dim, device=device))
        self.register_buffer('adaptivity', torch.zeros(field_dim, device=device))
        
        # Learnable
        self.stability = nn.Parameter(torch.tensor(0.88, device=device))
        self.propagation = nn.Parameter(torch.tensor(0.15, device=device))
        self.excitation = nn.Parameter(torch.tensor(0.4, device=device))
        
        self.energy_history = []
        
        self.attractors = []
        self.field_history = []
        self.field_diff_history = []

        # with stable pattern
        self._initialize_field()
        
    def _initialize_field(self):
        h, w = self.field_dim
        cx, cy = w // 2, h // 2

        with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    dist = torch.sqrt(torch.tensor(((i-cy)/cy)**2 + ((j-cx)/cx)**2))
                    self.field[i, j] = torch.exp(-3.0 * dist)

            self.field.data = self.field.data / torch.norm(self.field.data)
        
    def forward(self, input_tensor=None, input_pos=None):
        """
        Process an input through the photosynthesis-inspired energy system.

        Args:
            input_tensor: Optional input tensor to process
            input_pos: Optional position to apply input (default: center)

        Returns:
            Dict with field state and energy system metrics
        """
        previous_field = self.field.detach().clone()
   
        if input_tensor is None:
            self.update_field_physics()
            return {
                'field': self.field,
                'attractors': self.attractors,
                'stability': self.calculate_stability(),
                'energy_level': torch.mean(self.energy_current).item()
            }
        
        if input_pos is None:
            input_pos = (self.field_dim[0]//2, self.field_dim[1]//2)
            
        # Does it match
        if input_tensor.shape != self.field.shape:
            processed_input = torch.zeros_like(self.field)
            
            # Around specified position
            i, j = input_pos
            radius = 5  
            for di in range(-radius, radius+1):
                for dj in range(-radius, radius+1):
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.field_dim[0] and 0 <= nj < self.field_dim[1]:
                        dist = torch.sqrt(torch.tensor(float(di**2 + dj**2)))
                        factor = torch.exp(-0.5 * (dist / 2)**2)
                        
                        if isinstance(input_tensor, torch.Tensor) and input_tensor.numel() == 1:
                            processed_input[ni, nj] = input_tensor.item() * factor
                        else:
                            # If multi-dimensional, take mean
                            processed_input[ni, nj] = torch.mean(input_tensor) * factor
        else:
            processed_input = input_tensor

        signal_metrics = self.signal_analyzer(processed_input, self.field)

        available_energy = self.calculate_available_energy()

        energy_distribution = self.energy_distributor(signal_metrics, available_energy)
        
        modulation_effects = self.field_modulator(self.field, energy_distribution)
        
        self.field.data = self.field.data + modulation_effects["field_delta"]

        self.update_energy_state(energy_distribution)
        
        if len(self.field_history) >= 10:
            self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())
        
        self.update_field_physics()
        
        self.detect_attractors()
        
        field_diff = torch.norm(self.field - previous_field).item()
        if len(self.field_diff_history) >= 10:
            self.field_diff_history.pop(0)
        self.field_diff_history.append(field_diff)
        
        self.energy_history.append({
            'signal_metrics': signal_metrics,
            'energy_distribution': energy_distribution,
            'available_energy': available_energy,
            'modulation_effects': {k: v for k, v in modulation_effects.items() if not isinstance(v, torch.Tensor)}
        })
        
        return {
            'field': self.field,
            'attractors': self.attractors,
            'stability': self.calculate_stability(),
            'signal_metrics': signal_metrics,
            'energy_distribution': energy_distribution,
            'modulation_effects': {k: v for k, v in modulation_effects.items() if not isinstance(v, torch.Tensor)},
            'energy_level': torch.mean(self.energy_current).item()
        }
        
    def calculate_available_energy(self):
        # Capacity
        base_energy = torch.mean(self.energy_capacity).item()
        
        current_state = torch.mean(self.energy_current).item()
        
        available = max(0.1, base_energy - (current_state * 0.5))
        
        return available
    
    def update_energy_state(self, distribution):
         # Energy consumption
        total_allocated = sum(distribution.values()) if isinstance(distribution, dict) else distribution
        
        # Simple version so energy decreases based on usage and recovers slowly
        self.energy_current = self.energy_current * 0.9 + torch.ones_like(self.energy_current) * total_allocated * 0.3
        
        self.energy_current = torch.clamp(self.energy_current, min=0.0)
        
        self.energy_current = torch.minimum(self.energy_current, self.energy_capacity)
    
    def update_field_physics(self):
        laplacian = self.compute_laplacian()
        
        self.velocity += self.propagation * laplacian

        self.field.data += self.velocity
        
        # Dampening
        self.velocity *= self.stability
        
        # So no field explosion
        field_norm = torch.norm(self.field)
        if field_norm > 2.0:
            self.field.data = self.field.data * (2.0 / field_norm)
    
    def compute_laplacian(self):
        """simplified version"""
        h, w = self.field.shape
        laplacian = torch.zeros_like(self.field)
        
        # This is for efficient computation
        kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], device=self.device).view(1, 1, 3, 3)
        
        # So it can handle boundaries
        padded_field = F.pad(self.field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
        
        laplacian = F.conv2d(padded_field, kernel).squeeze()
        
        return laplacian
        
    def detect_attractors(self):
        """Find emergent attractors in the field with adaptive basin sizing"""
        h, w = self.field.shape
        smoothed = F.avg_pool2d(
            self.field.unsqueeze(0).unsqueeze(0),
            kernel_size=3, stride=1, padding=1
        ).squeeze()
        
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
    
    def calculate_basin(self, i, j):
        #  Around the point
        h, w = self.field.shape
        grad_sum = 0.0
        count = 0

        for ni in range(max(0, i-2), min(h, i+3)):
            for nj in range(max(0, j-2), min(w, j+3)):
                if ni == i and nj == j:
                    continue
                
                grad = abs(self.field[ni, nj].item() - self.field[i, j].item())
                grad_sum += grad
                count += 1
        
        # smaller means wider basin
        avg_grad = grad_sum / count if count > 0 else 1.0
        
        # smaller gradient makes wider basin
        if avg_grad < 0.001:
            return 5.0  # Cap on basin size
        return min(5.0, 1.0 / avg_grad)
    
    def calculate_stability(self):
        if len(self.field_diff_history) < 2:
            return 0.5  # Default neutral stability
        
        # more weight to recent
        recent_diffs = self.field_diff_history[-5:]
        weights = torch.linspace(0.5, 1.0, len(recent_diffs), device=self.device)
        weighted_diff = sum(d * w for d, w in zip(recent_diffs, weights)) / sum(weights)
        
        stability = 1.0 / (1.0 + weighted_diff)
        return stability
    
    def reset(self):
        with torch.no_grad():
            self._initialize_field()
            self.velocity.zero_()
            self.energy_current.zero_()
            self.field_history = []
            self.field_diff_history = []
            self.attractors = []
            self.energy_history = []
    
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
        
        # how consistent energy levels are
        energy_stability = 1.0 - min(1.0, torch.std(torch.tensor(energy_levels)).item())
        
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
        Requires matplotlib to be installed. (Sorry, had lots of
        weird issues with visuals.)
        
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
        
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
        
        # To see energy allocation over time
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
        
        # To see energy distribution in field
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
        
        # For signal metrics
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