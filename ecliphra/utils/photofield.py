"""
Ecliphra PhotoField Module

This module extends the Ecliphra model architecture with a photosynthesis like
signal processing that intelligently routes incoming signals based on
their characteristics. (Currently retired until I can fix it.)

Key components:
- SignalRouter: Classifies signals as noise, seeds, or challenges
- IntegrationThresholdLayer: Determines which signals pass through to affect the field
- FieldRewritingProtocol: Controls how accepted signals rewrite the field
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import random


class SignalRouter(nn.Module):
    """
    Sends incoming signals based on their characteristics relative to the current field state.

    Determines whether a signal is:
    - Noise (to be filtered out)
    - A seed (to be integrated)
    - A challenge (requiring field redirection)
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim

        # characterization parameters
        self.novelty_weight = nn.Parameter(torch.tensor(0.4, device=device))
        self.resonance_weight = nn.Parameter(torch.tensor(0.3, device=device))
        self.disruption_weight = nn.Parameter(torch.tensor(0.2, device=device))

        # classification thresholds
        self.noise_threshold = nn.Parameter(torch.tensor(0.3, device=device))
        self.seed_threshold = nn.Parameter(torch.tensor(0.7, device=device))
        # Note: Anything above seed_threshold is considered a challenge

        self.signal_history = []
        self.history_capacity = 10

        # Signal type statistics for metacognition
        self.register_buffer('signal_stats', torch.zeros(3, device=device))  # noise, seed, challenge

    def forward(self,
                input_signal: torch.Tensor,
                current_field: torch.Tensor,
                field_velocity: torch.Tensor,
                attractors: List) -> Dict:
        """
        Process an incoming signal and determine its routing classification.

        Args:
            input_signal: The incoming signal tensor
            current_field: The current state of the field
            field_velocity: The current velocity of the field
            attractors: List of current attractors in the field

        Returns:
            Dict containing signal classification and metrics
        """
        novelty = self.calculate_novelty(input_signal)
        resonance = self.calculate_resonance(input_signal, current_field, attractors)
        disruption = self.calculate_disruption(input_signal, field_velocity)

        signal_score = (
            self.novelty_weight * novelty +
            self.resonance_weight * resonance +
            self.disruption_weight * disruption
        )

        if signal_score < self.noise_threshold:
            signal_type = "noise"
            self.signal_stats[0] += 1
        elif signal_score < self.seed_threshold:
            signal_type = "seed"
            self.signal_stats[1] += 1
        else:
            signal_type = "challenge"
            self.signal_stats[2] += 1

        self.update_history(input_signal, signal_type)

        return {
            "signal_type": signal_type,
            "signal_score": signal_score.item(),
            "novelty": novelty.item(),
            "resonance": resonance.item(),
            "disruption": disruption.item()
        }

    def calculate_novelty(self, input_signal: torch.Tensor) -> torch.Tensor:
        """Calculate how novel the signal is compared to history"""
        if not self.signal_history:
            return torch.tensor(1.0, device=self.device)  # First signal is maximally novel

        max_similarity = 0.0
        input_flat = input_signal.view(-1)

        for past_signal, _ in self.signal_history:
            past_flat = past_signal.view(-1)
            similarity = F.cosine_similarity(input_flat.unsqueeze(0), past_flat.unsqueeze(0))
            max_similarity = max(max_similarity, similarity.item())

        novelty = 1.0 - max_similarity
        return torch.tensor(novelty, device=self.device)

    def calculate_resonance(self,
                           input_signal: torch.Tensor,
                           current_field: torch.Tensor,
                           attractors: List) -> torch.Tensor:
        """Calculate how much the signal resonates with current attractors"""
        field_resonance = F.cosine_similarity(
            input_signal.view(-1).unsqueeze(0),
            current_field.view(-1).unsqueeze(0)
        )

        attractor_resonance = 0.0
        if attractors:
            attractor_masks = []
            for attractor in attractors:
                if len(attractor) >= 2:  # Handle different attractor formats
                    pos = attractor[0] if isinstance(attractor[0], tuple) else attractor[:2]
                    mask = torch.zeros_like(current_field)

                    # Making a gaussian mask around attractor
                    i, j = pos
                    radius = 5
                    for di in range(-radius, radius+1):
                        for dj in range(-radius, radius+1):
                            ni, nj = i+di, j+dj
                            if 0 <= ni < self.field_dim[0] and 0 <= nj < self.field_dim[1]:
                                dist = np.sqrt(di**2 + dj**2)
                                mask[ni, nj] = np.exp(-0.5 * (dist / 2)**2)

                    attractor_masks.append(mask)

            # Getting weighted average of attractor resonances
            if attractor_masks:
                for mask in attractor_masks:
                    masked_input = input_signal * mask
                    masked_field = current_field * mask

                    if torch.sum(mask) > 0:
                        local_resonance = F.cosine_similarity(
                            masked_input.view(-1).unsqueeze(0),
                            masked_field.view(-1).unsqueeze(0)
                        )
                        attractor_resonance += local_resonance.item()

                attractor_resonance /= len(attractor_masks)

        combined_resonance = 0.3 * field_resonance + 0.7 * attractor_resonance
        return torch.tensor(combined_resonance, device=self.device)

    def calculate_disruption(self,
                           input_signal: torch.Tensor,
                           field_velocity: torch.Tensor) -> torch.Tensor:
        current_energy = torch.norm(field_velocity)

        signal_energy = torch.norm(input_signal)

        # Disruption is proportion of energy added relative to current energy
        if current_energy < 1e-6:  
            disruption = min(signal_energy.item(), 1.0)
        else:
            disruption = min((signal_energy / current_energy).item(), 1.0)

        return torch.tensor(disruption, device=self.device)

    def update_history(self, signal: torch.Tensor, signal_type: str):
        self.signal_history.append((signal.detach().clone(), signal_type))

        if len(self.signal_history) > self.history_capacity:
            self.signal_history.pop(0)


class IntegrationThresholdLayer(nn.Module):
    """
    Determines whether signals pass through to affect the field.
    Acts as a gating mechanism similar to a Zener diode.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        # Base 
        self.noise_pass_ratio = nn.Parameter(torch.tensor(0.1, device=device))  # want very little noise passing
        self.seed_pass_ratio = nn.Parameter(torch.tensor(0.6, device=device))   # let most seed content pass
        self.challenge_pass_ratio = nn.Parameter(torch.tensor(0.8, device=device))  # let most challenge content pass

        # Adaptive threshold modifiers
        self.field_stability_influence = nn.Parameter(torch.tensor(0.2, device=device))
        self.field_adaptivity_influence = nn.Parameter(torch.tensor(0.2, device=device))

        # Tracking
        self.register_buffer('pass_counts', torch.zeros(3, device=device))  
        self.register_buffer('block_counts', torch.zeros(3, device=device)) 

    def forward(self,
               signal: torch.Tensor,
               signal_info: Dict,
               field_stability: float,
               field_adaptivity: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Apply threshold gating to incoming signal.

        Args:
            signal: Input signal tensor
            signal_info: Dict with signal classification from SignalRouter
            field_stability: Current stability metric of the field
            field_adaptivity: Optional field adaptivity tensor

        Returns:
            Tuple of (modified_signal, gate_info)
        """
        signal_type = signal_info["signal_type"]
        signal_score = signal_info.get("signal_score", 0.5)

        if signal_type == "noise":
            base_pass_ratio = self.noise_pass_ratio * 0.8  # reducing
            type_idx = 0
        elif signal_type == "seed":
            confidence = signal_info.get("classification_confidence", 0.5)
            base_pass_ratio = self.seed_pass_ratio * min(1.0, confidence + 0.2)
            type_idx = 1
        else:  
            # challenge pass needs to be more variable
            base_pass_ratio = self.challenge_pass_ratio * 1.2  # maybe boost
            type_idx = 2

        stability_modifier = -self.field_stability_influence * field_stability

        adaptivity_modifier = 0.0
        if field_adaptivity is not None:
            mean_adaptivity = torch.mean(field_adaptivity)
            adaptivity_modifier = self.field_adaptivity_influence * mean_adaptivity

        adjusted_pass_ratio = torch.clamp(
            base_pass_ratio + stability_modifier + adaptivity_modifier,
            min=0.0, max=1.0
        )

        noise_factor = 0.05
        noise = torch.randn_like(signal) * noise_factor
        gate_mask = torch.clamp(torch.ones_like(signal) * adjusted_pass_ratio + noise, 0.0, 1.0)
        gated_signal = signal * gate_mask

        if adjusted_pass_ratio > 0.5:  
            self.pass_counts[type_idx] += 1
        else:
            self.block_counts[type_idx] += 1

        return gated_signal, {
            "pass_ratio": adjusted_pass_ratio.item(),
            "stability_modifier": stability_modifier.item(),
            "adaptivity_modifier": adaptivity_modifier,
            "signal_type": signal_type
        }


class FieldRewritingProtocol(nn.Module):
    """
    Controls how accepted signals change the field's attractor structure.
    Lets it be gentle field adaptation rather than complete overwriting.
    """
    def __init__(self, field_dim=(32, 32), device='cpu'):
        super().__init__()
        self.device = device
        self.field_dim = field_dim

        self.noise_influence = nn.Parameter(torch.tensor(0.1, device=device))
        self.seed_influence = nn.Parameter(torch.tensor(0.3, device=device))
        self.challenge_influence = nn.Parameter(torch.tensor(0.5, device=device))

        self.global_influence_ratio = nn.Parameter(torch.tensor(0.3, device=device))
        self.local_influence_ratio = nn.Parameter(torch.tensor(0.7, device=device))

        self.influence_activation = nn.Sigmoid()

        self.register_buffer('cumulative_rewrite', torch.zeros(field_dim, device=device))
        self.register_buffer('rewrite_count', torch.tensor(0, device=device))

    def forward(self,
               field: torch.Tensor,
               signal: torch.Tensor,
               signal_info: Dict,
               attractors: List) -> Tuple[torch.Tensor, Dict]:
        """
        Apply the signal to rewrite the field using appropriate protocol.

        Args:
            field: Current field tensor to be modified
            signal: Gated input signal tensor
            signal_info: Signal classification and metrics
            attractors: Current attractors in the field

        Returns:
            Tuple of (modified_field, rewrite_info)
        """
        signal_type = signal_info["signal_type"]

        if signal_type == "noise":
            influence_strength = self.noise_influence
        elif signal_type == "seed":
            influence_strength = self.seed_influence
        else:  # challenge
            influence_strength = self.challenge_influence

        influence_map = self.create_influence_map(field, signal, attractors)

        field_update = (field * (1 - influence_map * influence_strength) +
                        signal * (influence_map * influence_strength))

        # no field explosion
        field_norm = torch.norm(field_update)
        if field_norm > 0:
            field_update = field_update * (torch.norm(field) / field_norm)

        self.cumulative_rewrite += torch.abs(field_update - field)
        self.rewrite_count += 1

        avg_influence = torch.mean(influence_map * influence_strength).item()
        max_influence = torch.max(influence_map * influence_strength).item()

        return field_update, {
            "signal_type": signal_type,
            "influence_strength": influence_strength.item(),
            "avg_influence": avg_influence,
            "max_influence": max_influence,
            "rewrite_count": self.rewrite_count.item()
        }

    def create_influence_map(self,
                           field: torch.Tensor,
                           signal: torch.Tensor,
                           attractors: List) -> torch.Tensor:
        """
        A spatial map of where the signal should influence the field.
        Combines global influence and attractor-focused influence.

        Returns:
            influence_map: Tensor of same shape as field, with values 0-1
        """
        h, w = self.field_dim

        influence_map = torch.ones((h, w), device=self.device) * self.global_influence_ratio

        # focused component 
        if attractors:
            for attractor in attractors:
                if len(attractor) >= 2:  
                    pos = attractor[0] if isinstance(attractor[0], tuple) else attractor[:2]

                    #  using for scaling influence
                    strength = attractor[1] if len(attractor) > 1 else 1.0

                    i, j = pos
                    radius = int(5 * strength)  
                    for di in range(-radius, radius+1):
                        for dj in range(-radius, radius+1):
                            ni, nj = i+di, j+dj
                            if 0 <= ni < h and 0 <= nj < w:
                                dist = torch.sqrt(torch.tensor(float(di**2 + dj**2)))
                                falloff = torch.exp(-0.5 * (dist / (2 * strength))**2)

                                attractor_influence = falloff * self.local_influence_ratio * strength
                                influence_map[ni, nj] += attractor_influence

        # activation useful to keep values in reasonable range
        influence_map = self.influence_activation(influence_map)

        influence_map = torch.clamp(influence_map, 0.0, 1.0)

        return influence_map


class EcliphraPhotoField(nn.Module):
    """
    Enhanced Ecliphra Field with photosynthesis-like signal processing.

    Integrates the three components:
    - SignalRouter
    - IntegrationThresholdLayer
    - FieldRewritingProtocol

    This can be used as a standalone model or integrated with existing Ecliphra models.
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

        # Photosynthesis stuff
        self.signal_router = SignalRouter(field_dim, device)
        self.threshold_layer = IntegrationThresholdLayer(device)
        self.rewriting_protocol = FieldRewritingProtocol(field_dim, device)

        self.attractors = []  # position, strength, content

        self.field_history = []
        self.field_diff_history = []
        self.signal_processing_history = []

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
        Process an input through the photosynthesis like field.

        Args:
            input_tensor: Optional input tensor to process
            input_pos: Optional position to apply input (default: center)

        Returns:
            Dict with field state and processing metrics
        """
        previous_field = self.field.detach().clone()

        if input_tensor is None:
            self.update_field_physics()
            return {
                'field': self.field,
                'attractors': self.attractors,
                'stability': self.calculate_stability()
            }

        if input_pos is None:
            input_pos = (self.field_dim[0]//2, self.field_dim[1]//2)

        if input_tensor.shape != self.field.shape:
            processed_input = torch.zeros_like(self.field)

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
                            processed_input[ni, nj] = torch.mean(input_tensor) * factor
        else:
            processed_input = input_tensor

        signal_info = self.signal_router(
            processed_input,
            self.field,
            self.velocity,
            self.attractors
        )

        stability = self.calculate_stability()
        gated_signal, gate_info = self.threshold_layer(
            processed_input,
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

        if len(self.field_history) >= 10:
            self.field_history.pop(0)
        self.field_history.append(self.field.detach().clone())

        self.update_field_physics()

        self.detect_attractors()

        field_diff = torch.norm(self.field - previous_field).item()
        if len(self.field_diff_history) >= 10:
            self.field_diff_history.pop(0)
        self.field_diff_history.append(field_diff)

        self.update_adaptivity(previous_field, self.field)

        processing_summary = {
            'signal_type': signal_info['signal_type'],
            'pass_ratio': gate_info['pass_ratio'],
            'avg_influence': rewrite_info['avg_influence']
        }
        self.signal_processing_history.append(processing_summary)

        return {
            'field': self.field,
            'attractors': self.attractors,
            'stability': self.calculate_stability(),
            'signal_info': signal_info,
            'gate_info': gate_info,
            'rewrite_info': rewrite_info
        }

    def update_field_physics(self):
        laplacian = self.compute_laplacian()

        self.velocity += self.propagation * laplacian

        self.field.data += self.velocity

        self.velocity *= self.stability

    def compute_laplacian(self):
        h, w = self.field.shape
        laplacian = torch.zeros_like(self.field)

        # Using convolution for efficient computation
        kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], device=self.device).view(1, 1, 3, 3)

        padded_field = F.pad(self.field.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')

        laplacian = F.conv2d(padded_field, kernel).squeeze()

        return laplacian

    def update_adaptivity(self, prev_field, current_field):
        field_change = torch.abs(current_field - prev_field)

        disruption_mask = field_change > self.disruption_threshold
        growth = field_change * self.adaptivity_growth_rate

        self.adaptivity[disruption_mask] += growth[disruption_mask]

        self.adaptivity = torch.clamp(self.adaptivity, max=self.adaptivity_maximum)

        # Decay adaptivity everywhere
        self.adaptivity = self.adaptivity * (1.0 - self.adaptivity_decay_rate)

    def detect_attractors(self):
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

                    adaptive_multiplier = 1.0
                    if hasattr(self, 'adaptivity'):
                        adaptive_multiplier = 1.0 + self.adaptivity[i, j].item()

                    adaptive_basin_size = base_basin_size * adaptive_multiplier

                    strength = center * adaptive_basin_size

                    candidates.append(((i, j), strength, base_basin_size, adaptive_basin_size))

        candidates.sort(key=lambda x: x[1], reverse=True)

        self.attractors = [(pos, strength, base_size, adaptive_size)
                          for (pos, strength, base_size, adaptive_size) in candidates[:3]]

    def calculate_basin(self, i, j):
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

        avg_grad = grad_sum / count if count > 0 else 1.0

        # smaller gradient needs a wider basin
        if avg_grad < 0.001:
            return 5.0  
        return min(5.0, 1.0 / avg_grad)

    def calculate_stability(self):
        if len(self.field_diff_history) < 2:
            return 0.5  

        recent_diffs = self.field_diff_history[-5:]
        weights = torch.linspace(0.5, 1.0, len(recent_diffs), device=self.device)
        weighted_diff = sum(d * w for d, w in zip(recent_diffs, weights)) / sum(weights)

        stability = 1.0 / (1.0 + weighted_diff)
        return stability

    def reset(self):
        with torch.no_grad():
            self._initialize_field()
            self.velocity.zero_()
            self.field_history = []
            self.field_diff_history = []
            self.attractors = []
            self.signal_processing_history = []

            self.signal_router.signal_history = []
            self.signal_router.signal_stats.zero_()
            self.threshold_layer.pass_counts.zero_()
            self.threshold_layer.block_counts.zero_()
            self.rewriting_protocol.cumulative_rewrite.zero_()
            self.rewriting_protocol.rewrite_count.zero_()

    def get_signal_statistics(self):
        total_signals = torch.sum(self.signal_router.signal_stats).item()
        if total_signals == 0:
            signal_distribution = [0, 0, 0]
        else:
            signal_distribution = [
                self.signal_router.signal_stats[0].item() / total_signals,
                self.signal_router.signal_stats[1].item() / total_signals,
                self.signal_router.signal_stats[2].item() / total_signals
            ]

        pass_rates = []
        for i in range(3):
            total = self.threshold_layer.pass_counts[i] + self.threshold_layer.block_counts[i]
            if total > 0:
                pass_rates.append(self.threshold_layer.pass_counts[i].item() / total)
            else:
                pass_rates.append(0)

        recent_signal_types = [
            entry['signal_type'] for entry in self.signal_processing_history[-10:]
        ]

        if len(self.signal_processing_history) > 0:
            avg_influence = sum(entry['avg_influence'] for entry in self.signal_processing_history) / len(self.signal_processing_history)
        else:
            avg_influence = 0

        return {
            'signal_distribution': signal_distribution,  # noise, seed, challenge
            'pass_rates': pass_rates,  # noise, seed, challenge
            'recent_signal_types': recent_signal_types,
            'avg_influence': avg_influence,
            'total_processed': total_signals
        }

class MemoryAugmentedClassifier(nn.Module):
    """
    Memory-augmented classification system that builds an episodic memory
    of past signals and their classifications to improve future decisions.
    """
    def __init__(self, memory_capacity=100, feature_dim=None, device='cpu'):
        super().__init__()
        self.device = device
        self.memory_capacity = memory_capacity
        self.step = 0

        self.signal_memory = []

        self.signal_types = ['noise', 'seed', 'challenge']
        self.type_to_idx = {t: i for i, t in enumerate(self.signal_types)}
        self.idx_to_type = {i: t for i, t in enumerate(self.signal_types)}

        self.revision_history = []  
        self.confidence_history = []  

        self.memory_weight = nn.Parameter(torch.tensor(0.4, device=device))
        self.feature_weight = nn.Parameter(torch.tensor(0.6, device=device))

        self.learning_rate = 0.01

        self.signal_counts = {t: 0 for t in self.signal_types}
        self.last_classifications = []  

    def forward(self, signal_features, signal_metrics, original_classification=None):
        """
        Classify a signal using both its features and memory of past signals.

        Args:
            signal_features: Feature tensor of the signal (fingerprint or derived features)
            signal_metrics: Dict of metrics like novelty, resonance, disruption
            original_classification: The initial classification (if already classified)

        Returns:
            Dict containing final classification and confidence
        """
        if original_classification is None:
            original_classification, feature_confidence = self._classify_by_features(signal_metrics)
        else:
            feature_confidence = self._estimate_feature_confidence(signal_metrics, original_classification)

        memory_classification, memory_confidence, similar_signals = self._classify_by_memory(
            signal_features, signal_metrics)

        final_type, confidence, was_revised = self._combine_classifications(
            original_classification, feature_confidence,
            memory_classification, memory_confidence)
        
        # EMERGENCY OVERRIDE Force non seed classifications temporarily
        # Enabling this if seed bias is detected
        EMERGENCY_MODE = True  # Set to False when no longer needed
        
        if EMERGENCY_MODE and final_type == 'seed':
            # Need to check if this should be classified as something else
            if signal_metrics.get('novelty', 0) > 0.6:
                forced_type = 'noise'
            elif signal_metrics.get('disruption', 0) > 0.4:
                forced_type = 'challenge'
            else:
                forced_type = 'challenge'
                
            print(f"EMERGENCY OVERRIDE: Forcing {forced_type} instead of seed")
            final_type = forced_type
            confidence *= 0.8  
            was_revised = True

        self._update_memory(signal_features, signal_metrics, final_type, confidence)

        self._track_classification(final_type, confidence, original_classification, was_revised)
        self.step += 1
        self.apply_weight_neutrality_decay()


        self._update_weights()

        return {
            'signal_type': final_type,
            'confidence': confidence,
            'original_type': original_classification,
            'was_revised': was_revised,
            'similar_in_memory': len(similar_signals),
            'memory_weight': self.memory_weight.item(),
            'feature_weight': self.feature_weight.item()
        }

    # Rescue plan 
    def _classify_by_features(self, metrics):
        """Override with completely fixed classify_by_features"""
        novelty = metrics.get('novelty', 0.0)
        resonance = metrics.get('resonance', 0.0)
        disruption = metrics.get('disruption', 0.0)
        stability = metrics.get('stability', 0.0)

        noise_score = (novelty * 1.2) - (resonance * 0.5) + ((1.0 - stability) * 0.4) - (disruption * 0.2) + 0.1
        seed_score = (resonance * 1.0) - (novelty * 0.3) - (disruption * 0.7) + (stability * 0.4) - 0.1
        challenge_score = (disruption * 1.0) + (novelty * 0.3) - (resonance * 0.5) + ((1.0 - stability) * 0.6) - 0.1

        scores = {
            'noise': noise_score,
            'seed': seed_score,
            'challenge': challenge_score
        }

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        sorted_values = sorted(scores.values(), reverse=True)
        if len(sorted_values) > 1:
            confidence = min((sorted_values[0] - sorted_values[1]) / max(abs(sorted_values[0]), 0.1), 1.0)
        else:
            confidence = 0.5

        print(f"Selected type: {max_type} with score {max_score:.2f} (confidence: {confidence:.2f})")

        return max_type, confidence

    def _estimate_feature_confidence(self, metrics, classification):
        predicted_type, predicted_confidence = self._classify_by_features(metrics)

        if predicted_type == classification:
            return predicted_confidence
        else:
            # Returning lower confidence since they don't match
            return 0.5 * predicted_confidence

    def _classify_by_memory(self, features, metrics, k=5):
        if not self.signal_memory or features is None:
            default_type, default_conf = self._classify_by_features(metrics)
            return default_type, 0.1, []  

        similarities = []
        for mem_entry in self.signal_memory:
            if 'features' not in mem_entry:
                continue

            if features.shape == mem_entry['features'].shape:
                feature_sim = F.cosine_similarity(
                    features.view(1, -1),
                    mem_entry['features'].view(1, -1)
                ).item()
            else:
                feature_sim = 0.5

            # simple one
            metric_sim = 0.5  
            if 'metrics' in mem_entry:
                metric_matches = 0
                total_metrics = 0
                for key in ['novelty', 'resonance', 'disruption']:
                    if key in metrics and key in mem_entry['metrics']:
                        total_metrics += 1
                        diff = abs(metrics[key] - mem_entry['metrics'][key])
                        if diff < 0.2:  
                            metric_matches += 1

                if total_metrics > 0:
                    metric_sim = metric_matches / total_metrics

            combined_sim = 0.7 * feature_sim + 0.3 * metric_sim
            similarities.append((mem_entry, combined_sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        if not top_k:
            default_type, default_conf = self._classify_by_features(metrics)
            return default_type, 0.1, []

        type_votes = {t: 0.0 for t in self.signal_types}

        for entry, sim in top_k:
            signal_type = entry['signal_type']
            type_votes[signal_type] += sim

        winning_type = max(type_votes, key=type_votes.get)


        total_votes = sum(type_votes.values())
        if total_votes > 0:
            winning_votes = type_votes[winning_type]
            confidence = winning_votes / total_votes
        else:
            confidence = 0.0

        avg_similarity = sum(sim for _, sim in top_k) / len(top_k)
        confidence *= avg_similarity

        return winning_type, confidence, [entry for entry, _ in top_k]

    def _combine_classifications(self, feature_type, feature_conf, memory_type, memory_conf):
        print(f"Combining: feature={feature_type}({feature_conf:.2f}), memory={memory_type}({memory_conf:.2f})")
        print(f"Weights: feature={self.feature_weight.item():.2f}, memory={self.memory_weight.item():.2f}")
     
        self._normalize_weights()
        
        # Anti bias correction so if both suggest seed, reduce confidence
        if feature_type == "seed" and memory_type == "seed":
            feature_conf *= 0.7
            memory_conf *= 0.7
            print("Applied anti-seed correction")
        
        if feature_type != memory_type:
            feature_weight = self.feature_weight.item()
            memory_weight = self.memory_weight.item()
            
            feature_score = feature_conf * feature_weight
            memory_score = memory_conf * memory_weight
            
            print(f"Scores: feature={feature_score:.2f}, memory={memory_score:.2f}")
            
            if feature_score >= memory_score:
                final_type = feature_type
                confidence = feature_conf * (feature_score / (feature_score + memory_score + 1e-10))
                was_revised = False
            else:
                final_type = memory_type
                confidence = memory_conf * (memory_score / (feature_score + memory_score + 1e-10))
                was_revised = True
        else:
            final_type = feature_type
            confidence = (feature_conf * self.feature_weight.item() + 
                        memory_conf * self.memory_weight.item())
            was_revised = False
        
        confidence = min(confidence, 0.95)
        
        if final_type == "seed":
            # To force other classifications
            if random.random() < 0.3:  
                alternative = "noise" if feature_conf < 0.4 else "challenge"
                print(f"Random override: {final_type} -> {alternative}")
                final_type = alternative 
                confidence *= 0.8
                was_revised = True
        
        print(f"Final decision: {final_type}({confidence:.2f}), revised={was_revised}")
        return final_type, confidence, was_revised

    def encourage_classification_diversity(self, classification):
        """Encourage diversity in classifications"""
        if not hasattr(self, 'recent_classifications'):
            self.recent_classifications = []

        if len(self.recent_classifications) > 5:
            self.recent_classifications.pop(0)
        
        count = self.recent_classifications.count(classification)
        if count >= 3:  
            alternatives = [t for t in self.signal_types if t != classification]
            return random.choice(alternatives)
        
        self.recent_classifications.append(classification)
        return classification
    
    def emergency_classification_override(self, classification, confidence, metrics):
        """Apply emergency override to break out of classification ruts"""
        if not hasattr(self, 'classification_counts'):
            self.classification_counts = {t: 0 for t in self.signal_types}
        
        self.classification_counts[classification] = self.classification_counts.get(classification, 0) + 1
        
        # Find dominance
        total = sum(self.classification_counts.values())
        if total < 5:  
            return classification, confidence, False
            
        dominant_ratio = max(self.classification_counts.values()) / total
        dominant_type = max(self.classification_counts, key=self.classification_counts.get)
        
        if dominant_ratio > 0.7 and dominant_type == classification:
            print(f"Emergency override: breaking {dominant_type} dominance ({dominant_ratio:.1%})")
            
            if metrics.get('novelty', 0) > 0.5:
                new_type = 'noise'
            else:
                new_type = 'challenge'
                
            return new_type, confidence * 0.8, True
        
        return classification, confidence, False
    
    def reset_memory_classifier(self):
        """Reset and rebuild the memory classifier's database"""
        if not hasattr(self, 'memory_classifier'):
            return
            
        print("EMERGENCY: Resetting memory classifier to fix seed bias")
        
        self.memory_classifier.signal_memory = []
        
        with torch.no_grad():
            self.memory_classifier.memory_weight.fill_(0.2)
            self.memory_classifier.feature_weight.fill_(0.8)
        
 
        noise_metrics = {'novelty': 0.9, 'resonance': 0.2, 'disruption': 0.3, 'stability': 0.4}
        noise_features = torch.randn(64, device=self.device)  # Random features for noise
        noise_entry = {
            'features': noise_features,
            'metrics': noise_metrics,
            'signal_type': 'noise',
            'confidence': 0.8,
            'timestamp': time.time() - 1000  # Make it look older
        }
        self.memory_classifier.signal_memory.append(noise_entry)
        
        challenge_metrics = {'novelty': 0.5, 'resonance': 0.2, 'disruption': 0.9, 'stability': 0.3}
        challenge_features = torch.ones(64, device=self.device) * 0.2  # Different pattern
        challenge_entry = {
            'features': challenge_features,
            'metrics': challenge_metrics,
            'signal_type': 'challenge',
            'confidence': 0.9,
            'timestamp': time.time() - 500
        }
        self.memory_classifier.signal_memory.append(challenge_entry)
        
        # but only one to avoid bias
        seed_metrics = {'novelty': 0.3, 'resonance': 0.8, 'disruption': 0.2, 'stability': 0.7}
        seed_features = torch.ones(64, device=self.device) * -0.2  
        seed_entry = {
            'features': seed_features,
            'metrics': seed_metrics,
            'signal_type': 'seed',
            'confidence': 0.7,
            'timestamp': time.time() - 200
        }
        self.memory_classifier.signal_memory.append(seed_entry)
        
        print(f"Initialized classifier with balanced memory examples (weights: feature={self.memory_classifier.feature_weight.item():.2f}, memory={self.memory_classifier.memory_weight.item():.2f})")


    def _update_memory(self, features, metrics, signal_type, confidence):
        entry = {
            'features': features.detach().clone() if features is not None else None,
            'metrics': {k: v for k, v in metrics.items()},
            'signal_type': signal_type,
            'confidence': confidence,
            'timestamp': time.time()
        }

        self.signal_memory.append(entry)

        if len(self.signal_memory) > self.memory_capacity:
            if len(self.signal_memory) > self.memory_capacity * 1.5:
                cutoff = int(self.memory_capacity * 0.2)

                self.signal_memory.sort(key=lambda x:
                    x['confidence'] + 0.2 * (time.time() - x.get('timestamp', 0)) / 3600)

                self.signal_memory = self.signal_memory[cutoff:]
            else:
                self.signal_memory = self.signal_memory[1:]

    def _track_classification(self, signal_type, confidence, original_type, was_revised):
        self.signal_counts[signal_type] += 1

        if len(self.last_classifications) >= 20:
            self.last_classifications.pop(0)
        self.last_classifications.append(signal_type)

        if len(self.confidence_history) >= 100:
            self.confidence_history.pop(0)
        self.confidence_history.append(confidence)

        if was_revised:
            revision = {
                'original': original_type,
                'final': signal_type,
                'confidence': confidence,
                'timestamp': time.time()
            }
            self.revision_history.append(revision)

            if len(self.revision_history) > 100:
                self.revision_history.pop(0)

    def _update_weights(self):
        """Fixed weight update mechanism"""
        if len(self.revision_history) < 5 or random.random() > 0.3:
            return
        
        recent_revisions = self.revision_history[-5:]
        revision_rate = len(recent_revisions) / 5.0
        
        # for monitoring 
        old_memory_weight = self.memory_weight.item()
        old_feature_weight = self.feature_weight.item()
        
        # much smaller than before
        with torch.no_grad():
            if revision_rate > 0.5:  # means memory is wrong
                adjustment = 0.01 * revision_rate
                self.memory_weight.sub_(adjustment)
                self.feature_weight.add_(adjustment)
            elif revision_rate < 0.1 and len(self.signal_memory) > 20:
                # Memory is reliable 
                adjustment = 0.005
                self.memory_weight.add_(adjustment)
                self.feature_weight.sub_(adjustment)
        
        self._normalize_weights()
        
        if abs(old_memory_weight - self.memory_weight.item()) > 0.01:
            print(f"Weight update: {old_memory_weight:.2f}/{old_feature_weight:.2f} -> {self.memory_weight.item():.2f}/{self.feature_weight.item():.2f}")

    def _normalize_weights(self):
        """Ensure weights sum to 1.0 and are non-negative"""
        with torch.no_grad():
            # Safety check for negative values
            if self.memory_weight < 0:
                self.memory_weight.fill_(0.1)
            if self.feature_weight < 0:
                self.feature_weight.fill_(0.1)
                
            total = self.memory_weight + self.feature_weight
            self.memory_weight.div_(total)
            self.feature_weight.div_(total)

    def _update_weights_with_balance_correction(self):
        self._update_weights()
        
        recent_types = [entry['signal_type'] for entry in self.last_classifications[-20:]]
        type_counts = {t: recent_types.count(t) for t in self.signal_types}
        
        total = sum(type_counts.values())
        if total > 0:
            type_ratios = {t: type_counts[t]/total for t in self.signal_types}
            
            dominant_type = max(type_ratios, key=type_ratios.get)
            if type_ratios[dominant_type] > 0.6:
                print(f"Applying balance correction for dominant type: {dominant_type} ({type_ratios[dominant_type]:.2f})")
                
                with torch.no_grad():
                    # Strengthen memory influence to break pattern
                    self.memory_weight.add_(0.1)
                    self.feature_weight.sub_(0.1)
                    
                    self._normalize_weights()

    def apply_weight_neutrality_decay(self, steps=10, decay_factor=0.95):
        """Apply decay towards neutrality (0.5) every N steps"""
        # Only apply if it's time
        if self.step % steps != 0:
            return

        with torch.no_grad():
            self.memory_weight.mul_(decay_factor).add_((1 - decay_factor) * 0.5)
            self.feature_weight.mul_(decay_factor).add_((1 - decay_factor) * 0.5)
            
            self._normalize_weights()
            
            print(f"Applied neutrality decay. Weights: {self.memory_weight.item():.4f}/{self.feature_weight.item():.4f}")

    def inject_entropy(self):
        """
        Periodically introduce entropy to prevent stagnation
        in classification behaviors
        """

        if len(self.last_classifications) >= 10:
            type_counts = {}
            for t in self.signal_types:
                type_counts[t] = self.last_classifications.count(t)

            dominant_type = max(type_counts, key=type_counts.get)
            dominant_ratio = type_counts[dominant_type] / len(self.last_classifications)

            if dominant_ratio > 0.7:
                print(f"STRONG entropy injection: system is stuck classifying {dominant_ratio:.1%} signals as {dominant_type}")
                with torch.no_grad():
                    if dominant_type == "seed":
                        self._adjust_classification_bias(boost_types=["noise", "challenge"], 
                                                    reduce_types=["seed"], 
                                                    boost_factor=0.3)
                        
                    elif dominant_type == "noise":
                        # Strengthen 
                        self._adjust_classification_bias(boost_types=["seed", "challenge"], 
                                                    reduce_types=["noise"], 
                                                    boost_factor=0.3)
                    else:
                        # Strengthen 
                        self._adjust_classification_bias(boost_types=["noise", "seed"], 
                                                    reduce_types=["challenge"], 
                                                    boost_factor=0.3)
                    
                    # Forcing exploration by boosting memory
                    self.memory_weight.fill_(0.8)
                    self.feature_weight.fill_(0.2)
                    self._normalize_weights()
                
                print(f"Adjusted weights to memory={self.memory_weight.item():.2f}, feature={self.feature_weight.item():.2f}")
                
            self.last_classifications = []

    def apply_oscillatory_cycle(self, cycle_length=20):
        """
        Apply oscillatory cycle to parameters to prevent getting stuck
        This creates natural rhythms of exploration and exploitation
        """
        if self.step % cycle_length != 0:
            return
        
        phase = (self.step / cycle_length) % 1.0
        
        with torch.no_grad():
            memory_bias = 0.1 * math.sin(2 * math.pi * phase)
            self.memory_weight.add_(memory_bias)
            self.feature_weight.sub_(memory_bias)
            self._normalize_weights()
            
            threshold_bias = 0.05 * math.sin(2 * math.pi * phase + math.pi/3)  # Offset phase
            self.signal_router.noise_threshold.mul_(1.0 + threshold_bias)
            self.threshold_layer.seed_pass_ratio.mul_(1.0 - threshold_bias)
            
            print(f"Oscillatory cycle at phase {phase:.2f}, memory weight: {self.memory_weight.item():.2f}")

    def get_revision_stats(self):
        if not self.revision_history:
            return {
                'revision_rate': 0.0,
                'most_common_revision': None,
                'revision_patterns': {}
            }

        revision_rate = len(self.revision_history) / max(1, sum(self.signal_counts.values()))

        revision_patterns = {}
        for rev in self.revision_history:
            pattern = f"{rev['original']} → {rev['final']}"
            if pattern in revision_patterns:
                revision_patterns[pattern] += 1
            else:
                revision_patterns[pattern] = 1

        sorted_patterns = sorted(revision_patterns.items(), key=lambda x: x[1], reverse=True)

        most_common = sorted_patterns[0][0] if sorted_patterns else None

        return {
            'revision_rate': revision_rate,
            'most_common_revision': most_common,
            'revision_patterns': dict(sorted_patterns),
            'total_classifications': sum(self.signal_counts.values()),
            'total_revisions': len(self.revision_history)
        }
    
    def _detect_and_correct_systematic_errors(self):
        """Detect systematic errors in classification and adjust parameters"""
        if len(self.revision_history) < 10:
            return
        
        revision_patterns = {}
        for rev in self.revision_history[-10:]:
            pattern = f"{rev['original']} → {rev['final']}"
            revision_patterns[pattern] = revision_patterns.get(pattern, 0) + 1
        
        if revision_patterns:
            most_common = max(revision_patterns.items(), key=lambda x: x[1])
            pattern, count = most_common
            
            if count > 3:
                orig_type, final_type = pattern.split(' → ')
                print(f"Detected systematic error: {pattern} ({count} times)")
                
                with torch.no_grad():
                    if orig_type == 'seed' and final_type == 'noise':
                        self.threshold_adjustment = {
                            'seed': -0.15,  # seed harder
                            'noise': +0.1   # noise easier
                        }
                        # will add more patterns as needed
       
    # Keeping memory reset as a function but only call it in extreme cases
    def conditional_memory_reset(self):
        """Only reset memory when classifications are extremely imbalanced"""
        memory_types = {}
        for entry in self.signal_memory:
            signal_type = entry.get('signal_type')
            memory_types[signal_type] = memory_types.get(signal_type, 0) + 1
        
        total = len(self.signal_memory)
        if total < 10:  # Need enough entries to make a decision
            return False
            
        dominant_type = max(memory_types, key=memory_types.get)
        dominant_ratio = memory_types[dominant_type] / total
        
        # Only reset in extreme cases
        if dominant_ratio > 0.9: 
            print(f"MEMORY RESET: Extreme {dominant_type} dominance ({dominant_ratio:.1%})")
            self.reset_memory()
            return True
            
        return False
