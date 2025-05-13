"""
Enhanced Fingerprinting for Ecliphra

This module provides pattern-aware fingerprinting to improve
pattern differentiation and memory diversity in Ecliphra.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class EnhancedFingerprinting:
    """
    Enhanced fingerprinting methods that can be integrated with Ecliphra.

    This class provides the ability to create pattern specific fingerprints
    that preserve the distinctive features of different pattern types.
    """

    def __init__(self, field_dim=(32, 32), device='cpu'):
        """
        Initialize the enhanced fingerprinting system.

        Args:
            field_dim: Dimensions of the field (height, width)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.field_dim = field_dim
        self.device = device

        # Cache some calculations for efficiency
        self._init_spatial_indices()

        self.pattern_thresholds = {
            'stripe_ratio': 1.5,       
            'circular_symmetry': 0.7,  
            'spiral_variance': 0.001   
        }

    def _init_spatial_indices(self):
        """Initialize spatial indices for pattern analysis."""
        h, w = self.field_dim
        center_y, center_x = h // 2, w // 2

        # coordinate grids
        y_indices = torch.arange(h, device=self.device).float()
        x_indices = torch.arange(w, device=self.device).float()

        # normalized coordinate grids
        y_norm = (y_indices / h).view(-1, 1).expand(-1, w)
        x_norm = (x_indices / w).view(1, -1).expand(h, -1)

        y_dist = ((y_indices.view(-1, 1) - center_y) / h) ** 2
        x_dist = ((x_indices.view(1, -1) - center_x) / w) ** 2
        dist_matrix = torch.sqrt(y_dist + x_dist)

        y_offset = (y_indices.view(-1, 1) - center_y).expand(-1, w)
        x_offset = (x_indices.view(1, -1) - center_x).expand(h, -1)
        angle_matrix = torch.atan2(y_offset, x_offset)

        # store for later
        self.y_norm = y_norm
        self.x_norm = x_norm
        self.dist_matrix = dist_matrix
        self.angle_matrix = angle_matrix
        self.center = (center_y, center_x)

    def create_robust_fingerprint(self, input_tensor):
        """
        Create pattern aware fingerprint that keeps distinctive features.

        Args:
            input_tensor: Input pattern tensor

        Returns:
            Fingerprint vector optimized for the detected pattern type
        """
        input_tensor = input_tensor.to(self.device)

        pattern_type, pattern_stats = self.detect_pattern_type(input_tensor)
        print(f"[DEBUG] Detected pattern type: {pattern_type}")

         # Trying to ensure consistent dimensionality
        max_features = 64  # Had to go fixed size

        # Base fingerprint
        if pattern_type == "stripes":
            fingerprint = self.create_frequency_fingerprint(input_tensor, pattern_stats)
        elif pattern_type == "spiral":
            fingerprint = self.create_spiral_fingerprint(input_tensor, pattern_stats)
        elif pattern_type == "blob":
            fingerprint = self.create_spatial_fingerprint(input_tensor, pattern_stats)
        else:
            fingerprint = self.create_generic_fingerprint(input_tensor)

        fingerprint = F.normalize(fingerprint, p=2, dim=0)

        # Ensure consistent dimensionality
        if fingerprint.shape[0] > max_features:
            fingerprint = fingerprint[:max_features]
        elif fingerprint.shape[0] < max_features:
            padding = torch.zeros(max_features - fingerprint.shape[0], device=self.device)
            fingerprint = torch.cat([fingerprint, padding])

        type_encoding = self.encode_pattern_type(pattern_type) * 2.0
        fingerprint = torch.cat([fingerprint, type_encoding])

        fingerprint = F.normalize(fingerprint, p=2, dim=0)

        return fingerprint, pattern_type

    def detect_pattern_type(self, input_tensor):
        """
        Based on spatial statistics.

        Args:
            input_tensor: Input pattern tensor

        Returns:
            Tuple of (pattern_type, pattern_stats)
        """
        h_var = torch.var(input_tensor, dim=0).mean().item()
        v_var = torch.var(input_tensor, dim=1).mean().item()
        directional_ratio = max(h_var, v_var) / (min(h_var, v_var) + 1e-10)

        circular_stats = self.calculate_circular_statistics(input_tensor)
        circular_symmetry = circular_stats['circular_symmetry']

        spiral_stats = self.calculate_spiral_statistics(input_tensor)
        spiral_angular = spiral_stats['spiral_strength']

        spiral_radial = self.detect_spiral_features(input_tensor)

        spiral_strength = (spiral_angular + spiral_radial) / 2.0

        traits = {
            'h_var': h_var,
            'v_var': v_var,
            'directional_ratio': directional_ratio,
            'dominant_direction': 'horizontal' if h_var > v_var else 'vertical',
            'circular_symmetry': circular_symmetry,
            'spiral_angular': spiral_angular,
            'spiral_radial': spiral_radial,
            'spiral_strength': spiral_strength
        }
            # different patterns have different frequency distributions
        fft = torch.fft.rfft2(input_tensor)
        fft_mag = torch.abs(fft)
        high_freq_energy = torch.sum(fft_mag[fft_mag.shape[0]//2:, fft_mag.shape[1]//2:]) / torch.sum(fft_mag)
        
        # entropy
        normalized = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-8)
        bins = 10
        hist = torch.histc(normalized, bins=bins, min=0, max=1)
        hist = hist / torch.sum(hist)
        entropy = -torch.sum(hist * torch.log2(hist + 1e-10))
        
        # Enhanced pattern detection criteria
        traits.update({
            'high_freq_energy': high_freq_energy.item(),
            'entropy': entropy.item(),
            'h_v_ratio': h_var / max(v_var, 1e-10)
        })

        print(f"[DEBUG] Pattern detection stats — "
            f"directional_ratio: {directional_ratio:.3f}, "
            f"circular_symmetry: {circular_symmetry:.3f}, "
            f"spiral_strength: {spiral_strength:.6f}")

        # Pattern classification logic
        if spiral_strength > 1e-5 and circular_symmetry > 0.9:
            pattern_type = "spiral"
        elif directional_ratio > 1.5:
            pattern_type = "stripes"
        elif circular_symmetry > 0.7:
            pattern_type = "blob"
        else:
            pattern_type = "other"

        return pattern_type, traits

    def detect_spiral_features(self, input_tensor):
        """Improved spiral detection."""
        # Look at how intensity changes along radial lines
        spiral_scores = []

        # Check multiple angles
        for angle_degrees in range(0, 360, 15):
            angle = np.radians(angle_degrees)
            dx, dy = np.cos(angle), np.sin(angle)

            # Sample along the radial line
            samples = []
            for r in np.linspace(0.1, 0.9, 20):
                x = int((0.5 + r * dx) * self.field_dim[1])
                y = int((0.5 + r * dy) * self.field_dim[0])

                # Ensure within bounds
                x = min(max(0, x), self.field_dim[1]-1)
                y = min(max(0, y), self.field_dim[0]-1)

                samples.append(input_tensor[y, x].item())

            # Calculate how often the signal crosses zero or changes sign
            # More zero crossings makes it more spiral like
            samples = torch.tensor(samples)
            zero_crossings = ((samples[:-1] * samples[1:]) < 0).sum().item()
            spiral_scores.append(zero_crossings)

        # Higher max score stronger spiral features
        return max(spiral_scores) / 10.0  

    def calculate_circular_statistics(self, input_tensor):
        """
        Getting results for circular patterns

        Args:
            input_tensor: Input pattern tensor

        Returns:
            Dictionary of circular pattern statistics
        """
        #  For different radii
        circular_var = []
        ring_values = []
        center_y, center_x = self.center

        for r in np.linspace(0.1, 0.5, 5):  # Check rings at different normalized radii

            ring_mask = ((self.dist_matrix > r-0.02) & (self.dist_matrix < r+0.02)).float()

            if ring_mask.sum() > 0:
                # Calculate variance along the ring
                ring_vals = input_tensor * ring_mask
                ring_sum = ring_vals.sum()
                ring_mean = ring_sum / ring_mask.sum()
                ring_variance = (((ring_vals - ring_mean) ** 2) * ring_mask).sum() / ring_mask.sum()

                circular_var.append(ring_variance.item())
                ring_values.append(ring_mean.item())

        # lower will mean more circular symmetry
        avg_circular_var = sum(circular_var) / len(circular_var) if circular_var else 1.0
        circular_symmetry = 1.0 - min(avg_circular_var * 10, 1.0)  # Scale and invert

        # Detect center of mass to see if it's centered
        total_mass = input_tensor.sum().item()
        if total_mass > 0:
            y_center = torch.sum(self.y_norm * input_tensor) / total_mass
            x_center = torch.sum(self.x_norm * input_tensor) / total_mass

            # Distance from geometric center
            center_offset = torch.sqrt((y_center - 0.5)**2 + (x_center - 0.5)**2).item()
        else:
            center_offset = 0.5  

        return {
            'circular_symmetry': circular_symmetry,
            'center_offset': center_offset,
            'ring_values': ring_values
        }


    def calculate_spiral_statistics(self, input_tensor):
        """
        Getting results for sprial patterns

        Args:
            input_tensor: Input pattern tensor

        Returns:
            Dictionary of spiral pattern statistics
        """
        # For spiral detection will need to check how values change with angle at each radius
        angle_derivatives = []

        #print("[SPIRAL DEBUG] Calculating spiral strength") # dont need right now

        num_angle_bins = 36  # 10-degree bins
        angle_bins = torch.linspace(-np.pi, np.pi, num_angle_bins+1, device=self.device)

        for r in np.linspace(0.1, 0.4, 4):  # Check several radii
            ring_mask = ((self.dist_matrix > r-0.03) & (self.dist_matrix < r+0.03)).float()

            if ring_mask.sum() > 0:
                bin_values = torch.zeros(num_angle_bins, device=self.device)
                bin_counts = torch.zeros(num_angle_bins, device=self.device)

                for i in range(num_angle_bins):
                    angle_mask = ((self.angle_matrix >= angle_bins[i]) &
                                (self.angle_matrix < angle_bins[i+1])).float()

                    combined_mask = ring_mask * angle_mask
                    if combined_mask.sum() > 0:
                        bin_values[i] = (input_tensor * combined_mask).sum() / combined_mask.sum()
                        bin_counts[i] = combined_mask.sum()

                # Calculate derivatives between adjacent 
                valid_bins = bin_counts > 0
               # print(f"[SPIRAL DEBUG] Radius {r:.2f} — valid bins: {valid_bins.sum()}") # dont need right now

                if valid_bins.sum() > 2:
                    valid_values = bin_values[valid_bins]
                   # print(f"[SPIRAL DEBUG] Bin values: {bin_values}") # dont need right now

                    # Getting the change between adjacent bins
                    if len(valid_values) > 1:  # It must have at least 2 valid values
                        derivatives = torch.abs(valid_values[1:] - valid_values[:-1])
                     #   print(f"[SPIRAL DEBUG] Derivatives: {derivatives}") # dont need right now

                        # High derivative variance indicates spiral pattern
                        if len(derivatives) > 0:
                            der_var = torch.var(derivatives).item()
                          #  print(f"[SPIRAL DEBUG] Derivative variance: {der_var}") # dont need right now
                            angle_derivatives.append(der_var)
                else:
                    print("[SPIRAL DEBUG] Not enough valid bins")

        # Trying to find spiral strength from angle derivatives
        spiral_strength = sum(angle_derivatives) / len(angle_derivatives) if angle_derivatives else 0
        print(f"[SPIRAL DEBUG] Final spiral strength: {spiral_strength}")

        return {
            'spiral_strength': spiral_strength,
            'angle_derivatives': angle_derivatives
        }

    def plot_angular_profile(self, input_tensor, radius=0.25, num_bins=36):
        """
        Visualize angle-binned values at a given radius.
        Useful for debugging spiral detection.
        """
        import matplotlib.pyplot as plt

        ring_mask = ((self.dist_matrix > radius - 0.03) & (self.dist_matrix < radius + 0.03)).float()
        angle_bins = torch.linspace(-np.pi, np.pi, num_bins + 1, device=self.device)

        bin_values = torch.zeros(num_bins, device=self.device)
        bin_counts = torch.zeros(num_bins, device=self.device)

        for i in range(num_bins):
            angle_mask = ((self.angle_matrix >= angle_bins[i]) &
                        (self.angle_matrix < angle_bins[i + 1])).float()
            combined_mask = ring_mask * angle_mask
            if combined_mask.sum() > 0:
                bin_values[i] = (input_tensor * combined_mask).sum() / combined_mask.sum()
                bin_counts[i] = combined_mask.sum()

        bin_values = bin_values.cpu().numpy()
        angles = np.linspace(0, 360, num_bins, endpoint=False)

        plt.figure(figsize=(10, 4))
        plt.plot(angles, bin_values, marker='o')
        plt.title(f"Angular Profile at Radius = {radius:.2f}")
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Average Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def create_frequency_fingerprint(self, input_tensor, pattern_stats):
        """
        Create fingerprint saving frequency information for striped patterns.

        Args:
            input_tensor: Input pattern tensor
            pattern_stats: Statistics from pattern detection

        Returns:
            Frequency-based fingerprint vector
        """
        # capturing frequency domain info
        fft = torch.fft.rfft2(input_tensor)
        fft_mag = torch.abs(fft)

        # Extract key frequency components
        features = []

        # Carefully adding directional info
        if 'dominant_direction' in pattern_stats:
            is_horizontal = pattern_stats['dominant_direction'] == 'horizontal'
            direction_feature = torch.tensor([1.0, 0.0] if is_horizontal else [0.0, 1.0],
                                            device=self.device)
        else:
            # If vertical is missing
            direction_feature = torch.tensor([0.0, 1.0], device=self.device)

        features.append(direction_feature)

        try:
            if 'dominant_direction' in pattern_stats and pattern_stats['dominant_direction'] == 'horizontal':
                row_avg = fft_mag.mean(dim=1)
                _, top_rows = torch.topk(row_avg[1:min(16, len(row_avg))], k=3)
                dominant_freqs = (top_rows + 1).float() / input_tensor.shape[0]
            else:
                col_avg = fft_mag.mean(dim=0)
                _, top_cols = torch.topk(col_avg[1:min(16, len(col_avg))], k=3)
                dominant_freqs = (top_cols + 1).float() / input_tensor.shape[1]

            features.append(dominant_freqs)
        except Exception as e:
            print(f"[WARNING] Error extracting dominant frequencies: {e}")
            # if extraction fails
            features.append(torch.tensor([0.1, 0.2, 0.3], device=self.device))

        # Safely adding stripe intensity
        if 'h_var' in pattern_stats and 'v_var' in pattern_stats:
            is_horizontal = pattern_stats.get('dominant_direction', '') == 'horizontal'
            stripe_contrast = (pattern_stats['h_var'] if is_horizontal else pattern_stats['v_var'])
            features.append(torch.tensor([stripe_contrast], device=self.device))
        else:
            # if contrast is missing
            features.append(torch.tensor([0.2], device=self.device))

        # Add stripe width information (inverse of frequency)
        try:
            if len(dominant_freqs) > 0:
                stripe_width = input_tensor.shape[0] / (dominant_freqs[0] * input_tensor.shape[0])
                features.append(torch.tensor([stripe_width / input_tensor.shape[0]], device=self.device))
            else:
                # if calculation fails
                features.append(torch.tensor([0.2], device=self.device))
        except Exception as e:
            print(f"[WARNING] Error calculating stripe width: {e}")
            # if calculation fails
            features.append(torch.tensor([0.2], device=self.device))

        combined = torch.cat([f.float() for f in features])
        print(f"[DEBUG] Frequency fingerprint shape: {combined.shape}")
        return combined

    def create_spiral_fingerprint(self, input_tensor, pattern_stats):
        """
        Making fingerprint preserving spiral characteristics.

        Args:
            input_tensor: Input pattern tensor
            pattern_stats: Statistics from pattern detection

        Returns:
            Spiral-optimized fingerprint vector
        """
        features = []

        if 'center_offset' in pattern_stats:
            features.append(torch.tensor([pattern_stats['center_offset']], device=self.device))
        else:
            features.append(torch.tensor([0.5], device=self.device))

        if 'spiral_strength' in pattern_stats:
            features.append(torch.tensor([pattern_stats['spiral_strength']], device=self.device))
        else:
            features.append(torch.tensor([0.1], device=self.device))

        if 'ring_values' in pattern_stats and pattern_stats['ring_values']:
            ring_values = torch.tensor(pattern_stats['ring_values'], device=self.device)
            features.append(ring_values)
        else:
            features.append(torch.tensor([0.0] * 5, device=self.device))

        # clockwise vs counterclockwise
        spiral_direction = self.detect_spiral_direction(input_tensor)
        features.append(torch.tensor([spiral_direction], device=self.device))

        # how quickly it winds
        spiral_tightness = self.estimate_spiral_tightness(input_tensor)
        features.append(torch.tensor([spiral_tightness], device=self.device))

        # NEW Add pattern-specific position encoding
        center_y, center_x = self.center
        spiral_samples = []

        # this captures the distinctive spiral pattern
        for t in np.linspace(0, 4*np.pi, 16):  # Sample along 2 full rotations
            r = 0.1 + 0.3 * t / (4*np.pi)  # Gradually increasing radius
            y = int(center_y + r * np.sin(t) * self.field_dim[0]/2)
            x = int(center_x + r * np.cos(t) * self.field_dim[1]/2)

            # stay in bounds
            y = min(max(0, y), self.field_dim[0]-1)
            x = min(max(0, x), self.field_dim[1]-1)

            spiral_samples.append(input_tensor[y, x].item())

        features.append(torch.tensor(spiral_samples, device=self.device))

        combined = torch.cat([f.float() for f in features])
        print(f"[DEBUG] Spiral fingerprint shape: {combined.shape}")
        return combined



    def detect_spiral_direction(self, input_tensor):
        """
        Detect whether a spiral is clockwise or counterclockwise.

        Returns 1.0 for clockwise, -1.0 for counterclockwise, 0.0 if unclear.
        """
        # This is a simplified approx. 
        # Need a more accurate version that would trace the spiral path

        cw_score = 0
        ccw_score = 0

        for r in np.linspace(0.1, 0.4, 4):
            angle_samples = []

            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                y = int(self.center[0] + r * self.field_dim[0] * np.sin(angle))
                x = int(self.center[1] + r * self.field_dim[1] * np.cos(angle))

                y = max(0, min(y, self.field_dim[0] - 1))
                x = max(0, min(x, self.field_dim[1] - 1))

                angle_samples.append(input_tensor[y, x].item())

            # Checking for clockwise or counterclockwise pattern in the samples
            for i in range(len(angle_samples)):
                if angle_samples[i] > angle_samples[(i+1) % len(angle_samples)]:
                    cw_score += 1
                else:
                    ccw_score += 1

        # Which is dominant 
        if cw_score > ccw_score * 1.5:
            return 1.0  # Clockwise
        elif ccw_score > cw_score * 1.5:
            return -1.0  # Counterclockwise
        else:
            return 0.0  # Unclear

    def estimate_spiral_tightness(self, input_tensor):
        """
        Estimate how tightly wound a spiral is.

        Returns a value between 0 and 1, where higher values mean more tightly wound.
        """
        # Still simple
        angle_change_rates = []

        for r in np.linspace(0.1, 0.4, 4):
            ring_mask = ((self.dist_matrix > r-0.03) & (self.dist_matrix < r+0.03)).float()

            if ring_mask.sum() > 0:
                grad_x = torch.zeros_like(input_tensor)
                grad_y = torch.zeros_like(input_tensor)

                # Approx gradients for points on the ring
                grad_x[:, :-1] = (input_tensor[:, 1:] - input_tensor[:, :-1]) * ring_mask[:, :-1]
                grad_y[:-1, :] = (input_tensor[1:, :] - input_tensor[:-1, :]) * ring_mask[:-1, :]

                grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
                ring_grad = (grad_magnitude * ring_mask).sum() / (ring_mask.sum() + 1e-10)

                angle_change_rates.append(ring_grad.item())

        # Across all rings
        avg_change_rate = sum(angle_change_rates) / len(angle_change_rates) if angle_change_rates else 0

        # Higher values makes for tighter spiral
        tightness = min(avg_change_rate * 5, 1.0)

        return tightness

    def create_spatial_fingerprint(self, input_tensor, pattern_stats):
        """
        Create fingerprint based on spatial statistics for blob patterns.

        Args:
            input_tensor: Input pattern tensor
            pattern_stats: Statistics from pattern detection

        Returns:
            Spatial fingerprint vector
        """
        features = []

        if 'center_offset' in pattern_stats:
            features.append(torch.tensor([pattern_stats['center_offset']], device=self.device))
        else:
            features.append(torch.tensor([0.5], device=self.device))

        if 'circular_symmetry' in pattern_stats:
            features.append(torch.tensor([pattern_stats['circular_symmetry']], device=self.device))
        else:
            features.append(torch.tensor([0.5], device=self.device))

        if 'ring_values' in pattern_stats and pattern_stats['ring_values']:
            ring_values = torch.tensor(pattern_stats['ring_values'], device=self.device)
            features.append(ring_values)
        else:
            features.append(torch.tensor([0.0] * 5, device=self.device))

        moments = self.calculate_spatial_moments(input_tensor)
        features.append(moments)

        blob_size = self.estimate_blob_size(input_tensor)
        features.append(torch.tensor([blob_size], device=self.device))

        intensity = input_tensor.max().item()
        features.append(torch.tensor([intensity], device=self.device))

        combined = torch.cat([f.float() for f in features])
        print(f"[DEBUG] Spatial fingerprint shape: {combined.shape}")
        return combined

    def calculate_spatial_moments(self, input_tensor):
        total_mass = input_tensor.sum().item()
        if total_mass <= 0:
            return torch.zeros(4, device=self.device)

        y_center = torch.sum(self.y_norm * input_tensor) / total_mass
        x_center = torch.sum(self.x_norm * input_tensor) / total_mass

        # second moments 
        y_var = torch.sum(((self.y_norm - y_center) ** 2) * input_tensor) / total_mass
        x_var = torch.sum(((self.x_norm - x_center) ** 2) * input_tensor) / total_mass

        cov = torch.sum((self.y_norm - y_center) * (self.x_norm - x_center) * input_tensor) / total_mass

        return torch.tensor([y_center.item(), x_center.item(),
                           y_var.item() * 10, x_var.item() * 10],
                          device=self.device)

    def estimate_blob_size(self, input_tensor):
        threshold = (input_tensor.max() + input_tensor.min()) / 2
        contour = (input_tensor >= threshold).float()

        blob_area = contour.sum().item()

        normalized_size = blob_area / (self.field_dim[0] * self.field_dim[1])

        return normalized_size

    def create_generic_fingerprint(self, input_tensor):
        """
        This is for patterns that don't fit specific categories.

        Args:
            input_tensor: Input pattern tensor

        Returns:
            Generic fingerprint vector
        """
        # For generic patterns, use a combination of approaches
        features = []

        downsampled = F.adaptive_avg_pool2d(
            input_tensor.unsqueeze(0).unsqueeze(0), (8, 8)
        ).squeeze()

        # Flattened
        features.append(downsampled.reshape(-1))

        features.append(torch.tensor([
            input_tensor.mean().item(),
            input_tensor.std().item(),
            input_tensor.max().item(),
            input_tensor.min().item()
        ], device=self.device))

        return torch.cat([f.float() for f in features])

    def encode_pattern_type(self, pattern_type):
        """
        A one hot encoding of the pattern type.

        Args:
            pattern_type: String identifying the pattern type

        Returns:
            One-hot encoded tensor
        """
        pattern_types = ["stripes", "spiral", "blob", "other"]

        encoding = torch.zeros(len(pattern_types), device=self.device)
        if pattern_type in pattern_types:
            encoding[pattern_types.index(pattern_type)] = 1.0

        return encoding

    def visualize_pattern_detection(self, input_tensor, output_dir=None):
        """
        The pattern detection process.

        Args:
            input_tensor: Input pattern tensor
            output_dir: Directory to save visualization (optional)

        Returns:
            Pattern type and visualization figure
        """
        pattern_type, pattern_stats = self.detect_pattern_type(input_tensor)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(input_tensor.cpu().numpy(), cmap='viridis')
        axes[0].set_title(f"Original Pattern")

        if pattern_type == "stripes":
            fft = torch.fft.rfft2(input_tensor)
            fft_mag = torch.log(torch.abs(fft) + 1)
            axes[1].imshow(fft_mag.cpu().numpy(), cmap='inferno')
            axes[1].set_title(f"FFT Magnitude (log scale)")

            axes[2].bar(['Horizontal', 'Vertical'],
                      [pattern_stats['h_var'], pattern_stats['v_var']])
            axes[2].set_title(f"Directional Variance")

        elif pattern_type in ["spiral", "blob"]:
            axes[1].imshow(self.dist_matrix.cpu().numpy(), cmap='plasma')
            axes[1].set_title(f"Distance from Center")

            if pattern_stats['ring_values']:
                radii = np.linspace(0.1, 0.5, len(pattern_stats['ring_values']))
                axes[2].plot(radii, pattern_stats['ring_values'])
                axes[2].set_title(f"Radial Profile")
                axes[2].set_xlabel("Normalized Radius")
                axes[2].set_ylabel("Average Value")

        else:
            axes[1].imshow(self.angle_matrix.cpu().numpy(), cmap='hsv')
            axes[1].set_title(f"Angle from Center")

            axes[2].hist(input_tensor.cpu().numpy().flatten(), bins=30)
            axes[2].set_title(f"Value Distribution")

        plt.suptitle(f"Detected Pattern Type: {pattern_type.upper()}")

        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/pattern_detection.png")

        return pattern_type, fig
