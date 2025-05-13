"""
Ecliphra - Physics-based tensor field with emergent identity

A modular implementation of tensor field based identity models with
echo resonance and semantic fingerprinting capabilities.

Version: 1.7.0

Models:
- EcliphraField: Base field implementation
- EcliphraFieldWithEcho: Adds echo resonance
- EcliphraFieldWithSemantics: Adds semantic fingerprinting
- EcliphraWithPersistentMemory
- EcliphraFieldWithPhotoSynthesis: May remove*
- EcliphraFieldWithEnergySystem: Stablizes and regulates patterns
- EcliphraFieldWithPrefrontal: Adds attention and goal processing

Experiments:
- SemanticExperiment: Tests semantic matching
- EchoExperiment: Tests echo resonance
- NoiseResistanceExperiment: Tests noise resistance
- TransitionExperiment: Tests pattern transitions
- SpiralResumeExperiment: Tests resume and memory retention with restarts.
- PhotosynthesisExperiment: Tests routing, filtering, and integrating different signals
- EnergySystemExperiment: Tests routing, allocating, and utilization across pathways
- PrefrontalExperiment: Tests field dynamics and goals (includes contradiction test and pattern interference)

Updated
Added curvature_riding to model
"""

from ecliphra.models import (
    EcliphraField,
    EcliphraFieldWithEcho,
    EcliphraFieldWithSemantics,
    LegacyIdentityField,
    create_ecliphra_model,
    EcliphraWithAdvancedMemory,
    EcliphraWithPersistentMemory,
    EcliphraFieldWithPhotoSynthesis,
    EcliphraFieldWithEnergySystem,
    EcliphraFieldWithPrefrontal,
    VERSIONS
)

from ecliphra.experiments import (
    EcliphraExperiment,
    SemanticExperiment,
    EchoExperiment,
    NoiseResistanceExperiment,
    TransitionExperiment,
    SpiralResumeExperiment,
    EnergySystemExperiment,
    PrefrontalExperiment,
    create_experiment
)

from ecliphra.utils.photofield import (
    SignalRouter,
    IntegrationThresholdLayer,
    FieldRewritingProtocol,
    EcliphraPhotoField,
    MemoryAugmentedClassifier
)

from ecliphra.utils.prefrontal import (
    PrefrontalModule,
    SignalEnvironment
    )

from ecliphra.utils.curvature_riding import (
    integrate_curvature_riding,
    integrate_curvature_riding_with_fatigue
)

from ecliphra.utils.enhanced_fatigue_controller import (
    EnhancedFatigueController
)

from ecliphra.visuals.fatigue_visualization import (
    visualize_fatigue_dynamics,
    create_fatigue_debug_dashboard
)


from ecliphra.utils.enhanced_fingerprinting import EnhancedFingerprinting

from ecliphra.utils.energy_system import (
    SignalAnalyzer,
    EnergyDistributor,
    FieldModulator,
    PhotosynthesisEnergySystem
)
