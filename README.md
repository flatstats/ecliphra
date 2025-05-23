# Ecliphra : Neural Field Dynamics with Emergent Pattern Recognition
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


# **Ecliphra Overview**

Ecliphra is an experimental AI framework that models dynamic tensor fields with emergent attractor patterns, enhanced fingerprinting, echo resonance, and semantic memory capabilities.

Inspired by neuroscientific principles, Ecliphra demonstrates pattern formation, memory persistence, and recognition through field dynamics rather than traditional static neural networks.

By treating cognition as an evolving energy process, Ecliphra explores how structured intelligence can arise from fluid, self-organizing fields.



# **Motivation**

Traditional AI architectures focus on fixed mappings between inputs and outputs, often struggling with long-term memory formation, goal modulation, and resilience under disruption.

Ecliphra challenges this paradigm by modeling intelligence as a self-organizing, energy-driven field system.  With this, attractors are able to form, memory structures are able to stabilize, and goal-directed biases can then shape field evolution without predefined endpoints.

By treating cognition as a dynamic, recursive process, Ecliphra aims to uncover deeper principles behind stability, adaptation, and emergent identity.


# **Architecture**

**Emergent System Flow**
Ecliphra’s cognition model emerges through a recursive process of energy redistribution, memory reinforcement, semantic drift, and goal-driven modulation.



```mermaid
flowchart TD
    subgraph Input
        I[Input Signal/Pattern] --> FP[Create Fingerprint]
    end

    subgraph "Core Field Dynamics"
        F[Field Tensor] --> L[Laplacian Computation]
        L --> V[Velocity Update]
        V --> FU[Field Update]
        FU --> N[Field Normalization]
        N --> A[Attractor Detection]
        A --> S[Stability Calculation]
        S --> F
    end

    subgraph "Echo Resonance"
        A --> EM[Attractor Memory Storage]
        EM --> ERA[Echo Resonance Check]
        ERA -->|Yes| EI[Echo Influence Injection]
        EI --> FU
    end

    subgraph "Semantic Processing"
        FP --> SM[Semantic Matching]
        SM --> SMA[Semantic Match Evaluation]
        SMA -->|Yes| SMI[Semantic Memory Influence]
        SMI --> FU
        A --> AF[Associate Fingerprints]
        AF --> EM
    end

    subgraph "Photosynthesis Signal Processing"
        I --> SR[Signal Router]
        SR --> MC[Memory-Enhanced Classifier]
        MC --> ST[Signal Type Analysis]
        ST --> TG[Threshold Gating]
        TG --> RP[Signal Rewriting Protocol]
        RP --> FU
    end

    subgraph "Energy System"
        I --> SA[Signal Energy Analyzer]
        SA --> ED[Energy Distributor]
        ED --> FM[Field Modulator]
        FM --> FU
        F --> AE[Available Energy Measurement]
        AE --> ED
    end

    subgraph "Prefrontal Control System"
        G[Goals] --> PFM[Prefrontal Modulation Module]
        WM[Working Memory Inputs] --> PFM
        F --> PFM
        ED --> PFM
        PFM --> |Field Modulation| FU
        PFM --> |Energy Redistribution| ED
        PFM --> IC[Inhibitory Control Signals]
        PFM --> AF[Attentional Focus Bias]
        IC --> F
        AF --> F
        ST --> GO[Goal Outcome Feedback]
        GO --> PFM
    end

    I --> F
    F --> Output[Emergent System Output]
```

# **System Components and Expansion Pathway**
Ecliphra’s modules are structured to build upon one another, creating increasingly complex cognitive dynamics.

```mermaid
classDiagram
    class EcliphraField {
        +field
        +velocity
        +adaptivity
        +stability
        +propagation
        +excitation
        +attractors[]
        +update_field_physics()
        +detect_attractors()
        +calculate_stability()
        +forward()
    }

    class EcliphraFieldWithEcho {
        +echo_strength
        +echo_decay
        +attractor_memory[]
        +apply_echo_resonance()
        +update_attractor_memory()
    }

    class EcliphraFieldWithSemantics {
        +semantic_threshold
        +fingerprinter
        +create_robust_fingerprint()
        +apply_semantic_matching()
        +associate_fingerprints()
    }

    class EcliphraFieldWithPhotoSynthesis {
        +signal_router
        +threshold_layer
        +rewriting_protocol
        +memory_classifier
        +get_classification_stats()
    }

    class EcliphraFieldWithEnergySystem {
        +signal_analyzer
        +energy_distributor
        +field_modulator
        +energy_capacity
        +energy_current
        +calculate_available_energy()
        +update_energy_state()
        +get_energy_statistics()
    }

    class EcliphraFieldWithPrefrontal {
        +prefrontal
        +set_goal()
        +inhibit_process()
        +focus_attention()
        +get_cognitive_metrics()
    }

    class EnhancedFingerprinting {
        +create_robust_fingerprint()
        +detect_pattern_type()
        +calculate_circular_statistics()
        +calculate_spiral_statistics()
    }

    class PrefrontalModule {
        +working_memory[]
        +current_goals[]
        +attention_mask
        +inhibition_field
        +set_goal()
        +update_goal_progress()
        +inhibit_process()
        +focus_attention()
        +allocate_resources()
    }
    class EnhancedFatigueController {
        +compute_fatigue_increment()
        +compute_fatigue_factor()
        +detect_unproductive_drift()
        +apply_rebalancing()
    }

    EcliphraField <|-- EcliphraFieldWithEcho
    EcliphraFieldWithEcho <|-- EcliphraFieldWithSemantics
    EcliphraFieldWithSemantics <|-- EcliphraFieldWithPhotoSynthesis
    EcliphraFieldWithSemantics <|-- EcliphraFieldWithEnergySystem
    EcliphraFieldWithEnergySystem <|-- EcliphraFieldWithPrefrontal

    EcliphraFieldWithSemantics --> EnhancedFingerprinting : uses
    EcliphraFieldWithPrefrontal --> PrefrontalModule : contains
    EcliphraFieldWithPrefrontal --> EnhancedFatigueController : uses
```

# **Key Features**

**Dynamic Field Physics**: This is the core implementation of tensor field dynamics with emergent attractor states.

**Echo Resonance**: Self-reinforcement mechanisms added for allowing the field to maintain and evolve coherent patterns over time.

**Enhanced Fingerprinting**: Robust pattern recognition created to be resilient to noise, variation, and perturbations.

**Semantic Memory**: Created for association and recall of semantically similar patterns through recursive reinforcement.

**Photosynthesis**:  (Retired for now, needs fixed) Energy-driven classification, growth, and integration of signals based on their structural characteristics.

**Energy System**: Adaptive resource allocation system inspired by biological energy regulation processes. This is a more functioning version of the photosynthesis model class.

**Prefrontal Capabilities**: Goal directed behavior modulation, working memory persistence, and executive control functions.  This is a simplified prototype to creating prefrontal cortex like functions

UPDATED **Enhanced Fatigue Controller**: Measures effort to how much has changed to see what is left after completing a goal.



# **Experiment Types**
Ecliphra supports multiple experiment types to test different aspects of emergent behavior:


**Semantic**: Test semantic fingerprinting and similarity recognition.

**Echo**: Test pattern maintenance over time through resonance.

**Noise**: Test field resistance to perturbations.

**Transition**: Study gradual field transitions between distinct patterns.

**Spiral Resume**: Test memory persistence across session resets.

**Photosynthesis**: Test signal classification, growth, and integration.

**Energy**: Test adaptive energy-driven resource distribution.

**Prefrontal**: Test goal directed processing and executive modulation. 


# **Output and Visualization**
Experiment results are saved in the specified output directory (default: timestamped directory in ecliphra_results/). Each experiment generates:


* Field state visualizations at key steps

* Evolution of field stability, attractors, and other metrics

* Detailed JSON data of results

* Experiment-specific visualizations (e.g., noise tolerance, energy allocation)

* Currently fatigue visuralizations are the only experiment handled by their own visualization tool. Hoping to move forward with this modular method.


# Getting Started
Prerequisites

Python 3.8 or higher

PyTorch 1.8 or higher

NumPy, Matplotlib

# Installation

Clone the repository:
```git clone https://github.com/flatstats/ecliphra.git cd ecliphra```
```pip install -e .```
```pip install ecliphra```



# Running Experiments

Use -m ecliphra.run to execute different types of experiments:
**Run a semantic experiment with semantics model for 40 steps** 

```python -m ecliphra.run --experiment semantic --model semantics --steps 40```

**Run an echo experiment with echo model for 30 steps with input every 5 steps**

```python -m ecliphra.run --experiment echo --model echo --steps 30 --input-frequency 5```

**Run a noise resistance experiment with multiple noise levels**

```python -m ecliphra.run --experiment noise --model semantics --noise-levels 0.1 0.2 0.3 0.4 0.5```

**Run a pattern transition experiment**

```python -m ecliphra.run --experiment transition --model semantics --transition-steps 15```

**Run a photosynthesis experiment**

```python -m ecliphra.run --experiment photosynthesis --model photo --steps 40 --field-size 32 --output-dir photo_experiment_results```

 **Run an energy system experiment**
 
```python -m ecliphra.run --experiment energy --model energy --steps 40 --field-size 32 --output-dir energy_experiment_results```

**Run a prefrontal capabilities experiment**

```python -m ecliphra.run --experiment prefrontal --model energy --prefrontal \```
    ```--steps 50 --field-size 32 --working-memory-size 7 --max-goals 3 \
    --output-dir prefrontal_experiments/test1```
    
**Run enhanced fatigue controller experiment**

 ```python -m ecliphra.run --experiment prefrontal --model energy --prefrontal --enhanced-fatigue --fatigue-recovery 0.45 --fatigue-decay 0.98 --fatigue-visualize --fatigue-debug --steps 60 --working-memory-size 7 --max-goals 3 --output-dir prefrontal_experiments/fatigue_test```

**Run all experiments with a specific model**

```python -m ecliphra.run --experiment all --model semantics --output-dir my_experiments```

# Technical Concepts

**Field Dynamics**

Ecliphra's core is a field that evolves according to physics inspired dynamics:

* Wave like propagation of perturbations
* Attractor detection and tracking
* Stability and resonance calculations

**Fingerprinting**

Enhanced fingerprinting provides:

* Pattern type detection such as stripes, spirals, and blobs
* Noise resisting recognition
* Semantic similarity matching

**Energy System**

The energy allocation system manages:

* Signal energy characteristics this includes intensity, coherence, and complexity
* Energy distribution across maintenance, growth, and adaptation pathways
* Field modulation based on energy allocation

**Prefrontal Control**

The prefrontal module implements:

* Goal setting and tracking
* Working memory for recent information
* Inhibitory control for focused processing
* Attentional resources allocation
* Decision making based on goals and context

# **Research Direction**

Future work includes:

* Enhancing stability and memory persistence under non-stationary noise.

* Deeper semantic memory drift resistance.

* Expansion of prefrontal control for multi-goal balancing.

* Exploration of multi-field networks and cooperative emergent cognition.


# **Ongoing Development**

Ecliphra is an active research framework under continuous development, in other words, it is far from perfect but continuing to develop just as I like to think I am as well.  I was inspired by the belief that emergence arises not from imposing structure onto chaos, but something like surfing the flow of hidden coherence within dynamic fields.

The system is currently stable across single field memory persistence tests, semantic matching, and energy modulation, with ongoing efforts focused on:

* Refining the field stabilization mechanics that were inspired by turbulence convergence.

* Expanding the prefrontal systems for more recursive goal arbitration.

* Modeling drift, collapse, and resilience when in identity structures across evolving fields.

* Ecliphra is not meant to be a static model but something of a living system that is continuously unfolding.

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.

Additional Restrictions:
Use of this codebase for training, fine-tuning, or dataset curation of commercial AI systems is prohibited without explicit permission from the author.

(c) 2025 Danielle Breegle



*"This is not the final form. It is the beginning of something that has not yet fully learned to become."*





