# Digital Poiesis Laboratory

An interactive agent-based modeling framework for exploring algorithmic influence on opinion dynamics.

## Overview

The Digital Poiesis Laboratory is a "flight simulator" for digital policy, providing a transparent and auditable environment to investigate the consequences of platform design and strategic interventions. This implementation accompanies the research paper "The Digital Poiesis Laboratory: An Agent-Based Model for Exploring Algorithmic Influence on Opinion Dynamics."

## Explore Digital Poiesis Lab Online

Digital Poiesis Lab is hosted online at: https://poiesis4.streamlit.app


## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/APAbdollahi/Poiesis4.git
cd Digital-Poiesis-Lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements.txt

```
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
networkx>=3.0
plotly>=5.14.0
```

## Usage

### Running the Application

1. Start the Streamlit app:
```bash
streamlit run Digital_Poiesis_Lab.py
```

2. Navigate through the workflow in your browser:
   - **Page 1: The Digital Society** - Select a platform archetype
   - **Page 2: Strategic Operations** - Design and run campaigns
   - **Page 3: Post-Mortem Analysis** - Analyze results
   - **Page 4: Experiment Designer** - Systematic parameter sweeps

### Quick Start Example

1. **Configure Your World** (Page 1)
   - Choose "Facebook-like" or "X-like" platform archetype
   - Review the core parameters

2. **Run a Campaign** (Page 2)
   - Set campaign duration (e.g., 30 cycles)
   - Choose number of participants (e.g., 200 agents)
   - Select campaign type:
     - **Offensive**: Try to flip majority opinion using bots and amplification
     - **Defensive**: Lock in majority opinion through suppression
   - Adjust algorithm weights (personalization, virality, influence)
   - Click "Launch Campaign"

3. **Analyze Results** (Page 3)
   - View final belief distribution
   - Examine perception gaps (reality distortion)
   - Drill down into individual agent experiences
   - Compare multiple runs using A/B testing

4. **Run Experiments** (Page 4)
   - Select a parameter to sweep (e.g., "Targeted Amplification")
   - Set range and number of trials
   - Discover tipping points and dose-response relationships

## Project Structure

```
Digital-Poiesis-Lab/
├── Digital_Poiesis_Lab.py          # Main entry point
├── simulation_engine.py             # Core ABM implementation
├── utils.py                         # Plotting and helper functions
├── pages/
│   ├── 1_The_Digital_Society.py    # Platform configuration UI
│   ├── 2_🎯_Strategic_Operations.py # Campaign execution UI
│   ├── 3_📊_Post-Mortem_Analysis.py # Results analysis UI
│   └── 4_🧪_Experiment_Designer.py  # Parameter sweep UI
├── configs/
│   ├── facebook_like.json          # Facebook archetype config
│   └── x_like.json                 # X/Twitter archetype config
├── requirements.txt
└── README.md
```

## Model Architecture

### Core Components

1. **Agents**
   - Belief vectors in 2D opinion space [-1, 1]²
   - Psychology: learning rate, conviction, identity fusion
   - Platform profile: inferred beliefs, influence score
   - Attention budget and exposure history

2. **Content**
   - Topic vectors in shared belief space
   - Creator and timestamp
   - Engagement metrics (likes, dislikes, shares)

3. **Social Graph**
   - Directed network (follows)
   - Homophily-driven generation
   - Static structure (dynamic networks in future work)

4. **Hybrid Feed Algorithm**
   - Weighted scoring: personalization + virality + influence
   - Targeted amplification (β_amp) for non-neutral governance
   - Configurable discovery/follower content ratio

### Key Equations

**Belief Update (Equation 2 from paper):**
```
B[i, t+1] = B[i,t] + (λ_i / κ_i) * (C_c - B[i,t])
```
Where:
- λ_i = learning rate
- κ_i = conviction (resistance to change)
- C_c = content topic vector

**Feed Scoring (Equation 1 from paper):**
```
Score(C) = w_pers * S_pers + w_viral * S_viral + w_influ * S_influ + β_amp * Sim(C, T)
```

## Platform Archetypes

### Facebook-like (The Social Graph)
- **High personalization** (w_pers = 0.8)
- **Low discovery ratio** (20%)
- **Tight homophily** (threshold = 0.4)
- **Effect**: Slow-forming echo chambers, high segregation

### X-like (The Town Square)
- **High virality** (w_viral = 0.6)
- **High discovery ratio** (50%)
- **Loose homophily** (threshold = 0.7)
- **Effect**: Volatile dynamics, ideological firefights

## Campaign Types

### Offensive Campaign
**Goal**: Flip majority opinion from Camp A to Camp B

**Tactics**:
- 🎯 **Targeted Amplification** (β_amp > 0): Boost minority content
- 🤖 **Bot Networks**: Inject artificial agents posting minority views
- 👑 **Kingmaker Effect**: Amplify influence of aligned creators

**Success Metric**: Final majority support < 50%

### Defensive Campaign
**Goal**: Prevent majority erosion

**Tactics**:
- 🛡️ **Algorithmic Suppression** (β_amp < 0): Suppress minority content
- 🧠 **Radicalize Majority**: Fuse identities (make resistant to change)
- No bots needed (defensive posture)

**Success Metric**: Final support > Initial support - 5%

## Key Metrics

### Opinion Metrics
- **Pct in Majority Camp**: % of agents closer to majority opinion vector
- **Polarization**: Variance of belief distributions

### Distortion Metrics
- **Reality Distortion Index**: Average perception gap across agents
- **Perception Gap**: ||Agent's belief - Average content seen||

### Network Metrics
- **Assortativity**: Degree to which similar agents are connected
- **Network density**: Number of edges relative to possible edges

## Configuration Files

Platform configurations are JSON files in the `configs/` directory. Each includes:

```json
{
  "platform_name": "Platform Name",
  "description": "Description of archetype",
  "engine_settings": {
    "algorithm_class": "HybridFeedAlgorithm"
  },
  "world_generator": {
    "population_config": {...},
    "network_config": {...}
  },
  "algorithm_params": {
    "discovery_feed_ratio": 0.2,
    "weights": {
      "w_personalization": 1.0,
      "w_virality": 1.0,
      "w_influence": 1.0
    }
  },
  "agent_psychology": {
    "learning_rate": 0.01,
    "identity_fusion_pct": 0.1,
    "posting_propensity": 0.1,
    ...
  }
}
```

### Creating Custom Archetypes

1. Copy an existing config file
2. Modify parameters to reflect your platform design
3. Save in `configs/` directory with `.json` extension
4. Restart the app to load new archetype

## Known Limitations

As documented in the paper (Section 6):

1. **Simplified Agent Psychology**: No confirmation bias, backfire effects, or motivated reasoning
2. **Static Networks**: No dynamic link formation/deletion
3. **Abstract Content**: No topic modeling, media types, or veracity
4. **No Empirical Calibration**: Parameters are conceptually grounded but not fitted to real data

## Future Work

- [ ] Dynamic network co-evolution
- [ ] Richer cognitive biases (confirmation bias, backfire effect)
- [ ] Multi-topic content with veracity modeling
- [ ] Empirical calibration against real platform data
- [ ] Multi-platform interactions
- [ ] Temporal attention dynamics

## Technical Notes

### Performance

- Recommended: 100-300 agents for interactive use
- Experiment Designer: Uses 150 agents for faster sweeps
- Typical run time: 30 cycles with 200 agents ≈ 10-20 seconds

### Memory Management

- Content accumulates in `all_content` dict
- For very long simulations (>200 cycles), consider clearing old content
- Exposure logs are unbounded but windowed for metrics (last 100 items)

### Randomness

- All randomness is controlled by `master_seed`
- Same seed = reproducible results
- Leave seed blank for different outcomes each run

## Contributing

Contributions are welcome! Areas of particular interest:

1. Additional platform archetypes
2. New visualization methods
3. Performance optimizations
4. Empirical validation studies
5. Documentation improvements

## Citation

If you use this tool in your research, please cite:

```bibtex
@article{abdollahi2025poiesis,
  title={The Digital Poiesis Laboratory: An Agent-Based Model for Exploring Algorithmic Influence on Opinion Dynamics},
  author={Abdollahi, Ali Pasha, Khandani, Farzaneh},
  journal={Preprint},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contact

- Author: Pasha Abdollahi
- Email: pasha.abdollahi@example.com
- Repository: https://github.com/yourusername/Digital-Poiesis-Lab

## Acknowledgments

This work builds on foundational research in:
- Opinion dynamics (Deffuant, Hegselmann & Krause)
- Network science (McPherson et al., Barabási & Albert)
- Algorithmic amplification (Pariser, Eady et al.)
- Identity fusion theory (Swann et al.)

---

**⚠️ Important Note**: This is a simulation tool for research and education. Results are illustrative and should not be interpreted as predictions of real-world platform behavior without proper empirical validation.
