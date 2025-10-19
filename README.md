# Digital Poiesis Laboratory

An interactive agent-based modeling framework for exploring algorithmic influence on opinion dynamics.

## Overview

The Digital Poiesis Laboratory is a "flight simulator" for digital policy, providing a transparent and auditable environment to investigate the consequences of platform design and strategic interventions. This implementation accompanies the research paper "The Digital Poiesis Laboratory: An Agent-Based Model for Exploring Algorithmic Influence on Opinion Dynamics."

## Recent Updates (Bug Fixes)

This version includes critical bug fixes that improve the accuracy and functionality of the simulation:

### Major Fixes

1. **Engagement System Now Functional** ✅
   - Agents now properly like/dislike content based on belief alignment
   - Virality scores are calculated correctly from engagement
   - Influence scores update based on actual engagement metrics

2. **Defensive Campaign Radicalization** ✅
   - The "Radicalize Majority" slider now properly fuses identities
   - Majority camp agents can be made resistant to opinion change

3. **Improved Cosine Similarity Calculation** ✅
   - Added `safe_cosine_similarity()` function to handle edge cases
   - Prevents NaN values from zero vectors
   - More numerically stable

4. **Influence Score Updates Fixed** ✅
   - Now counts engagement from current cycle only (not all-time)
   - Uses exponential decay for more realistic dynamics
   - Prevents unbounded score growth

5. **Perception Gap Calculation Improved** ✅
   - Uses windowed exposure history (last 100 items)
   - Prevents memory issues in long simulations
   - More accurate reflection of recent experience

6. **Feed Composition Fixed** ✅
   - Backfills with discovery content when follower feed is sparse
   - Ensures agents always see a full feed

7. **Missing Import Added** ✅
   - Added `plotly.graph_objects` import to Post-Mortem Analysis page
   - Fixed A/B testing visualization

8. **Deprecated API Calls Updated** ✅
   - Changed `st.experimental_rerun()` to `st.rerun()`

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
