# Digital_Poiesis_Lab.py

import streamlit as st
from utils import initialize_session_state, load_platform_configs

# --- FIX: Moved set_page_config to be the VERY FIRST Streamlit command ---
st.set_page_config(
    page_title="Digital Poiesis Laboratory",
    page_icon="🌐",
    layout="wide"
)

# Now, it is safe to call other Streamlit functions
initialize_session_state()
if 'platform_configs' not in st.session_state:
    st.session_state.platform_configs = load_platform_configs()

# --- The rest of the file remains the same ---
st.title("🌐 Welcome to the Digital Poiesis Laboratory")
st.markdown("---")
st.subheader("An interactive laboratory for understanding how social reality is made and manipulated online.")
st.markdown("""
This application is a strategic sandbox built on an agent-based model of a social media ecosystem. It is designed to provide policymakers, researchers, and citizens with a tangible understanding of the "secret laws" that govern our digital public squares.

The laboratory is structured as a workflow, accessible via the pages in the sidebar:

1.  **🌎 The Digital Society:**
    *   **Purpose:** Define the fundamental physics of your digital world by selecting a platform archetype.

2.  **🎯 Strategic Operations:**
    *   **Purpose:** Design and execute a strategic influence campaign (offensive or defensive).

3.  **📊 Post-Mortem Analysis:**
    *   **Purpose:** Conduct a deep analysis of the campaign's consequences using advanced, interactive visualizations.

4.  **🧪 Experiment Designer:**
    *   **Purpose:** Run systematic experiments by "sweeping" a parameter across a range of values to find tipping points.

**To begin, navigate to the `1_🌎_The_Digital_Society` page in the sidebar.**
""")