# pages/1_🌎_The_Digital_Society.py
import streamlit as st

st.set_page_config(layout="wide", page_title="The Digital Society")
st.title("🌎 The Digital Society")
st.markdown("Define the fundamental physics of your digital world by selecting a platform archetype. This configuration will be used as the baseline for your campaigns.")
st.markdown("---")

# Load configs if they aren't already in the session state
if 'platform_configs' not in st.session_state:
    st.error("Configuration files not found. Please ensure the `configs` directory exists and is populated.")
else:
    platform_configs = st.session_state.platform_configs
    
    st.sidebar.header("World Configuration")
    platform_choice = st.sidebar.selectbox("Choose a Platform Archetype:", list(platform_configs.keys()))

    # When a choice is made, store the full config in the session state
    st.session_state.config = platform_configs[platform_choice]
    
    # Add scenario-specific vectors, which are not part of the base archetype
    if 'majority_opinion_vector' not in st.session_state.config:
        st.session_state.config['majority_opinion_vector'] = [0.5, 0.1]
    if 'minority_opinion_vector' not in st.session_state.config:
        st.session_state.config['minority_opinion_vector'] = [-0.5, -0.1]
    
    st.header(f"Archetype: {st.session_state.config['platform_name']}")
    st.info(st.session_state.config['description'])
    
    st.subheader("Core Parameters of this World")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Agent Psychology**")
        st.json(st.session_state.config['agent_psychology'])
    with col2:
        st.write("**Algorithmic Priorities**")
        st.json(st.session_state.config['algorithm_params'])
    with col3:
        st.write("**Network Structure**")
        st.json(st.session_state.config['world_generator']['network_config'])
        
    st.success("Your digital society is configured. You can now proceed to the **`2_🎯_Strategic_Operations`** page to design and run a campaign in this world.")