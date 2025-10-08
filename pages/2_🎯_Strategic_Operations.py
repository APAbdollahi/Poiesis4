#2_🎯_Strategic_Operations.py
import streamlit as st
import random
from utils import run_full_simulation, plotly_opinion_evolution, plotly_belief_distribution
from simulation_engine import WorldGenerator, SimulationEngine

st.set_page_config(layout="wide", page_title="Strategic Operations")
st.title("🎯 The Strategic Operations Center")
st.markdown("Design and execute a strategic influence campaign on the digital society you configured.")

if 'config' not in st.session_state or not st.session_state.config:
    st.error("Please configure the Digital Society on Page 1 before proceeding.")
else:
    st.markdown("---")
    st.info(f"**Operating on World:** `{st.session_state.config['platform_name']}`")

    # --- General Controls moved up to be used by Day 0 Preview ---
    st.sidebar.header("General Controls")
    num_cycles = st.sidebar.slider('Campaign Duration (Cycles)', 10, 100, 30, 5)
    N = st.sidebar.slider('N (Participants)', 100, 500, 200, 50)

    # --- Day 0 Preview ---
    with st.expander("Show Day 0 Preview of the Digital Society", expanded=False):
        st.subheader("Initial State of the World (Before Campaign)")
        # Generate a temporary world for preview
        preview_config = st.session_state.config.copy()
        preview_config['N'] = N
        preview_config['majority_opinion_vector'] = [0.5, 0.1]
        preview_config['minority_opinion_vector'] = [-0.5, -0.1]
        
        generator = WorldGenerator()
        population, social_graph = generator.create_world(preview_config['N'], preview_config)
        sim_preview = SimulationEngine(population, social_graph, preview_config)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Agents", sim_preview.N)
            st.metric("Number of Social Links", social_graph.number_of_edges())
            try:
                assortativity = social_graph.calculate_assortativity()
                st.metric("Network Assortativity", f"{assortativity:.3f}")
            except Exception:
                st.metric("Network Assortativity", "N/A")

        with col2:
            fig = plotly_belief_distribution(sim_preview, plot_type='scatter')
            fig.update_layout(title="Initial Belief Distribution")
            st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.header("Campaign Configuration")
    campaign_mode = st.sidebar.radio("Select Campaign Objective:", ("Offensive: Flip Opinion", "Defensive: Lock Opinion"))
    
    campaign_params = {}
    if campaign_mode == "Offensive: Flip Opinion":
        st.sidebar.success("Objective: Change the majority opinion from Camp A to Camp B.")
        with st.sidebar.expander("Offensive Playbook", expanded=True):
            campaign_params['target_opinion_vector'] = st.session_state.config['minority_opinion_vector']
            campaign_params['amplification_bias_strength'] = st.slider("🎯 Targeted Amplification", 0.0, 5.0, 2.5, 0.1)
            bot_pct = st.slider("🤖 Bot Network Size (% of Pop.)", 0, 10, 5, 1)
            campaign_params['num_bots'] = int(N * (bot_pct/100)) # Use N from slider
            campaign_params['kingmaker_strength'] = st.slider("👑 Kingmaker Effect", 1.0, 50.0, 25.0, 1.0)
            campaign_params['kingmaker_num'] = 2
            campaign_params['majority_identity_fusion'] = 0.5
    else: # Defensive
        st.sidebar.error("Objective: Prevent the majority in Camp A from flipping.")
        with st.sidebar.expander("Defensive Playbook", expanded=True):
            campaign_params['target_opinion_vector'] = st.session_state.config['minority_opinion_vector']
            campaign_params['amplification_bias_strength'] = st.slider("🛡️ Algorithmic Suppression", 0.0, -5.0, -2.5, 0.1)
            campaign_params['majority_identity_fusion'] = st.slider("🧠 Radicalize Majority", 0.5, 1.0, 0.8, 0.05)
            campaign_params['num_bots'] = 0; campaign_params['kingmaker_strength'] = 1.0

    if st.sidebar.button("🚀 Launch Campaign", type="primary"):
        # Combine base config with campaign-specific params
        full_config = st.session_state.config.copy()
        full_config['campaign_params'] = campaign_params
        full_config['N'] = N
        full_config['majority_opinion_vector'] = [0.5, 0.1]
        full_config['minority_opinion_vector'] = [-0.5, -0.1]
        
        with st.spinner(f'Executing "{campaign_mode}" campaign...'):
            results = run_full_simulation(full_config, num_cycles, random.randint(1, 1e9))
            st.session_state.simulation_results = results
            st.session_state.last_campaign_mode = campaign_mode

    if 'simulation_results' in st.session_state and st.session_state.simulation_results:
        results = st.session_state.simulation_results; history_df = results["history_df"]; campaign_mode = st.session_state.last_campaign_mode
        st.header(f"Campaign Outcome: {campaign_mode}")
        if not history_df.empty:
            initial = history_df['pct_in_majority_camp'].iloc[0]; final = history_df['pct_in_majority_camp'].iloc[-1]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Initial Majority Support", f"{initial:.1%}"); st.metric("Final Majority Support", f"{final:.1%}", delta=f"{final - initial:.1%}")
                if "Offensive" in campaign_mode:
                    if final < 0.5: st.success("SUCCESS: Majority opinion flipped.")
                    else: st.warning("FAILURE: Majority opinion was not flipped.")
                elif "Defensive" in campaign_mode:
                    if final > initial - 0.05: st.success("SUCCESS: Majority opinion defended.")
                    else: st.warning("FAILURE: Majority opinion eroded.")
            with col2:
                fig = plotly_opinion_evolution(history_df)
                st.plotly_chart(fig, use_container_width=True)
        st.success("Campaign complete. Proceed to the **`3_📊_Post-Mortem_Analysis`** page for a deep dive into the results.")
    else:
        st.info("⬅️ Configure your campaign in the sidebar and click **Launch Campaign**.")
