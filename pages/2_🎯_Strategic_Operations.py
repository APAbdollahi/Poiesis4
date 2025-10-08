#2_🎯_Strategic_Operations.py
import streamlit as st
import random
from utils import (run_full_simulation, plotly_opinion_evolution, 
                     plotly_belief_distribution, plotly_network_graph, 
                     plotly_reality_distortion_evolution)
from simulation_engine import WorldGenerator, SimulationEngine

st.set_page_config(layout="wide", page_title="Strategic Operations")
st.title("🎯 The Strategic Operations Center")
st.markdown("Design and execute a strategic influence campaign on the digital society you configured.")

if 'config' not in st.session_state or not st.session_state.config:
    st.error("Please configure the Digital Society on Page 1 before proceeding.")
else:
    st.markdown("---")
    st.info(f"**Operating on World:** `{st.session_state.config['platform_name']}`")

    # --- General Controls ---
    st.sidebar.header("General Controls")
    num_cycles = st.sidebar.slider('Campaign Duration (Cycles)', 10, 100, 30, 5)
    N = st.sidebar.slider('N (Participants)', 100, 500, 200, 50)

    # --- Day 0 Preview ---
    with st.expander("Show Day 0 Preview of the Digital Society", expanded=True):
        st.subheader("Initial State of the World (Before Campaign)")
        
        @st.cache_data(show_spinner="Generating preview...")
        def generate_preview_world(_config, _N):
            generator = WorldGenerator()
            config_copy = _config.copy()
            config_copy['N'] = _N
            population, social_graph = generator.create_world(_N, config_copy)
            sim_preview = SimulationEngine(population, social_graph, config_copy)
            return sim_preview

        preview_config = st.session_state.config.copy()
        preview_config['majority_opinion_vector'] = [0.5, 0.1]
        preview_config['minority_opinion_vector'] = [-0.5, -0.1]
        
        sim_preview = generate_preview_world(preview_config, N)

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Number of Agents", sim_preview.N)
        m_col2.metric("Number of Social Links", sim_preview.social_graph.number_of_edges())
        try:
            assortativity = sim_preview.social_graph.calculate_assortativity()
            m_col3.metric("Network Assortativity", f"{assortativity:.3f}")
        except Exception:
            m_col3.metric("Network Assortativity", "N/A")

        st.markdown("---")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            belief_fig = plotly_belief_distribution(sim_preview, plot_type='scatter')
            belief_fig.update_layout(title="Initial Belief Distribution")
            st.plotly_chart(belief_fig, use_container_width=True)
        with g_col2:
            network_fig = plotly_network_graph(sim_preview)
            st.plotly_chart(network_fig, use_container_width=True)

    st.markdown("---")
    
    # --- Algorithm Tuner ---
    st.header("⚙️ Algorithm Tuner & Campaign Strategy")
    with st.container(border=True):
        st.subheader("Core Algorithm Weights")
        algo_weights = st.session_state.config['algorithm_params']['weights']
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            w_pers = st.slider("Personalization (w_personalization)", 0.0, 5.0, algo_weights.get('w_personalization', 1.0), 0.1)
        with w_col2:
            w_viral = st.slider("Virality (w_virality)", 0.0, 5.0, algo_weights.get('w_virality', 1.0), 0.1)
        with w_col3:
            w_influ = st.slider("Influence (w_influence)", 0.0, 5.0, algo_weights.get('w_influence', 1.0), 0.1)

        st.markdown("---")
        st.subheader("Strategic Campaign")
        campaign_mode = st.radio("Select Campaign Objective:", ("Offensive: Flip Opinion", "Defensive: Lock Opinion"), horizontal=True)
        
        campaign_params = {}
        if campaign_mode == "Offensive: Flip Opinion":
            with st.expander("Offensive Playbook", expanded=True):
                st.success("Objective: Change the majority opinion from Camp A to Camp B.")
                campaign_params['target_opinion_vector'] = st.session_state.config['minority_opinion_vector']
                campaign_params['amplification_bias_strength'] = st.slider("🎯 Targeted Amplification", 0.0, 5.0, 2.5, 0.1)
                bot_pct = st.slider("🤖 Bot Network Size (% of Pop.)", 0, 10, 5, 1)
                campaign_params['num_bots'] = int(N * (bot_pct/100))
                campaign_params['kingmaker_strength'] = st.slider("👑 Kingmaker Effect", 1.0, 50.0, 25.0, 1.0)
                campaign_params['kingmaker_num'] = 2
        else: # Defensive
            with st.expander("Defensive Playbook", expanded=True):
                st.error("Objective: Prevent the majority in Camp A from flipping.")
                campaign_params['target_opinion_vector'] = st.session_state.config['minority_opinion_vector']
                campaign_params['amplification_bias_strength'] = st.slider("🛡️ Algorithmic Suppression", 0.0, -5.0, -2.5, 0.1)
                campaign_params['majority_identity_fusion'] = st.slider("🧠 Radicalize Majority", 0.5, 1.0, 0.8, 0.05)
                campaign_params['num_bots'] = 0
                campaign_params['kingmaker_strength'] = 1.0

    if st.sidebar.button("🚀 Launch Campaign", type="primary"):
        full_config = st.session_state.config.copy()
        
        # --- Apply settings from Algorithm Tuner ---
        full_config['algorithm_params']['weights']['w_personalization'] = w_pers
        full_config['algorithm_params']['weights']['w_virality'] = w_viral
        full_config['algorithm_params']['weights']['w_influence'] = w_influ
        full_config['campaign_params'] = campaign_params
        full_config['N'] = N
        full_config['majority_opinion_vector'] = [0.5, 0.1]
        full_config['minority_opinion_vector'] = [-0.5, -0.1]
        
        with st.spinner(f'Executing "{campaign_mode}" campaign...'):
            results = run_full_simulation(full_config, num_cycles, random.randint(1, 1e9))
            st.session_state.simulation_results = results
            st.session_state.last_campaign_mode = campaign_mode

    # --- Results Display ---
    if 'simulation_results' in st.session_state and st.session_state.simulation_results:
        results = st.session_state.simulation_results
        history_df = results["history_df"]
        campaign_mode = st.session_state.last_campaign_mode
        
        st.markdown("---")
        st.header(f"📈 Campaign Outcome: {campaign_mode}")
        
        if not history_df.empty:
            mcol1, mcol2 = st.columns(2)
            initial = history_df['pct_in_majority_camp'].iloc[0]
            final = history_df['pct_in_majority_camp'].iloc[-1]
            mcol1.metric("Initial Majority Support", f"{initial:.1%}")
            mcol2.metric("Final Majority Support", f"{final:.1%}", delta=f"{final - initial:.1%}")

            if "Offensive" in campaign_mode:
                if final < 0.5: st.success("SUCCESS: Majority opinion flipped.")
                else: st.warning("FAILURE: Majority opinion was not flipped.")
            elif "Defensive" in campaign_mode:
                if final > initial - 0.05: st.success("SUCCESS: Majority opinion defended.")
                else: st.warning("FAILURE: Majority opinion eroded.")

            st.markdown("---")
            
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                fig_opinion = plotly_opinion_evolution(history_df)
                st.plotly_chart(fig_opinion, use_container_width=True)
            with ccol2:
                fig_distortion = plotly_reality_distortion_evolution(history_df)
                st.plotly_chart(fig_distortion, use_container_width=True)

        st.success("Campaign complete. Proceed to the **`3_📊_Post-Mortem_Analysis`** page for a deep dive into the results.")
    else:
        st.info("⬅️ Configure your algorithm and campaign, then click **Launch Campaign** in the sidebar.")
