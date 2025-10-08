#4_🧪_Experiment_Designer.py
import streamlit as st
import numpy as np
import pandas as pd
import random
from utils import run_full_simulation
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Experiment Designer")
st.title("🧪 Experiment Designer")
st.markdown("Systematically test the impact of a single parameter by 'sweeping' it across a range of values. This is a powerful tool for finding tipping points and understanding dose-response relationships.")
st.markdown("---")

if 'config' not in st.session_state or not st.session_state.config:
    st.error("Please configure the Digital Society on Page 1 before proceeding.")
else:
    st.sidebar.header("Experiment Setup")
    
    sweep_param_options = {'Targeted Amplification': ('campaign_params', 'amplification_bias_strength'), 'Algorithmic Discovery %': ('algorithm_params', 'discovery_feed_ratio'), 'Echo Chamber Strength': ('world_generator', 'network_config', 'homophily_threshold'), 'Belief Plasticity': ('agent_psychology', 'learning_rate')}
    param_display_name = st.sidebar.selectbox("Parameter to Sweep:", list(sweep_param_options.keys()))
    param_path = sweep_param_options[param_display_name]

    if param_display_name == 'Targeted Amplification': min_val, max_val, default_start, default_end = -5.0, 5.0, 0.0, 5.0
    elif param_display_name == 'Algorithmic Discovery %': min_val, max_val, default_start, default_end = 0.0, 1.0, 0.1, 0.9
    elif param_display_name == 'Echo Chamber Strength': min_val, max_val, default_start, default_end = 0.1, 1.0, 1.0, 0.2
    else: min_val, max_val, default_start, default_end = 0.0, 0.02, 0.001, 0.015

    sweep_range = st.sidebar.slider("Sweep Range", min_val, max_val, (default_start, default_end), step=max(0.001, (max_val-min_val)/100))
    num_steps = st.sidebar.slider("Number of Steps", 2, 20, 10)
    num_trials = st.sidebar.slider("Trials per Step", 1, 5, 2)
    
    st.sidebar.info(f"This will run a total of **{num_steps * num_trials}** simulations.")

    if st.sidebar.button("🔬 Run Experiment", type="primary"):
        st.header("Experiment Results")
        
        base_config = st.session_state.config.copy()
        base_config['N'] = 150 # Use a smaller N for faster experiments
        base_config['majority_opinion_vector'] = [0.5, 0.1]
        base_config['minority_opinion_vector'] = [-0.5, -0.1]
        base_config['campaign_params'] = base_config.get('campaign_params', {'target_opinion_vector': base_config['minority_opinion_vector']})

        sweep_values = np.linspace(sweep_range[0], sweep_range[1], num_steps)
        results_list = []
        progress_bar = st.progress(0); status_text = st.empty()
        total_runs = num_steps * num_trials; current_run = 0
        
        for i, value in enumerate(sweep_values):
            trial_outcomes = []
            for j in range(num_trials):
                current_run += 1
                status_text.text(f"Running trial {j+1}/{num_trials} for {param_display_name} = {value:.3f}...")
                run_config = base_config.copy()
                # A bit of logic to set the nested parameter value
                if len(param_path) == 2: run_config[param_path[0]][param_path[1]] = value
                elif len(param_path) == 3: run_config[param_path[0]][param_path[1]][param_path[2]] = value
                
                results = run_full_simulation(run_config, 30, random.randint(1,1e9))
                trial_outcomes.append(results['history_df']['pct_in_majority_camp'].iloc[-1])
                progress_bar.progress(current_run / total_runs)

            results_list.append({'param_value': value, 'mean_outcome': np.mean(trial_outcomes), 'std_outcome': np.std(trial_outcomes)})
        
        status_text.success("Experiment complete!"); results_df = pd.DataFrame(results_list)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['param_value'], y=results_df['mean_outcome'], error_y=dict(type='data', array=results_df['std_outcome'], visible=True), mode='lines+markers', name='Mean Outcome'))
        fig.update_layout(title=f"Impact of '{param_display_name}' on Final Majority Support", xaxis_title=param_display_name, yaxis_title="Final Support for Camp A (%)", yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(results_df)
    else:
        st.info("⬅️ Configure your experiment in the sidebar and click 'Run Experiment'.")