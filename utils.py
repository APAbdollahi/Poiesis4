# utils.py
# Shared functions for config loading, state management, and plotting.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from simulation_engine import WorldGenerator, SimulationEngine
import json
import os

@st.cache_data
def load_platform_configs(config_dir="configs"):
    configs = {}
    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            with open(os.path.join(config_dir, filename), 'r') as f:
                config = json.load(f)
                configs[config['platform_name']] = config
    return configs

def initialize_session_state():
    if 'simulation_results' not in st.session_state: st.session_state.simulation_results = None
    if 'pinned_results' not in st.session_state: st.session_state.pinned_results = {}
    if 'config' not in st.session_state: st.session_state.config = None

@st.cache_data(show_spinner=False)
def run_full_simulation(config, num_cycles, _run_id):
    generator = WorldGenerator()
    population, social_graph = generator.create_world(config['N'], config)
    sim = SimulationEngine(population, social_graph, config)
    history = []
    agent_sample = [p for p in np.random.choice(sim.population, size=min(30, sim.N), replace=False) if not p.is_bot]
    trajectory_log = {agent.agent_id: [] for agent in agent_sample}
    for i in range(num_cycles):
        sim.run_single_cycle(); stats = sim.get_stats()
        if not stats: continue
        stats['cycle'] = sim.current_cycle; history.append(stats)
        for agent in agent_sample: trajectory_log[agent.agent_id].append(agent.belief_vector.copy())
    return {"history_df": pd.DataFrame(history), "final_sim_state": sim, "trajectory_log": trajectory_log}

# ... (All Plotly plotting functions from the previous final version remain here, unchanged) ...
def plotly_opinion_evolution(history_df, pinned_results={}):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df['cycle'], y=history_df['pct_in_majority_camp'] * 100, mode='lines', name='Current Run'))
    for name, data in pinned_results.items():
        fig.add_trace(go.Scatter(x=data['history_df']['cycle'], y=data['history_df']['pct_in_majority_camp'] * 100, mode='lines', name=name, line=dict(dash='dash')))
    fig.add_hline(y=50, line_dash="dot", line_color="red", annotation_text="50% Threshold")
    fig.update_layout(title="Evolution of Public Opinion", xaxis_title="Cycle", yaxis_title="Support for Camp A (%)", yaxis_range=[0,100])
    return fig
def plotly_polarization_evolution(history_df):
    fig = px.line(history_df, x='cycle', y='polarization_x', title="Evolution of Polarization")
    fig.update_layout(xaxis_title="Cycle", yaxis_title="Polarization (Variance of Beliefs)")
    return fig
def plotly_belief_distribution(sim_state, plot_type='scatter'):
    human_agents = [agent for agent in sim_state.population if not agent.is_bot]
    if not human_agents: return go.Figure()
    df = pd.DataFrame([{'agent_id': a.agent_id, 'belief_x': a.belief_vector[0], 'belief_y': a.belief_vector[1]} for a in human_agents])
    if plot_type == 'scatter':
        fig = px.scatter(df, x='belief_x', y='belief_y', color='belief_x', color_continuous_scale='RdBu', hover_data=['agent_id'], title="Final Belief Distribution (Scatter)")
    else:
        fig = px.density_heatmap(df, x='belief_x', y='belief_y', nbinsx=20, nbinsy=20, color_continuous_scale='viridis', title="Final Belief Distribution (Heatmap)")
    fig.update_layout(xaxis_title="Topic X", yaxis_title="Topic Y", xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])
    fig.add_vline(x=0, line_width=1, line_color="black"); fig.add_hline(y=0, line_width=1, line_color="black")
    return fig
def plotly_perception_gap_scatter(sim_state):
    human_agents = [p for p in sim_state.population if not p.is_bot]
    ground_truth_x = np.mean([p.belief_vector[0] for p in human_agents])
    data = []
    for agent in human_agents:
        if agent.exposure_log:
            perceived_x = np.mean([vec[0] for vec in agent.exposure_log])
            data.append({'agent_id': agent.agent_id, 'true_belief_x': agent.belief_vector[0], 'perceived_belief_x': perceived_x})
    if not data: return go.Figure()
    df = pd.DataFrame(data)
    fig = px.scatter(df, x='true_belief_x', y='perceived_belief_x', hover_data=['agent_id'], title="The Perception Gap: True vs. Perceived Beliefs")
    fig.add_shape(type='line', x0=-1, y0=-1, x1=1, y1=1, line=dict(color='red', dash='dash'), name='y=x (No Distortion)')
    fig.add_vline(x=ground_truth_x, line_dash="dot", line_color="green", annotation_text=f"Ground Truth ({ground_truth_x:.2f})")
    fig.update_layout(xaxis_title="Agent's True Belief (Topic X)", yaxis_title="Agent's Perceived Reality (Avg. Content)", xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])
    return fig
def plotly_belief_trajectories(trajectory_log):
    fig = go.Figure()
    for agent_id, path in trajectory_log.items():
        path = np.array(path)
        fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode='lines', line=dict(width=1, color='grey')))
        fig.add_trace(go.Scatter(x=[path[0, 0]], y=[path[0, 1]], mode='markers', marker=dict(color='green', size=5), name=f'Start {agent_id}'))
        fig.add_trace(go.Scatter(x=[path[-1, 0]], y=[path[-1, 1]], mode='markers', marker=dict(color='red', size=8), name=f'End {agent_id}'))
    fig.update_layout(title="Belief Trajectories of Sampled Agents", xaxis_title="Topic X", yaxis_title="Topic Y", xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1], showlegend=False)
    return fig