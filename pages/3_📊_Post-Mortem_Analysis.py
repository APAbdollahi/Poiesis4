#3_📊_Post-Mortem_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # FIX: Added missing import
from utils import (plotly_belief_distribution, plotly_perception_gap_scatter, 
                     plotly_belief_trajectories, plotly_opinion_evolution, 
                     plotly_polarization_evolution, plotly_agent_exposure_heatmap, 
                     plotly_reality_distortion_evolution)

st.set_page_config(layout="wide", page_title="Post-Mortem Analysis")
st.title("📊 Post-Mortem Analysis")
st.markdown("Conduct a deep analysis of the campaign's outcome and consequences. Understand *why* it succeeded or failed.")
st.markdown("---")

if 'simulation_results' not in st.session_state or not st.session_state.simulation_results:
    st.error("No simulation has been run yet. Please run a campaign on the `2_🎯_Strategic_Operations` page first.")
else:
    results = st.session_state.simulation_results
    history_df = results["history_df"]
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Outcome Overview", "🔬 Deep Analysis", "🕵️ Agent Drill-Down", "⚖️ A/B Testing"])

    with tab1:
        st.header("Campaign Outcome Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Final Opinion Landscape")
            plot_type = st.radio("Chart Type", ('Scatter', 'Heatmap'), horizontal=True)
            fig = plotly_belief_distribution(results["final_sim_state"], plot_type=plot_type.lower())
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Opinion & Key Metrics Evolution")
            fig_opinion = plotly_opinion_evolution(history_df)
            st.plotly_chart(fig_opinion, use_container_width=True)
            
            fig_distortion = plotly_reality_distortion_evolution(history_df)
            st.plotly_chart(fig_distortion, use_container_width=True)


    with tab2:
        st.header("Deep Analysis: Uncovering the 'Secret Laws'")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("The Perception Gap (Scatter Plot)")
            st.markdown("Each dot is an agent. This shows the relationship between an agent's **True Belief** (x-axis) and the **Average Content They Saw** (y-axis). Dots far from the red line were shown a distorted reality.")
            fig = plotly_perception_gap_scatter(results["final_sim_state"])
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Agent Belief Trajectories")
            st.markdown("This plot shows the paths a sample of agents took through the belief space. Green dots are start points, red dots are end points.")
            fig = plotly_belief_trajectories(results["trajectory_log"])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Agent Drill-Down")
        trajectory_log = results["trajectory_log"]
        inspectable_agent_ids = list(trajectory_log.keys())
        if inspectable_agent_ids:
            agent_id = st.selectbox("Select a Sampled Agent ID to inspect:", inspectable_agent_ids)
            if agent_id is not None:
                agent = next((p for p in results["final_sim_state"].population if p.agent_id == agent_id), None)
                
                if agent:
                    st.subheader(f"Stat Sheet for Agent {agent_id}")
                    initial_belief = trajectory_log[agent_id][0]
                    final_belief = agent.belief_vector
                    
                    scol1, scol2 = st.columns(2)
                    with scol1:
                        st.metric("Initial Belief (X)", f"{initial_belief[0]:.3f}")
                        st.metric("Final Belief (X)", f"{final_belief[0]:.3f}", delta=f"{final_belief[0] - initial_belief[0]:.3f}")
                    with scol2:
                        st.metric("Initial Belief (Y)", f"{initial_belief[1]:.3f}")
                        st.metric("Final Belief (Y)", f"{final_belief[1]:.3f}", delta=f"{final_belief[1] - initial_belief[1]:.3f}")
                    
                    st.markdown("---")
                    st.subheader("Agent's Personal Experience")
                    
                    exp_col1, exp_col2 = st.columns(2)
                    with exp_col1:
                        st.markdown("**Belief Trajectory**")
                        st.markdown("The path this agent took through the belief space. (Green=Start, Red=End)")
                        fig_traj = plotly_belief_trajectories({agent.agent_id: trajectory_log[agent_id]})
                        st.plotly_chart(fig_traj, use_container_width=True)
                    with exp_col2:
                        st.markdown("**What This Agent Saw (Perceived Reality)**")
                        st.markdown("A heatmap of the content this agent was exposed to. The red 'X' marks their final belief.")
                        fig_exp = plotly_agent_exposure_heatmap(agent)
                        st.plotly_chart(fig_exp, use_container_width=True)
                else:
                    st.warning(f"Could not find agent {agent_id} in final simulation state.")
        else:
            st.warning("No agent trajectory data was logged for this run.")

    with tab4:
        st.header("A/B Testing: Comparative Analysis")
        st.markdown("Pin the results of a run to rigorously compare strategies.")
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("Pin Current Run as CONTROL (A)"):
            st.session_state.pinned_results['CONTROL (A)'] = {"history_df": history_df, "mode": st.session_state.last_campaign_mode}
            st.success("Pinned as CONTROL (A)")
        if col_b.button("Pin Current Run as TREATMENT (B)"):
            st.session_state.pinned_results['TREATMENT (B)'] = {"history_df": history_df, "mode": st.session_state.last_campaign_mode}
            st.success("Pinned as TREATMENT (B)")
        if col_c.button("Clear All Pinned Runs"):
            st.session_state.pinned_results = {}
            st.rerun()  # FIX: Changed from deprecated st.experimental_rerun()
        
        if st.session_state.pinned_results:
            st.subheader("Comparison of Pinned Runs")
            fig_comp = go.Figure()
            summary_data = []
            for name, data in st.session_state.pinned_results.items():
                df = data['history_df']
                fig_comp.add_trace(go.Scatter(x=df['cycle'], y=df['pct_in_majority_camp'] * 100, mode='lines', name=name))
                summary_data.append({
                    "Run Name": name, 
                    "Mode": data['mode'], 
                    "Initial Support": f"{df['pct_in_majority_camp'].iloc[0]:.1%}", 
                    "Final Support": f"{df['pct_in_majority_camp'].iloc[-1]:.1%}", 
                    "Change": f"{(df['pct_in_majority_camp'].iloc[-1] - df['pct_in_majority_camp'].iloc[0]):.1%}",
                    "Final Distortion": f"{df['reality_distortion_index'].iloc[-1]:.3f}" if 'reality_distortion_index' in df.columns else 'N/A'
                })
            
            fig_comp.update_layout(title="Comparison of Public Opinion Evolution", xaxis_title="Cycle", yaxis_title="Support for Camp A (%)", yaxis_range=[0,100])
            st.plotly_chart(fig_comp, use_container_width=True)
            st.table(pd.DataFrame(summary_data).set_index("Run Name"))
        else:
            st.info("Pin one or more runs to see a comparison chart.")
