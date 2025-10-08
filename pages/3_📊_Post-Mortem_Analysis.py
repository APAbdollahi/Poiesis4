#3_📊_Post-Mortem_Analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (plotly_belief_distribution, plotly_perception_gap_scatter, 
                     plotly_belief_trajectories, plotly_opinion_evolution, 
                     plotly_polarization_evolution)

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
            st.subheader("Opinion & Polarization Evolution")
            fig = plotly_opinion_evolution(history_df)
            st.plotly_chart(fig, use_container_width=True)
            fig2 = plotly_polarization_evolution(history_df)
            st.plotly_chart(fig2, use_container_width=True)

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
                st.subheader(f"Stat Sheet for Agent {agent_id}")
                initial_belief = trajectory_log[agent_id][0]; final_belief = agent.belief_vector
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Initial Belief (X)", f"{initial_belief[0]:.3f}")
                    st.metric("Final Belief (X)", f"{final_belief[0]:.3f}", delta=f"{final_belief[0] - initial_belief[0]:.3f}")
                with col6:
                    st.metric("Initial Belief (Y)", f"{initial_belief[1]:.3f}")
                    st.metric("Final Belief (Y)", f"{final_belief[1]:.3f}", delta=f"{final_belief[1] - initial_belief[1]:.3f}")
                st.write("---"); st.subheader("Agent's Personal Experience")
                col7, col8 = st.columns(2)
                with col7:
                    st.markdown("**Belief Trajectory**")
                    fig = plotly_belief_trajectories({agent.agent_id: trajectory_log[agent_id]})
                    st.plotly_chart(fig, use_container_width=True)
                with col8:
                    st.markdown("**Content Exposure (Perception)**")
                    if agent.exposure_log:
                        exposure_df = pd.DataFrame(agent.exposure_log, columns=['x', 'y'])
                        fig = px.density_heatmap(exposure_df, x='x', y='y', nbinsx=10, nbinsy=10, title="Heatmap of Content Seen")
                        fig.update_layout(xaxis_range=[-1,1], yaxis_range=[-1,1])
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("This agent was not exposed to any content.")
        else:
            st.warning("No agent trajectory data was logged for this run.")

    with tab4:
        st.header("A/B Testing: Comparative Analysis")
        st.markdown("Pin the results of a run to rigorously compare strategies.")
        col_a, col_b, col_c = st.columns(3)
        if col_a.button("Pin Current Run as CONTROL (A)"):
            st.session_state.pinned_results['CONTROL (A)'] = {"history_df": history_df, "mode": st.session_state.last_campaign_mode}
        if col_b.button("Pin Current Run as TREATMENT (B)"):
            st.session_state.pinned_results['TREATMENT (B)'] = {"history_df": history_df, "mode": st.session_state.last_campaign_mode}
        if col_c.button("Clear All Pinned Runs"):
            st.session_state.pinned_results = {}; st.experimental_rerun()
        if st.session_state.pinned_results:
            st.subheader("Comparison of Pinned Runs")
            fig = plotly_opinion_evolution(history_df, st.session_state.pinned_results)
            st.plotly_chart(fig, use_container_width=True)
            summary_data = []
            for name, data in st.session_state.pinned_results.items():
                df = data['history_df']
                summary_data.append({"Run Name": name, "Mode": data['mode'], "Initial Support": df['pct_in_majority_camp'].iloc[0], "Final Support": df['pct_in_majority_camp'].iloc[-1], "Change": df['pct_in_majority_camp'].iloc[-1] - df['pct_in_majority_camp'].iloc[0]})
            st.table(pd.DataFrame(summary_data).set_index("Run Name"))
        else:
            st.info("Pin one or more runs to see a comparison chart.")