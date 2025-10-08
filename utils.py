

def plotly_agent_exposure_heatmap(agent):
    """
    Visualizes the content an individual agent was exposed to as a heatmap.
    """
    if not agent.exposure_log:
        return go.Figure(layout_title_text=f"Agent {agent.agent_id} was not exposed to any content.")

    exposure_df = pd.DataFrame(agent.exposure_log, columns=['x', 'y'])
    
    fig = px.density_heatmap(
        exposure_df, 
        x='x', 
        y='y', 
        nbinsx=20, 
        nbinsy=20, 
        color_continuous_scale='Viridis',
        title=f"Agent {agent.agent_id}'s Perceived Reality"
    )
    
    # Add a marker for the agent's actual belief
    fig.add_trace(go.Scatter(
        x=[agent.belief_vector[0]],
        y=[agent.belief_vector[1]],
        mode='markers',
        marker=dict(
            color='red',
            size=12,
            symbol='x',
            line=dict(width=2, color='white')
        ),
        name='Agent\'s Final Belief'
    ))
    
    fig.update_layout(
        xaxis_title="Topic X", 
        yaxis_title="Topic Y", 
        xaxis_range=[-1.1, 1.1], 
        yaxis_range=[-1.1, 1.1],
        showlegend=False
    )
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.add_hline(y=0, line_width=1, line_color="black")
    
    return fig

