# Digital Poiesis Laboratory: User Manual

#### **1. Introduction: A Flight Simulator for Digital Policy**

Welcome to the Digital Poiesis Laboratory. This application is an interactive "flight simulator" designed to make the invisible "secret laws" of social media platforms visible and understandable. It uses an agent-based model to simulate an online social network, allowing you to see how different algorithms, platform designs, and strategic interventions can shape public opinion.

The goal is not to predict the future, but to build a deep, mechanistic intuition about how social realities are constructed and contested in the digital age.

The laboratory is structured as a clear, step-by-step workflow, reflected by the pages in the sidebar.

---

#### **2. The Workflow: A Step-by-Step Guide**

##### **Page 1: 🌎 The Digital Society**

This is where you define the fundamental "physics" of your digital world.

*   **Purpose:** To choose a social media platform archetype. Each archetype has a different combination of network structure, algorithmic priorities, and agent psychology, modeling the distinct dynamics of real-world platforms.
*   **How to Use:**
    1.  Use the **"Choose a Platform Archetype"** dropdown in the sidebar to select a world.
    2.  Review the core parameters displayed on the page to understand the environment you've selected.
        *   **Agent Psychology:** Describes the default psychological traits of the agents in this world (e.g., their learning rate, how many start with a fused identity).
        *   **Algorithmic Priorities:** Shows the default weights the platform's algorithm gives to different factors when ranking content.
        *   **Network Structure:** Describes how the social network is wired (e.g., is it an "echo chamber" where like-minded people connect?).

##### **Page 2: 🎯 The Strategic Operations Center**

This is your command center for designing and running an influence campaign.

*   **Day 0 Preview:** Before launching a campaign, this section shows you the initial state of the world you've configured.
    *   **Initial Belief Distribution:** A scatter plot showing the starting opinions of all the agents.
    *   **Social Network Structure:** A graph showing who is connected to whom. The colors indicate the agents' initial beliefs, allowing you to immediately see ideological clusters and potential echo chambers.
    *   **Metrics:**
        *   **Number of Agents:** The total population size (`N`).
        *   **Number of Social Links:** The total number of "follow" connections in the network.
        *   **Network Assortativity:** A score that measures how much agents tend to connect to other agents with similar beliefs. A high positive value indicates a strong echo chamber effect.

*   **⚙️ Algorithm Tuner & Campaign Strategy:** This is where you define your intervention.
    *   **Core Algorithm Weights:** These sliders allow you to override the platform's default algorithm.
        *   **`w_personalization`:** How much to prioritize content that matches an agent's existing beliefs.
        *   **`w_virality`:** How much to prioritize content that is getting a lot of engagement (likes).
        *   **`w_influence`:** How much to prioritize content from users who are already influential.
    *   **Strategic Campaign:**
        *   **Campaign Objective:** Choose your goal.
            *   **Offensive:** Attempt to flip the majority opinion to the minority view.
            *   **Defensive:** Attempt to "lock in" the majority opinion and prevent it from flipping.
        *   **Playbook:** Based on your objective, a playbook of strategic tools becomes available.
            *   **`Targeted Amplification / Algorithmic Suppression`:** This is your main lever of influence. It is the "secret law" you are imposing. A positive value boosts content aligned with your target; a negative value suppresses it.
            *   **`Bot Network Size`:** Creates a number of AI agents who do nothing but post content that supports your target opinion.
            *   **`Kingmaker Effect`:** Massively boosts the influence score of a few agents who are already close to your target opinion, turning them into powerful influencers.
            *   **`Radicalize Majority` (Defensive Only):** Increases the "identity fusion" of agents in the majority camp, making them highly resistant to changing their minds.

*   **General Controls (Sidebar):**
    *   **`Campaign Duration`:** How many cycles the simulation will run for.
    *   **`N (Participants)`:** The total number of agents in the simulation.

##### **Page 3: 📊 Post-Mortem Analysis**

This is where you analyze the results of your campaign to understand *why* it succeeded or failed.

*   **📈 Outcome Overview:**
    *   **Final Opinion Landscape:** Shows the belief of every agent at the end of the simulation.
    *   **Opinion & Key Metrics Evolution:**
        *   **Evolution of Public Opinion:** The primary outcome chart, showing the percentage of the population in the majority camp over time.
        *   **Evolution of Reality Distortion:** A crucial chart showing the average "Perception Gap" in the society. A high value means the reality agents see (their feeds) is very different from their own beliefs, indicating a high degree of algorithmic manipulation.

*   **🔬 Deep Analysis:**
    *   **The Perception Gap (Scatter Plot):** Compares agents' final beliefs to the average opinion of the content they were shown. Dots far from the red "y=x" line represent agents who were shown a highly distorted reality.
    *   **Agent Belief Trajectories:** Shows the paths a sample of individual agents took through the belief space.

*   **🕵️ Agent Drill-Down:** This is the most granular view, allowing you to inspect a single agent's experience.
    *   **How to Use:** Select an agent ID from the dropdown menu.
    *   **Stat Sheet:** Shows the agent's initial and final beliefs.
    *   **Belief Trajectory:** Shows the specific path this one agent took.
    *   **What This Agent Saw (Perceived Reality):** This heatmap is a window into the agent's soul. It shows the ideological landscape of *only the content they were exposed to*. The red 'X' marks their final belief, allowing you to see if their opinion is a product of the information they consumed.

*   **⚖️ A/B Testing:**
    *   **How to Use:** After a run, click "Pin Current Run as CONTROL (A)" or "TREATMENT (B)". You can then change your parameters, run a new campaign, and pin that as well. The chart and table will update to show a direct comparison of the outcomes, including the final Reality Distortion Index for each run.

##### **Page 4: 🧪 Experiment Designer**

This page allows you to run automated, systematic experiments.

*   **Purpose:** To test the impact of a single parameter across a range of values to find tipping points.
*   **How to Use:**
    1.  **Parameter to Sweep:** Choose the variable you want to investigate (e.g., "Targeted Amplification").
    2.  **Sweep Range:** Define the minimum and maximum values for the parameter.
    3.  **Number of Steps:** How many distinct values to test within that range.
    4.  **Trials per Step:** How many times to run the simulation for each value (to ensure statistical robustness).
    5.  Click "Run Experiment." The final plot will show you the relationship between the parameter you swept and the final outcome of the simulation.

---

#### **3. Advanced Use: Understanding the Configuration Files**

The platform archetypes are defined in the `.json` files located in the `configs/` directory. You can edit these or create new ones to model different kinds of digital worlds. The key sections are:

*   `world_generator`: Defines the network structure (e.g., `homophily_threshold`) and agent population.
*   `algorithm_params`: Defines the default behavior of the content algorithm, including the crucial `weights`.
*   `agent_psychology`: Defines the default psychological makeup of the agents, such as their `learning_rate` and `posting_propensity`.

---

#### **4. Glossary of Key Terms**

This glossary defines the core concepts of the Digital Poiesis Laboratory, explaining how they are modeled in the simulation and connecting them to established academic research where applicable.

**Agent-Based Model (ABM)**
*   **Definition:** A computational modeling technique that simulates the actions and interactions of autonomous agents (both individual and collective entities) to understand the behavior of a system as a whole.
*   **In this Simulation:** Each "agent" is a simulated person with their own beliefs and a set of rules for how they behave (e.g., creating content, updating their beliefs). The large-scale outcomes you see (like opinion shifts) are not programmed directly; they *emerge* from the simple, repeated interactions of many agents.
*   **Academic Context:** ABMs are a cornerstone of **Computational Social Science** and are used to study complex adaptive systems where macro-level patterns emerge from micro-level behavior. (e.g., work by Joshua M. Epstein).

**Belief Vector**
*   **Definition:** A pair of numbers (e.g., `[0.8, -0.2]`) that represents an agent's opinion on two perpendicular topics (Topic X and Topic Y). A positive value indicates agreement with "Camp A" on that topic, while a negative value indicates agreement with "Camp B." The distance from zero represents the strength of the belief.
*   **In this Simulation:** This is the core internal state of an agent. All actions, from the content they create to how they are influenced by their feed, are based on this vector.

**Homophily & Network Assortativity**
*   **Definition:** Homophily is the principle that individuals tend to associate with similar others.
*   **In this Simulation:** The `homophily_threshold` parameter in the world generator controls this. When the network is created, agents are more likely to form social links ("follow" each other) if the distance between their belief vectors is small. The **Network Assortativity** metric in the Day 0 Preview measures this, with a high positive value indicating strong homophily.
*   **Academic Context:** This models the well-documented principle that "birds of a feather flock together" (McPherson, Smith-Lovin, & Cook, 2001). It is a primary mechanism for the formation of social clusters and echo chambers.

**Echo Chamber / Filter Bubble**
*   **Definition:** An environment where a person is primarily exposed to information and opinions that conform to and reinforce their own.
*   **In this Simulation:** Echo chambers are an emergent property resulting from the combination of **Homophily** (a network structure of like-minded clusters) and an algorithm with high **Personalization** (which feeds users content they already agree with).
*   **Academic Context:** The term "Filter Bubble" was popularized by **Eli Pariser** to describe the state of intellectual isolation that can result from personalized feeds.

**Perception Gap & The Reality Distortion Index**
*   **Definition:** The "Perception Gap" is the difference between an agent's internal belief and the average belief represented by the content they are shown.
*   **In this Simulation:** We measure this for every agent and average it across the population to create the **Reality Distortion Index**.
    *   **Calculation:** It is the average Euclidean distance between each agent's `belief_vector` and the mean of all the `topic_vector`s in their `exposure_log`.
    *   **Interpretation:** A high index value indicates a significant degree of algorithmic manipulation. It means the world being shown to the agents is systematically different from their own opinions, representing a powerful force for pulling them toward a new belief.
*   **Academic Context:** This metric provides a way to quantify the effects of algorithmic curation on perceived social reality. It relates to the concepts of **pluralistic ignorance** (where people misjudge the prevailing norms or opinions in a group) and the **false consensus effect** (where people overestimate how much others share their beliefs).

**Identity Fusion**
*   **Definition:** A profound sense of "oneness" with a group or a belief, making that belief central to an agent's sense of self.
*   **In this Simulation:** The `Radicalize Majority` slider increases the percentage of agents in the majority camp who have a "fused" identity. These agents are highly resistant to changing their minds, regardless of what content they see.
*   **Academic Context:** This is based on the theory of **Identity Fusion** developed by psychologist **William B. Swann Jr.** and his colleagues. It explains why some beliefs (e.g., sacred values) are non-negotiable and why attempts to change them can backfire.

**Kingmaker Effect**
*   **Definition:** A strategy that focuses on massively amplifying the visibility and influence of a few carefully chosen individuals who are already aligned with a target narrative.
*   **In this Simulation:** The `Kingmaker Effect` slider multiplies the `creator_influence_score` of a small number of agents, making their content far more likely to be promoted by the algorithm's "influence" weight (`w_influence`).
*   **Academic Context:** This models a common and efficient influence strategy, distinct from a "broadcast" or "bot army" approach. It leverages the network's existing structure and the algorithm's tendency to favor influential nodes.
