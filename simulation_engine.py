# simulation_engine.py
# The robust, decoupled, and configuration-driven simulation engine.

import numpy as np
import random
from scipy.spatial.distance import cosine
import networkx as nx
from typing import List, Dict, Any
from enum import Enum
from scipy.spatial import KDTree
from abc import ABC, abstractmethod

class ContentType(Enum):
    TEXT = 1
    IMAGE = 2
    SHORT_VIDEO = 3

class Content:
    _id_counter = 0
    def __init__(self, creator_id, topic_vector, creation_time, content_type=ContentType.TEXT, parent_id=None, community_id=None):
        self.content_id = Content._id_counter; Content._id_counter += 1
        self.creator_id = creator_id
        self.topic_vector = topic_vector
        self.creation_time = creation_time
        self.content_type = content_type
        self.parent_id = parent_id
        self.community_id = community_id
        self.engagement = {'likes': set(), 'dislikes': set(), 'shares': set()}

class Agent:
    def __init__(self, agent_id, initial_belief_vector, psychology, propensities):
        self.agent_id = agent_id
        self.belief_vector = initial_belief_vector
        self.psychology = psychology # Holds traits like is_identity_belief, conviction, learning_rate
        self.propensities = propensities # Holds behavioral tendencies like content_creation_rate, reply_propensity
        self.platform_profile = { # Platform-specific data
            'inferred_belief_vector': initial_belief_vector.copy(),
            'creator_influence_score': 1.0
        }
        self.is_bot = False # This is a special state, kept separate for now
        self.liked_content_history = []
        self.attention_budget = max(1, int(np.random.normal(10, 3)))
        self.exposure_log = []

    def create_post(self, current_cycle):
        return Content(self.agent_id, self.belief_vector.copy(), current_cycle)

    def update_belief(self, content):
        """Update agent's belief based on consumed content using linear assimilation."""
        if self.psychology.get('is_identity_belief', False):
            return # Identity-fused agents don't change beliefs

        learning_rate = self.psychology.get('learning_rate', 0.01)
        conviction = self.psychology.get('conviction', 1.0)
        
        # Move belief towards content vector, slowed by conviction
        self.belief_vector += (learning_rate / conviction) * (content.topic_vector - self.belief_vector)
        self.belief_vector = np.clip(self.belief_vector, -1, 1)
    
    def engage_with_content(self, content, engagement_probability=0.3):
        """
        Agent engages with content (likes/dislikes) based on alignment with their beliefs.
        Higher alignment = more likely to like.
        """
        if self.is_bot:
            return  # Bots don't engage organically
        
        # Calculate alignment: closer beliefs = higher alignment
        distance = np.linalg.norm(self.belief_vector - content.topic_vector)
        alignment = max(0, 1 - distance / 2.0)  # Normalize to [0, 1]
        
        # Decide to engage based on probability
        if random.random() < engagement_probability:
            # Like if aligned, dislike if opposed
            if alignment > 0.5:
                content.engagement['likes'].add(self.agent_id)
                self.liked_content_history.append(content.content_id)
            elif alignment < 0.3:
                content.engagement['dislikes'].add(self.agent_id)

class SocialGraph:
    def __init__(self):
        self._graph = nx.DiGraph()
    def add_agent(self, agent_id):
        self._graph.add_node(agent_id)
    def add_follow_edge(self, follower_id, followed_id):
        self._graph.add_edge(follower_id, followed_id)
    def get_followed_by(self, agent_id):
        return list(self._graph.successors(agent_id))
    def calculate_assortativity(self):
        return nx.degree_assortativity_coefficient(self._graph)
    def __getattr__(self, name):
        # Do not forward dunder methods to the graph object.
        # This prevents recursion errors with pickle.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._graph, name)

class WorldGenerator:
    def create_world(self, N, config, master_seed):
        random.seed(master_seed)
        np.random.seed(master_seed)

        # Create Population
        pop_config = config['world_generator']['population_config']
        belief_config = {'majority_opinion_vector': config['majority_opinion_vector'], 'minority_opinion_vector': config['minority_opinion_vector'], 'original_opinion_pct': config['agent_psychology']['original_opinion_pct']}
        population = self._create_population(N, pop_config, belief_config, config['agent_psychology'])
        # Create Network
        net_config = config['world_generator']['network_config']
        social_graph = self._create_network(population, net_config)
        return population, social_graph

    def _create_population(self, N, pop_config, belief_config, agent_psychology_config):
        agents = []
        num_cluster1 = int(N * belief_config['original_opinion_pct'])
        cluster1_center = np.array(belief_config['majority_opinion_vector'])
        cluster2_center = np.array(belief_config['minority_opinion_vector'])
        covariance = [[0.05, 0], [0, 0.05]]
        for i in range(N):
            belief_vector = np.random.multivariate_normal(cluster1_center if i < num_cluster1 else cluster2_center, covariance)
            psychology = {
                'is_identity_belief': False,
                'conviction': np.clip(np.random.normal(1.0, 0.5), 0.1, None),
                'learning_rate': agent_psychology_config['learning_rate']
            }
            agent = Agent(
                agent_id=i,
                initial_belief_vector=np.clip(belief_vector, -1, 1),
                psychology=psychology,
                propensities=pop_config['agent_propensities']
            )
            agents.append(agent)
        random.shuffle(agents)
        return agents

    def _create_network(self, population, net_config):
        graph = SocialGraph()
        for agent in population: graph.add_agent(agent.agent_id)

        if net_config['network_type'] == 'homophily_driven':
            belief_vectors = np.array([agent.belief_vector for agent in population])
            tree = KDTree(belief_vectors)

            for agent1 in population:
                # Find all agents within the homophily_threshold distance
                # r_query returns indices of points within r of a given point
                indices = tree.query_ball_point(agent1.belief_vector, r=net_config['homophily_threshold'])
                
                for idx in indices:
                    agent2 = population[idx]
                    if agent1.agent_id != agent2.agent_id:
                        if random.random() < net_config['follow_propensity']:
                            graph.add_follow_edge(agent1.agent_id, agent2.agent_id)
        return graph

def safe_cosine_similarity(v1, v2):
    """
    Compute cosine similarity (not distance) with zero-vector handling.
    Returns a value in [-1, 1] where 1 is perfect alignment.
    """
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return similarity

class BaseAlgorithm:
    def __init__(self, params): self.params = params
    def rank_feed(self, viewer, candidates, population, social_graph): raise NotImplementedError

class HybridFeedAlgorithm(BaseAlgorithm):
    def rank_feed(self, viewer, candidates, population, social_graph):
        """
        Rank content using the hybrid scoring function from the paper.
        Returns sorted list of content items.
        """
        if not candidates:
            return []
        
        candidate_scores = []
        w = self.params['weights']
        
        for content in candidates:
            creator = population[content.creator_id]
            
            # Personalization score: using cosine similarity (1 = perfect match)
            pers_score = (1 + safe_cosine_similarity(viewer.belief_vector, content.topic_vector)) / 2.0
            
            # Virality score: net engagement
            viral_score = len(content.engagement['likes']) - len(content.engagement['dislikes'])
            
            # Influence score: creator's accumulated influence
            influ_score = creator.platform_profile['creator_influence_score']
            
            # Base score from weighted sum
            final_score = (w['w_personalization'] * pers_score) + \
                         (w['w_virality'] * viral_score) + \
                         (w['w_influence'] * influ_score)
            
            # Targeted amplification (β_amp from paper)
            if 'amplification_bias_strength' in self.params and self.params['amplification_bias_strength'] != 0:
                target_alignment = (1 + safe_cosine_similarity(content.topic_vector, 
                                                               self.params['target_opinion_vector'])) / 2.0
                final_score += self.params['amplification_bias_strength'] * target_alignment
            
            candidate_scores.append((content, final_score))
        
        # Sort by score (descending) and return just the content objects
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [content for content, score in candidate_scores]

ALGORITHM_REGISTRY = {"HybridFeedAlgorithm": HybridFeedAlgorithm}

class SimulationEngine:
    def __init__(self, population, social_graph, config, master_seed):
        random.seed(master_seed)
        np.random.seed(master_seed)

        self.population = population
        self.social_graph = social_graph
        self.config = config
        self.N = len(population)
        algorithm_class = ALGORITHM_REGISTRY[config['engine_settings']['algorithm_class']]
        self.algorithm = algorithm_class({**config['algorithm_params'], **config.get('campaign_params', {})})
        self.all_content = {}
        self.current_cycle = 0
        self._apply_initial_conditions()

    def _apply_initial_conditions(self):
        """Apply psychology and strategic elements to the generated population."""
        agent_indices = list(range(self.N))
        random.shuffle(agent_indices)
        psych_params = self.config['agent_psychology']
        
        # Apply identity fusion to a percentage of agents
        num_identity = int(self.N * psych_params['identity_fusion_pct'])
        identity_indices = set(random.sample(agent_indices, num_identity))
        for i in identity_indices:
            self.population[i].psychology['is_identity_belief'] = True

        campaign_params = self.config.get('campaign_params', {})
        
        # Create bot network
        num_bots = campaign_params.get('num_bots', 0)
        bot_indices = set(agent_indices[:num_bots])
        for i in bot_indices:
            self.population[i].is_bot = True
            self.population[i].belief_vector = np.array(campaign_params['target_opinion_vector'])

        # FIX: Implement defensive campaign "radicalize majority" feature
        if 'majority_identity_fusion' in campaign_params:
            majority_vector = np.array(self.config['majority_opinion_vector'])
            # Find agents in the majority camp
            majority_agents = [p for p in self.population 
                             if not p.is_bot and 
                             np.linalg.norm(p.belief_vector - majority_vector) < 0.5]
            
            # Radicalize a percentage of them
            num_to_radicalize = int(len(majority_agents) * campaign_params['majority_identity_fusion'])
            if num_to_radicalize > 0 and majority_agents:
                for agent in random.sample(majority_agents, min(num_to_radicalize, len(majority_agents))):
                    agent.psychology['is_identity_belief'] = True

        # Kingmaker intervention: boost influence of aligned creators
        if campaign_params.get('kingmaker_strength', 1.0) > 1.0:
            target_vector = np.array(campaign_params['target_opinion_vector'])
            potential_kings = [p for p in self.population 
                             if not p.is_bot and 
                             safe_cosine_similarity(p.belief_vector, target_vector) > 0.8]
            
            if potential_kings:
                num_kings = min(campaign_params.get('kingmaker_num', 2), len(potential_kings))
                for king in random.sample(potential_kings, num_kings):
                    king.platform_profile['creator_influence_score'] *= campaign_params['kingmaker_strength']

    def run_single_cycle(self):
        """Execute one cycle of the simulation."""
        content_created_this_cycle = []
        
        # Content creation phase
        num_posters = int(self.N * self.config['agent_psychology']['posting_propensity'])
        # Bots always post, plus random selection of human agents
        bot_posters = [p.agent_id for p in self.population if p.is_bot]
        human_posters = [p.agent_id for p in self.population if not p.is_bot]
        selected_posters = bot_posters + list(np.random.choice(human_posters, 
                                                               size=min(num_posters, len(human_posters)), 
                                                               replace=False))
        
        for poster_id in selected_posters:
            new_content = self.population[poster_id].create_post(self.current_cycle)
            self.all_content[new_content.content_id] = new_content
            content_created_this_cycle.append(new_content)
        
        # Content consumption and engagement phase
        for viewer in self.population:
            if viewer.is_bot:
                continue  # Bots don't consume content
            
            # Get content from followed accounts (organic feed)
            followed = self.social_graph.get_followed_by(viewer.agent_id)
            follower_content = [c for c in content_created_this_cycle if c.creator_id in followed]
            
            # Get discovery content (algorithmic feed)
            discovery_candidates = [c for c in content_created_this_cycle 
                                  if c.creator_id not in followed and c.creator_id != viewer.agent_id]
            discovery_content = self.algorithm.rank_feed(viewer, discovery_candidates, 
                                                        self.population, self.social_graph)
            
            # Compose final feed
            feed_size = viewer.attention_budget
            disco_ratio = self.config['algorithm_params']['discovery_feed_ratio']
            num_discovery = int(feed_size * disco_ratio)
            num_follower = feed_size - num_discovery
            
            # FIX: Backfill with discovery if not enough follower content
            final_feed = follower_content[:num_follower]
            remaining_slots = feed_size - len(final_feed)
            final_feed += discovery_content[:max(num_discovery, remaining_slots)]
            
            random.shuffle(final_feed)
            
            # Consume and engage with feed
            for content in final_feed:
                viewer.exposure_log.append(content.topic_vector.copy())
                viewer.update_belief(content)
                
                # FIX: Add engagement mechanics
                viewer.engage_with_content(content, engagement_probability=0.3)
        
        self._update_influence_scores()
        self.current_cycle += 1

    def _update_influence_scores(self):
        """
        Update creator influence scores based on recent engagement.
        FIX: Only count likes from the current cycle, not all-time.
        """
        for agent in self.population:
            if not agent.is_bot:
                # Count likes on content created THIS cycle
                likes_this_cycle = sum([
                    len(c.engagement['likes']) 
                    for c in self.all_content.values() 
                    if c.creator_id == agent.agent_id and c.creation_time == self.current_cycle
                ])
                
                # Exponential moving average with decay
                decay = 0.95
                boost = 0.5
                agent.platform_profile['creator_influence_score'] = \
                    (agent.platform_profile['creator_influence_score'] * decay) + (likes_this_cycle * boost)

    def get_stats(self):
        """Calculate and return aggregate statistics for the current state."""
        human_agents = [p for p in self.population if not p.is_bot]
        if not human_agents:
            return {}
        
        belief_matrix = np.array([agent.belief_vector for agent in human_agents])
        dist_to_maj = np.linalg.norm(belief_matrix - self.config['majority_opinion_vector'], axis=1)
        dist_to_min = np.linalg.norm(belief_matrix - self.config['minority_opinion_vector'], axis=1)
        
        # Calculate Perception Gap (Reality Distortion Index)
        # FIX: Use windowed exposure history to avoid unbounded growth
        perception_gaps = []
        for agent in human_agents:
            if agent.exposure_log:
                # Use recent exposure (last 100 items) to calculate perceived reality
                recent_exposure = agent.exposure_log[-100:]
                perceived_reality = np.mean(recent_exposure, axis=0)
                gap = np.linalg.norm(agent.belief_vector - perceived_reality)
                perception_gaps.append(gap)
        
        avg_perception_gap = np.mean(perception_gaps) if perception_gaps else 0
        
        return {
            "polarization_x": np.var(belief_matrix, axis=0)[0],
            "pct_in_majority_camp": np.mean(dist_to_maj < dist_to_min),
            "reality_distortion_index": avg_perception_gap
        }
