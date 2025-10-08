# simulation_engine.py
# The robust, decoupled, and configuration-driven simulation engine.

import numpy as np
import random
from scipy.spatial.distance import cosine
import networkx as nx
from typing import List, Dict, Any
from enum import Enum

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
        return Content(self.agent_id, self.belief_vector, current_cycle)

    def update_belief(self, content):
        if self.psychology.get('is_identity_belief', False):
            return # Identity-fused agents don't change beliefs

        learning_rate = self.psychology.get('learning_rate', 0.01)
        conviction = self.psychology.get('conviction', 1.0)
        
        # Move belief towards content vector, slowed by conviction
        self.belief_vector += (learning_rate / conviction) * (content.topic_vector - self.belief_vector)
        self.belief_vector = np.clip(self.belief_vector, -1, 1)

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
        return getattr(self._graph, name)

class WorldGenerator:
    def create_world(self, N, config):
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
            for agent1 in population:
                for agent2 in population:
                    if agent1.agent_id != agent2.agent_id and cosine(agent1.belief_vector, agent2.belief_vector) < net_config['homophily_threshold']:
                        if random.random() < net_config['follow_propensity']: graph.add_follow_edge(agent1.agent_id, agent2.agent_id)
        return graph

class BaseAlgorithm:
    def __init__(self, params): self.params = params
    def rank_feed(self, viewer, candidates, population, social_graph): raise NotImplementedError

class HybridFeedAlgorithm(BaseAlgorithm):
    def rank_feed(self, viewer, candidates, population, social_graph):
        scores = []
        for content in candidates:
            creator = population[content.creator_id]
            w = self.params['weights']
            pers_score = 1 - cosine(viewer.platform_profile['inferred_belief_vector'], content.topic_vector)
            viral_score = len(content.engagement['likes']) - len(content.engagement['dislikes'])
            influ_score = creator.platform_profile['creator_influence_score']
            final_score = (w['w_personalization'] * pers_score) + (w['w_virality'] * viral_score) + (w['w_influence'] * influ_score)
            if 'amplification_bias_strength' in self.params and self.params['amplification_bias_strength'] != 0:
                final_score += self.params['amplification_bias_strength'] * (1 - cosine(content.topic_vector, self.params['target_opinion_vector']))
            scores.append(final_score)
        return sorted(candidates, key=lambda c: scores[candidates.index(c)], reverse=True)

ALGORITHM_REGISTRY = {"HybridFeedAlgorithm": HybridFeedAlgorithm}

class SimulationEngine:
    def __init__(self, population, social_graph, config):
        self.population = population; self.social_graph = social_graph; self.config = config
        self.N = len(population)
        algorithm_class = ALGORITHM_REGISTRY[config['engine_settings']['algorithm_class']]
        self.algorithm = algorithm_class({**config['algorithm_params'], **config.get('campaign_params', {})})
        self.all_content = {}; self.current_cycle = 0
        self._apply_initial_conditions()

    def _apply_initial_conditions(self):
        # Apply psychology and strategic elements to the generated population
        agent_indices = list(range(self.N)); random.shuffle(agent_indices)
        psych_params = self.config['agent_psychology']
        num_identity = int(self.N * psych_params['identity_fusion_pct'])
        identity_indices = set(random.sample(agent_indices, num_identity))
        for i in identity_indices: self.population[i].psychology['is_identity_belief'] = True

        campaign_params = self.config.get('campaign_params', {})
        num_bots = campaign_params.get('num_bots', 0)
        bot_indices = set(agent_indices[:num_bots])
        for i in bot_indices:
            self.population[i].is_bot = True
            self.population[i].belief_vector = np.array(campaign_params['target_opinion_vector'])

        if campaign_params.get('kingmaker_strength', 1.0) > 1.0:
            potential_kings = [p for p in self.population if not p.is_bot and cosine(p.belief_vector, campaign_params['target_opinion_vector']) < 0.2]
            if potential_kings:
                for king in random.sample(potential_kings, min(campaign_params['kingmaker_num'], len(potential_kings))):
                    king.platform_profile['creator_influence_score'] *= campaign_params['kingmaker_strength']

    def run_single_cycle(self):
        content_created_this_cycle = []
        num_posters = int(self.N * self.config['agent_psychology']['posting_propensity'])
        poster_indices = [p.agent_id for p in self.population if p.is_bot] + list(np.random.permutation([p.agent_id for p in self.population if not p.is_bot]))[:num_posters]
        for poster_id in poster_indices:
            new_content = self.population[poster_id].create_post(self.current_cycle)
            self.all_content[new_content.content_id] = new_content; content_created_this_cycle.append(new_content)
        for viewer in self.population:
            if viewer.is_bot: continue
            followed = self.social_graph.get_followed_by(viewer.agent_id)
            follower_content = [c for c in content_created_this_cycle if c.creator_id in followed]
            discovery_candidates = [c for c in content_created_this_cycle if c.creator_id not in followed and c.creator_id != viewer.agent_id]
            discovery_content = self.algorithm.rank_feed(viewer, discovery_candidates, self.population, self.social_graph)
            feed_size = viewer.attention_budget; disco_ratio = self.config['algorithm_params']['discovery_feed_ratio']
            num_discovery = int(feed_size * disco_ratio); num_follower = feed_size - num_discovery
            final_feed = follower_content[:num_follower] + discovery_content[:num_discovery]; random.shuffle(final_feed)
            for content in final_feed:
                viewer.exposure_log.append(content.topic_vector)
                viewer.update_belief(content)
        self._update_influence_scores(); self.current_cycle += 1

    def _update_influence_scores(self):
        for agent in self.population:
            if not agent.is_bot:
                likes_on_my_content = sum([len(c.engagement['likes']) for c in self.all_content.values() if c.creator_id == agent.agent_id])
                agent.platform_profile['creator_influence_score'] = (agent.platform_profile['creator_influence_score'] * 0.9) + (likes_on_my_content * 0.1)

    def get_stats(self):
        human_agents = [p for p in self.population if not p.is_bot];
        if not human_agents: return {}
        belief_matrix = np.array([agent.belief_vector for agent in human_agents])
        dist_to_maj = np.linalg.norm(belief_matrix - self.config['majority_opinion_vector'], axis=1)
        dist_to_min = np.linalg.norm(belief_matrix - self.config['minority_opinion_vector'], axis=1)
        return {"polarization_x": np.var(belief_matrix, axis=0)[0], "pct_in_majority_camp": np.mean(dist_to_maj < dist_to_min)}