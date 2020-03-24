import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import norm

from mesa import Agent

from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.modules import ChartModule

from mesa.batchrunner import BatchRunner

from itertools import count
import networkx as nx
from matplotlib import pylab
from matplotlib.pyplot import pause
import matplotlib.animation as animation

class Opinion():
    EXTREME_QUALITY_LEVEL = 7
    EXTREME_CONFIDENCE_LEVEL = 5

    def __init__(self, extremity, quality, confidence):
        self.extremity = extremity
        self.quality = quality
        self.confidence = confidence

    def low_quality(self):
        return self.quality < -Opinion.EXTREME_QUALITY_LEVEL
    
    def high_quality(self):
        return self.quality > Opinion.EXTREME_QUALITY_LEVEL
    
    def low_confidence(self):
        return self.confidence < -Opinion.EXTREME_CONFIDENCE_LEVEL
    
    def high_confidence(self):
        return self.confidence > Opinion.EXTREME_CONFIDENCE_LEVEL

class Media(Agent):
    INFLUENCE_RANGE = 4
    SUBSCRIBER_RANGE = 2
    SEARCH_RANGE = 6

    def num_citizens_in_range(citizens, position, scan_range):
        citizen_extremities = list(map(lambda x: x.opinion.extremity, citizens))
        return len(list(filter(lambda x: abs(x - position) < scan_range, citizen_extremities))) 


    def __init__(self, unique_id, model, position, change_position=True):
        super().__init__(unique_id, model)
        self.position = position
        self.subscriber_count = Media.num_citizens_in_range(model.citizens(), position, Media.SUBSCRIBER_RANGE)
        
        self.planned_position = position
        self.planned_subscribers = self.subscriber_count

        self.change_position = change_position

    def serialize(self):
        connected_nodes = self.influencable_citizens()
        return {
            "type": "media",
            "id": self.unique_id,
            "extremity": self.position,
            "connected_nodes": list(map(lambda x: x.unique_id, connected_nodes)),
            "connection_strengths": list(map(lambda x: 1, connected_nodes))
        }

    POSITION_PARAMTER_LIMIT = 10
    def param_min_max(self, value, absolute_max=POSITION_PARAMTER_LIMIT):
        return max(-absolute_max, min(absolute_max, value))

    def influencable_citizens(self):        
        citizens = self.model.citizens()
        return list(filter(lambda citizen: abs(citizen.opinion.extremity - self.position) < Media.INFLUENCE_RANGE, citizens))

    # Determines how much media cares about competition. 0 is maximum care, 1 is minimum.
    COMPETITION_SCALE_FACTOR = 0
    def competition_score(self, position):
        media = self.model.media()
        avg_competition_distance = sum(list(map(lambda x: abs(x.position - position), media)))/float(len(media))
        competition_score = 1-(avg_competition_distance / 20.0)

        
        scaled_score = COMPETITION_SCALE_FACTOR + (competition_score/1.0*(1-COMPETITION_SCALE_FACTOR))
        
        return scaled_score

    def exert_influence_on_citizens(self):
        for citizen in self.influencable_citizens():
            update_distance = (self.position - citizen.opinion.extremity)/20.0
            citizen.planned_media_influence = update_distance

    def plan_position(self):
        search_distance = Media.SEARCH_RANGE/2.0
        
        max_subscribers = self.subscriber_count
        new_position = self.position
        new_competition_score = self.competition_score(self.position)
        
        for potential_position in np.linspace(self.position - search_distance, self.position + search_distance, 20):
            potential_competition_score = self.competition_score(potential_position)
            potential_subscribers = Media.num_citizens_in_range(self.model.citizens(), potential_position, Media.SUBSCRIBER_RANGE)
            
            if potential_subscribers * (1-potential_competition_score) > max_subscribers * (1-new_competition_score):
                max_subscribers = potential_subscribers
                new_position = potential_position
                new_competition_score = potential_competition_score

        self.planned_position = new_position
        self.planned_subscribers = max_subscribers

    def step(self):
        self.exert_influence_on_citizens()

        if self.change_position:
            self.plan_position()

    def advance(self):
        self.position = self.param_min_max(self.planned_position)
        self.subscriber_count = self.planned_subscribers

class Citizen(Agent):
    IN_GROUP_RANGE = 2
    OPINION_PARAMTER_LIMIT = 10

    def __init__(self, unique_id, model, opinion, connected_citzens=[]):
        super().__init__(unique_id, model)

        self.opinion = opinion
        self.connected_citzens = list(connected_citzens)
        self.connection_strengths = [1 for _ in connected_citzens]
        
        self.planned_extremity = None
        self.planned_quality = None
        self.planned_confidence = None
        self.planned_connections, self.planned_strengths = None, None
        self.planned_media_influence = None
    
    def serialize(self):
        return {
            "type": "citizen",
            "id": self.unique_id,
            "extremity": self.opinion.extremity,
            "connected_nodes": list(map(lambda x: x.unique_id, self.connected_nodes)),
            "connection_strengths": list(self.connection_strengths)
        }

    def param_min_max(self, value, absolute_max=OPINION_PARAMTER_LIMIT):
        return max(-absolute_max, min(absolute_max, value))
    
    def in_group_citizens(self, pool=None):
        if not pool:
            pool = self.connected_nodes
        
        return list(filter(lambda citizen: abs(citizen.opinion.extremity - self.opinion.extremity) < Citizen.IN_GROUP_RANGE, pool)) 
    
    def out_group_citizens(self):
        group = []
        in_group_ids = list(map(lambda x: x.unique_id, self.in_group_citizens()))
        
        for citizen in self.connected_nodes:
            if citizen.unique_id not in in_group_ids:
                group.append(citizen)
        return group
    
    def update_opinion_extremity(self):
        in_group_opinions = list(map(lambda x: x.opinion.extremity, self.in_group_citizens()))
        out_group_opinions = list(map(lambda x: x.opinion.extremity, self.out_group_citizens()))
           
        all_update_distance = 0
        if len(in_group_opinions) + len(out_group_opinions) > 0:
            all_group_average_extremity = sum(in_group_opinions) + sum(out_group_opinions) / float(len(in_group_opinions) + len(out_group_opinions))                    
            all_update_distance = all_group_average_extremity - self.opinion.extremity / 4.0
        
        out_update_distance = 0
        if len(out_group_opinions) > 0:
            out_group_average_extremity = sum(out_group_opinions) / float(len(out_group_opinions))
            out_update_distance = out_group_average_extremity - self.opinion.extremity / 2.0
        
        new_extremity = 0
        
        if self.opinion.low_quality():
            if self.opinion.low_confidence():
                # Move toward all group average
                new_extremity = self.opinion.extremity + all_update_distance
            else:
                # Move away from out group position
                new_extremity = self.opinion.extremity - out_update_distance
        else:
            if self.opinion.low_confidence():
                # Move toward group average more slowly.
                new_extremity = self.opinion.extremity + all_update_distance / 2.0
            else:
                # No change
                new_extremity = self.opinion.extremity
        
        return new_extremity
    
    def update_opinion_quality(self):
        in_group_opinion_quality = list(map(lambda x: x.opinion.quality, self.in_group_citizens()))
        out_group_opinion_quality = list(map(lambda x: x.opinion.quality, self.out_group_citizens()))
        
        in_update_distance = 0
        if len(in_group_opinion_quality) > 0:
            in_group_average_quality = sum(in_group_opinion_quality)/ float(len(in_group_opinion_quality))                    
            in_update_distance = in_group_average_quality - self.opinion.quality / 4.0

        return self.opinion.quality + in_update_distance
        
    def update_opinion_confidence(self):
        in_group_opinions = len(self.in_group_citizens())
        out_group_opinions = len(self.out_group_citizens())
        
        modifier = 1
        if in_group_opinions + out_group_opinions > 0:
            percent_in_group = in_group_opinions/float(in_group_opinions + out_group_opinions)
            modifier = 1 + (percent_in_group - 0.5)
        
        return self.opinion.confidence * modifier
        
    def update_connections(self):
        connected_citizens = list(self.connected_nodes)
        connection_strengths = list(self.connection_strengths)

        if len(self.connected_nodes) == 0:
            connection_source = np.random.choice(self.in_group_citizens(pool=self.model.citizens()))

        else:
            connection_source = np.random.choice(self.connected_nodes)
            count = 0
            while len(connection_source.connected_nodes) == 0:
                connection_source = np.random.choice(self.connected_nodes)
                count += 1
                if count == len(self.connected_nodes):
                    connection_source = np.random.choice(self.in_group_citizens(pool=self.model.citizens()))
                    break

        if len(connection_source.connected_nodes) > 0:
            new_connection = np.random.choice(connection_source.connected_nodes)
        
            if new_connection not in self.connected_nodes:
                connected_citizens.append(new_connection)
                connection_strengths.append(1)
        
        to_remove = []
        
        for other_node in connected_citizens:
            citizen_index = connected_citizens.index(other_node)
            percent_distance = abs(other_node.opinion.extremity - self.opinion.extremity)/20.0
            connection_strengths[citizen_index] -= percent_distance/2.0
            if connection_strengths[citizen_index] < 0.2:
                to_remove.append(other_node)
            
        for node in to_remove:
            ids = list(map(lambda x: x.unique_id, connected_citizens))
            citizen_index = ids.index(node.unique_id)
            del connected_citizens[citizen_index]
            del connection_strengths[citizen_index]
    
        return (connected_citizens, connection_strengths)
                
    # MESA supports simultaneous execution through step and advance. Step is called for all agents before advance.
    # Agents plan in the step phase and then all agents enact their plans in the advance phase.
    
    # REQUIRED METHOD: step is the name used by MESA for the plan stage.
    def step(self):
        self.planned_extremity = self.update_opinion_extremity()
        self.planned_quality = self.update_opinion_quality()
        self.planned_confidence = self.update_opinion_confidence()
        self.planned_connections, self.planned_strengths = self.update_connections()
        
    # REQUIRED METHOD: advance refers to implements planned changes. 
    def advance(self):
        self.planned_media_influence = self.planned_media_influence if self.planned_media_influence else 0
        new_extremity = self.planned_extremity + self.planned_media_influence
        
        self.opinion.extremity = self.param_min_max(new_extremity)
        self.opinion.quality = self.param_min_max(self.planned_quality)
        self.opinion.confidence = self.param_min_max(self.planned_confidence)
        self.connected_nodes = self.planned_connections
        self.connection_strengths =  self.planned_strengths

class SocietyModel(Model):
    
    # Returns a normal distribution that has x standard deviations
    # within the range supplied.
    def scaled_normal_distribution(n, minimum, maximum, stds_within_range=3.0):
        mean = (maximum + minimum)/2.0
        sigma = (maximum - mean)/stds_within_range
        return np.array(list(np.random.normal(mean, sigma, (n,))))
    
    # Model intialisation
    def __init__(self,
        num_citizens, connections_per_citizen, opinion_distribs,
        num_media, media_distrib, media_change_position,
        max_iterations):

        self.max_iters = max_iterations
        self.iterations = 0
        self.running = True
        self.schedule = SimultaneousActivation(self)
        self.opinion_distributions = {}
        self.num_citizens = num_citizens

        self.num_media = num_media
        
        self.history = []

        self.citizens_cache_updated_at = None
        self.citizens_cache = None

        self.media_cache_updated_at = None
        self.media_cache = None
        
        for opinion_metric in opinion_distribs:
            ditrib_dict = opinion_distribs[opinion_metric]
            try:
                distrib = ditrib_dict["distrib"]
                vals = distrib.split(",")
                vals = [float(val) for val in vals]
                self.opinion_distributions[opinion_metric] = vals
                self.num_citizens = len(vals)
            except:
                vals = SocietyModel.scaled_normal_distribution(num_citizens, ditrib_dict["minimum"], ditrib_dict["maximum"])
                self.opinion_distributions[opinion_metric] = vals
        
        unique_id = 0
        for citizen_id in range(1, self.num_citizens+1):
            extremity = self.opinion_distributions["extremity"][citizen_id-1]
            quality = self.opinion_distributions["quality"][citizen_id-1]
            confidence = self.opinion_distributions["confidence"][citizen_id-1]

            opinion = Opinion(extremity, quality, confidence)
            self.create_citizen(citizen_id, opinion)
            unique_id = citizen_id

        self.connect_citizens(connections_per_citizen)

        unique_id += 1
        if media_distrib["positions"] and len(media_distrib["positions"]) > 0:
            media_position_distribution = media_distrib["positions"]
            self.num_media = len(media_position_distribution)
        else:
            media_position_distribution = SocietyModel.scaled_normal_distribution(num_media, media_distrib["minimum"], media_distrib["maximum"])

        for index, media_id in enumerate(range(unique_id, unique_id + self.num_media)):
            self.create_media(media_id, media_position_distribution[index], media_change_position)
            unique_id = media_id

    def create_citizen(self, unique_id, opinion):
        citizen = Citizen(unique_id, self, opinion)
        self.schedule.add(citizen)

    def create_media(self, unique_id, position, media_change_position):
        media = Media(unique_id, self, position, media_change_position)
        self.schedule.add(media)
        
    def connect_citizens(self, connections_per_citizen):
        population = self.citizens()
        for citizen in population:
            citizen.connected_nodes = np.random.choice(population, connections_per_citizen, replace=False)
            citizen.connection_strengths = [1 for _ in citizen.connected_nodes]

    def citizens(self):
        if self.citizens_cache_updated_at != self.iterations:
            self.citizens_cache = list(filter(lambda x: isinstance(x, Citizen), self.schedule.agents))
            self.citizens_cache_updated_at = self.iterations

        return self.citizens_cache

    def media(self):
        if self.media_cache_updated_at != self.iterations:
            self.media_cache = list(filter(lambda x: isinstance(x, Media), self.schedule.agents))
            self.media_cache_updated_at = self.iterations

        return self.media_cache
    
    # Advance the model one step.
    def step(self):
        if self.running:
            self.schedule.step()
            self.iterations += 1
            
            self.history.append(list(map(lambda x: x.serialize(), self.schedule.agents)))
            
            if self.iterations > self.max_iters:
                self.running = False

def run_model(config):
    print("\n===============\nRUNNING MODEL\n===============")
    model = SocietyModel(**config)
    while model.running:   
        model.step()
    print("\n========\nCOMPLETE\n========")   
    return model

def export_network_file(graph, data, output_file_name):
    for nodes in range(len(data)):
        graph.add_node(data[nodes]["id"], extremity=float(data[nodes]["extremity"]), type=(data[nodes]["type"]))
    
    for a in range(len(data)):
        for con in range(len(data[a]["connected_nodes"])):
            graph.add_edge(data[a]["id"],data[a]["connected_nodes"][con])
    
    nx.draw(graph, pos=nx.spring_layout(graph))
    nx.write_gexf(graph, output_file_name + ".gexf")
 
# EXPERIMENT CODE 
iterations = 50
connections_per_citizen = 50     
num_citizens = 200 

config_dict = {
    "Experiment4": {
        "num_citizens": num_citizens,
        "connections_per_citizen": connections_per_citizen,
        "opinion_distribs": {
            "extremity": {"minimum": -7, "maximum": 7},
            "quality": {"minimum": -8, "maximum": -4},
            "confidence": {"minimum": 4, "maximum": 8},
        },
        "num_media": 10,
        "media_distrib": {
            "minimum": -2,
            "maximum": 2,
            "positions": [-8, -8, -7, 0, 0, 0, 1, 2, 8, 8]
        },
        "media_change_position": False,
        "max_iterations": iterations,
    }
}

print("Running experiments")
print("Output will be .gefx files for analysis in Gephi")
for config_name in config_dict:
    model_config = config_dict[config_name]
    model = run_model(model_config)
    history = model.history
    
    G = nx.Graph()
    export_network_file(G, history[iterations-1], config_name)
