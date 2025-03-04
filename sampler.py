import copy
import json
import networkx as nx  # type: ignore
import numpy as np  # type: ignore
import random
from typing import Dict, List, Tuple, Set
import utils  # type: ignore


class SamplerMethod:
    """
    Base class for different sampling methods. This class should be inherited by specific sampling method classes.
    
    Attributes:
        seed_graph_path (str): Path to the JSON file containing the seed graph.
        save_on_disk (bool): Flag indicating whether to save the sampled graphs to disk.
        seed_graph (nx.Graph): The seed graph loaded from the JSON file.
        data (dict): The data loaded from the JSON file.
    """
    def __init__(self, seed_graph_path: str, save_on_disk: bool) -> None:
        self.save_on_disk = save_on_disk
        # Load the seed graph from the provided JSON file
        with open(seed_graph_path, 'r') as f:
            self.data = json.load(f)
            edgelist: List[Tuple[int, int]] = self.data['graph']
            self.seed_graph = nx.Graph(edgelist)

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        """
        Abstract method to create a sample. This method should be implemented by subclasses.
        
        Args:
            output_file (str): Path to the directory where the sampled graphs will be saved.
        
        Returns:
            Dict[int, nx.Graph]: A dictionary where keys are trial numbers and values are sampled graphs.
        """
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def save_graph_to_file(self, G: nx.Graph, file_path: str) -> None:
        """
        Save the graph to a JSON file.
        
        Args:
            G (nx.Graph): The graph to be saved.
            file_path (str): Path to the file where the graph will be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            new_graph = copy.deepcopy(self.data)
            new_graph['graph'] = list(G.edges())
            json.dump(new_graph, f, ensure_ascii=False)



class Once(SamplerMethod):
    """
    A sampling method that modifies a seed graph by randomly adding or removing edges based on a stay probability.
    Attributes:
        stay_probability (float): The probability of staying with the current graph configuration.
        trials (int): The number of trials to perform.
        save_on_disk (bool): Whether to save the sampled graphs to disk.
    Args:
        seed_graph_path (str): The path to the seed graph file.
        stay_probability (float): The probability of staying with the current graph configuration.
        trials (int): The number of trials to perform.
        save_on_disk (bool): Whether to save the sampled graphs to disk.
    Methods:
        create_sample(output_file: str) -> Dict[int, nx.Graph]:
            Creates samples by modifying the seed graph and returns a dictionary of sampled graphs.
            Args:
                output_file (str): The directory path where the sampled graphs will be saved if save_on_disk is True.
            Returns:
                Dict[int, nx.Graph]: A dictionary where keys are trial numbers and values are the sampled graphs.
    """
    def __init__(self, seed_graph_path: str, stay_probability: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, save_on_disk)
        self.stay_probability = stay_probability
        self.trials = trials

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        """
        Generates a series of graph samples by randomly adding or removing edges from the seed graph.
        Parameters:
        output_file (str): The directory path where the graph samples will be saved if `save_on_disk` is True.
        Returns:
        Dict[int, nx.Graph]: A dictionary where the keys are trial numbers and the values are the corresponding graph samples.
        """
        G: nx.Graph = self.seed_graph.copy()
        edges: List[Tuple[int, int]] = list(G.edges())
        non_edges: List[Tuple[int, int]] = list(nx.non_edges(G))

        output: Dict[int, nx.Graph] = {}

        for trial_number in range(self.trials):
            if np.random.rand() < self.stay_probability:
                if np.random.rand() < 0.5:
                    if edges:
                        # Randomly remove an edge
                        selected_edge = random.choice(edges)
                        G.remove_edge(*selected_edge)
                        edges.remove(selected_edge)
                        non_edges.append(selected_edge)
                else:
                    if non_edges:
                        # Randomly add a non-edge
                        selected_edge = random.choice(non_edges)
                        G.add_edge(*selected_edge)
                        edges.append(selected_edge)
                        non_edges.remove(selected_edge)
            
            # Save the current graph after modifications
            output[trial_number] = G.copy()
            if self.save_on_disk:
                file_path = f"{output_file}/{trial_number + 1}.json"
                self.save_graph_to_file(output[trial_number], file_path)
        
        return output


class All(SamplerMethod):
    """
    A class used to implement the All sampling method.
    Attributes
    ----------
    flipping_probability : float
        The probability with which edges are flipped (removed or added).
    trials : int
        The number of trials to perform.
    save_on_disk : bool
        A flag indicating whether to save the generated samples to disk.
    Methods
    -------
    create_sample(output_file: str) -> Dict[int, nx.Graph]:
        Generates samples by flipping edges in the seed graph based on the flipping probability.
        Parameters
        ----------
        output_file : str
            The directory path where the generated samples will be saved if save_on_disk is True.
        Returns
        -------
        Dict[int, nx.Graph]
            A dictionary where keys are trial numbers and values are the sampled graphs.
    """
    def __init__(self, seed_graph_path: str, flipping_probability: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, save_on_disk)
        self.flipping_probability = flipping_probability
        self.trials = trials

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        
        
        G: nx.Graph = self.seed_graph.copy()
        output: Dict[int, nx.Graph] = {}

        for trial_number in range(self.trials):
            removed_edges: Set[Tuple[int, int]] = set()

            # Remove edges based on flipping probability
            for e in list(G.edges()):
                if random.random() < self.flipping_probability:
                    G.remove_edge(*e)
                    removed_edges.add(e)

            # Add non-edges based on flipping probability
            for e in nx.non_edges(G):
                if e not in removed_edges:
                    if random.random() < self.flipping_probability:
                        G.add_edge(*e)

            # Store the modified graph and save it to file
            output[trial_number] = G.copy()
            if self.save_on_disk:
                file_path = f"{output_file}/{trial_number + 1}.json"
                self.save_graph_to_file(output[trial_number], file_path)
        
        return output

class InProb(SamplerMethod):
    """
        InProb class for sampling methods that modifies a seed graph by randomly adding or removing edges 
        based on given probabilities for addition and removal.
            
        Attributes:
            p_a (float): The probability of adding an edge.
            p_r (float): The probability of removing an edge.
            trials (int): The number of trials to perform.
            save_on_disk (bool): Flag indicating whether to save the sampled graphs to disk.
            
        Args:
            seed_graph_path (str): The path to the seed graph file.
            p_a (float): The probability of adding an edge.
            p_r (float): The probability of removing an edge.
            trials (int): The number of trials to perform.
            save_on_disk (bool): Flag indicating whether to save the sampled graphs to disk.
            
        Methods:
            create_sample(output_file: str) -> Dict[int, nx.Graph]:
                Generates samples by modifying the seed graph and returns a dictionary of sampled graphs.
                    Parameters:
                        output_file (str): The directory path where the sampled graphs will be saved if save_on_disk is True.
                    Returns:
                        Dict[int, nx.Graph]: A dictionary where keys are trial numbers and values are the sampled graphs.
    """
    def __init__(self, seed_graph_path: str, p_a: float, p_r: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, save_on_disk)
        self.p_a = p_a
        self.p_r = p_r
        self.trials = trials

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        """
            Generates a series of graph samples by randomly adding or removing edges from the seed graph.
                
            Parameters:
                output_file (str): The directory path where the graph samples will be saved if `save_on_disk` is True.
                
            Returns:
                Dict[int, nx.Graph]: A dictionary where the keys are trial numbers and the values are the corresponding graph samples.
        """
        G: nx.Graph = self.seed_graph.copy()
        output: Dict[int, nx.Graph] = {}
        for trial_number in range(self.trials):
            removed_edges: Set[Tuple[int, int]] = set()

            # Remove edges based on removal probability
            for e in list(G.edges()):
                if random.random() < self.p_r:
                    G.remove_edge(*e)
                    removed_edges.add(e)

            # Add non-edges based on addition probability
            for e in nx.non_edges(G):
                if e not in removed_edges:
                    if random.random() < self.p_a:
                        G.add_edge(*e)

                    # Store the modified graph and save it to file
            output[trial_number] = G.copy()
            if self.save_on_disk:
                file_path = f"{output_file}/{trial_number + 1}.json"
                self.save_graph_to_file(output[trial_number], file_path)
                
        return output

class Merit(SamplerMethod):
    """
    Merit class for sampling methods.

    Args:
        seed_graph_path (str): The path to the seed graph.
        modification_ratio (float): The ratio of modifications to be applied.
        trials (int): The number of trials to perform.
        save_on_disk (bool): Flag indicating whether to save results on disk.

    Returns:
        None
    """
    def __init__(self, seed_graph_path: str, modification_ratio: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, save_on_disk)
        self.modification_ratio = modification_ratio
        self.trials = trials


class MeritWithReplace(Merit):
    """
    MeritWithReplace is a subclass of Merit that modifies a seed graph by randomly 
    replacing a portion of its edges and optionally saves the modified graphs to disk.
    Attributes:
        seed_graph_path (str): Path to the seed graph file.
        modification_ratio (float): Ratio of edges to be modified.
        trials (int): Number of trials to perform.
        save_on_disk (bool): Flag to save the modified graphs to disk.
    Methods:
        create_sample(output_file: str) -> Dict[int, nx.Graph]:
            Generates modified samples of the seed graph.
            Parameters:
                output_file (str): Directory path to save the modified graphs.
            Returns:
                Dict[int, nx.Graph]: A dictionary where keys are trial numbers and values are modified graphs.
        modify_graph(G: nx.Graph) -> nx.Graph:
            Modifies the given graph by randomly replacing a portion of its edges.
            Parameters:
                G (nx.Graph): The graph to be modified.
            Returns:
                nx.Graph: The modified graph.
    """
    def __init__(self, seed_graph_path: str, modification_ratio: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, modification_ratio, trials, save_on_disk)

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        G: nx.Graph = self.seed_graph.copy()
        output: Dict[int, nx.Graph] = {}
        
        for trial_number in range(self.trials):
            modified_graph = self.modify_graph(G)
            output[trial_number] = modified_graph.copy()
            if self.save_on_disk:
                file_path = f"{output_file}/{trial_number + 1}.json"
                self.save_graph_to_file(modified_graph, file_path)
        
        return output
    
    def modify_graph(self, G: nx.Graph) -> nx.Graph:
        tau = int(self.modification_ratio * G.number_of_edges() / 2)
        edge_list = list(G.edges())
        
        if tau > G.number_of_edges():
            edges_to_delete = list(G.edges())
        else:
            edges_to_delete = random.choices(edge_list, k=tau)
        
        G.remove_edges_from(edges_to_delete)
        edges_to_add = random.choices(list(nx.non_edges(G)), k=tau)
        G.add_edges_from(edges_to_add)
        
        return G


class MeritWithoutReplace(Merit):
    """
    MeritWithoutReplace is a class that extends the Merit class to create samples of a graph 
    by modifying it without replacement.
    Attributes:
        seed_graph_path (str): Path to the seed graph file.
        modification_ratio (float): Ratio of edges to modify in the graph.
        trials (int): Number of trials to perform.
        save_on_disk (bool): Flag to save the modified graphs on disk.
    Methods:
        create_sample(output_file: str) -> Dict[int, nx.Graph]:
            Creates samples of the graph by modifying it and optionally saves them to disk.
            Parameters:
                output_file (str): Directory path to save the modified graphs.
            Returns:
                Dict[int, nx.Graph]: A dictionary where keys are trial numbers and values are modified graphs.
        modify_graph(G: nx.Graph) -> nx.Graph:
            Modifies the given graph by removing and adding edges without replacement.
            Parameters:
                G (nx.Graph): The graph to be modified.
            Returns:
                nx.Graph: The modified graph.
    """
    def __init__(self, seed_graph_path: str, modification_ratio: float, trials: int, save_on_disk: bool) -> None:
        super().__init__(seed_graph_path, modification_ratio, trials, save_on_disk)

    def create_sample(self, output_file: str) -> Dict[int, nx.Graph]:
        G: nx.Graph = self.seed_graph.copy()
        output: Dict[int, nx.Graph] = {}
        
        for trial_number in range(self.trials):
            modified_graph = self.modify_graph(G)
            output[trial_number] = modified_graph.copy()
            if self.save_on_disk:
                file_path = f"{output_file}/{trial_number + 1}.json"
                self.save_graph_to_file(modified_graph, file_path)
        
        return output
    
    def modify_graph(self, G: nx.Graph) -> nx.Graph:
        tau = int(self.modification_ratio * G.number_of_edges() / 2)
        edge_list = list(G.edges())
        
        if tau > G.number_of_edges():
            edges_to_delete = list(G.edges())
        else:
            edges_to_delete = random.sample(edge_list, k=tau)
        
        G.remove_edges_from(edges_to_delete)
        new_edges_to_add = utils.get_unique_elements(list(nx.non_edges(G)), edges_to_delete)
        if tau > len(new_edges_to_add):
            edges_to_add = new_edges_to_add
        else:
            edges_to_add = random.sample(new_edges_to_add, k=tau)
        G.add_edges_from(edges_to_add)
        
        return G
 
# # Example usage of the Once class
# def example_usage_once(seed_graph_path: str, stay_probability: float, trials: int, save_on_disk: bool, output_file: str) -> None:
    

#     sampler = Once(seed_graph_path, stay_probability, trials, save_on_disk)
#     sampled_graphs = sampler.create_sample(output_file)

#     for trial, graph in sampled_graphs.items():
#         print(f"Trial {trial}: {graph.edges()}")
#         print(graph.number_of_edges())




# # Example usage of the All class
# def example_usage_all(seed_graph_path: str, flipping_probability: float, trials: int, save_on_disk: bool, output_file: str) -> None:
#     # seed_graph_path = '../AIDS/4.json'
#     # flipping_probability = 0.3
#     # trials = 5
#     # save_on_disk = True
#     # output_file = './output_dataset/All'

#     sampler = All(seed_graph_path, flipping_probability, trials, save_on_disk)
#     sampled_graphs = sampler.create_sample(output_file)

#     for trial, graph in sampled_graphs.items():
#         print(f"Trial {trial}: {graph.edges()}")
#         print(graph.number_of_edges())



# # Example usage of the MeritWithReplace class
# def example_usage_merit_with_replace(seed_graph_path: str, modification_ratio: float, trials: int, save_on_disk: bool, output_file: str) -> None:
#     # seed_graph_path = '../AIDS/4.json'
#     # modification_ratio = 1
#     # trials = 5
#     # save_on_disk = True
#     # output_file = './output_dataset/Merit_R'

#     sampler = MeritWithReplace(seed_graph_path, modification_ratio, trials, save_on_disk)
#     sampled_graphs = sampler.create_sample(output_file)

#     for trial, graph in sampled_graphs.items():
#         print(f"Trial {trial}: {graph.edges()}")
#         print(graph.number_of_edges())



# # Example usage of the MeritWithoutReplace class
# def example_usage_merit_without_replace(seed_graph_path: str, modification_ratio: float, trials: int, save_on_disk: bool, output_file: str) -> None:
#     # seed_graph_path = '../AIDS/4.json'
#     # modification_ratio = 1
#     # trials = 5
#     # save_on_disk = True
#     # output_file = './output_dataset/Merit_WR'

#     sampler = MeritWithoutReplace(seed_graph_path, modification_ratio, trials, save_on_disk)
#     sampled_graphs = sampler.create_sample(output_file)

#     for trial, graph in sampled_graphs.items():
#         print(f"Trial {trial}: {graph.edges()}")
#         print(graph.number_of_edges())


# """ if __name__ == "__main__":
#     folder_names = ['Once', 'All', 'Merit_R', 'Merit_WR']
#     for i, sampler_class in enumerate([Once, All, MeritWithReplace, MeritWithoutReplace]):
#         sampler = sampler_class('../AIDS/4.json', 0.5, 10, True)
#         sampler.create_sample('./output_dataset/'+folder_names[i]) """
# print('Once')
# example_usage_once('../AIDS/4.json', 0.5, 5, True, './output_dataset/Once')
# print('All')
# example_usage_all('../AIDS/4.json', 0.3, 5, True, './output_dataset/All')
# print('MeritWithReplace')
# example_usage_merit_with_replace('../AIDS/4.json', 0.5, 5, True, './output_dataset/Merit_R')
# print('MeritWithoutReplace')
# example_usage_merit_without_replace('../AIDS/4.json', 0.5, 5, True, './output_dataset/Merit_WR')