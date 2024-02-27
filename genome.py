# This code is from [rt_neat](https://github.com/ThiloK1410/rt_neat)
# Author: ThiloK1410
# File: rt_neat.py


import random

import numpy as np


class Genome:
    class Node:
        def __init__(self, bias):
            self.bias = bias

            # all edges which end in this node
            self.inputs = []

            self.value = 0
            self.next_value = 0

        def get_next_value(self):
            value = 0
            for edge in self.inputs:
                value += edge.get_value()

            self.next_value = value + self.bias

        def finish_activation(self):
            self.value = self.next_value
            self.next_value = 0

        def add_input(self, edge):
            self.inputs.append(edge)

        def remove_input(self, edge):
            self.inputs.remove(edge)

    class Edge:
        def __init__(self, start, end, weight):
            self.weight = weight
            self.start = start
            self.end = end
            end.add_input(self)

        def __del__(self):
            self.end.remove_input(self)

        def get_value(self):
            value = self.start.value * self.weight
            return value

    def __init__(self, inputs, outputs):
        self.nodes = []
        self.edges = []

        # classification of nodes (all are also in self.nodes)
        self.inputs = []
        self.outputs = []
        self.hidden_nodes = []

        for i in range(inputs):
            self.add_node(bias=0, input=True)

        for i in range(outputs):
            self.add_node(self.get_random(), output=True)

    def add_edge(self, start: Node, end: Node, weight: float):
        self.edges.append(self.Edge(start, end, weight=weight))

    # tries to add random edge if it already exists, it does nothing
    def add_random_edge(self, chance=1.0):
        if random.random() <= chance:
            end_choices = self.hidden_nodes + self.outputs
            start_node = random.choice(self.nodes)
            end_node = random.choice(end_choices)
            self.add_edge(start_node, end_node, self.get_random())

    def split_random_edge(self, chance=1.0):
        if random.random() <= chance and len(self.edges) > 0:
            edge = random.choice(self.edges)
            new_node = self.add_node(bias=self.get_random())
            self.add_edge(edge.start, new_node, weight=self.get_random())
            self.add_edge(new_node, edge.end, weight=self.get_random())
            self.edges.remove(edge)

    def add_node(self, bias, input=False, output=False):
        node = self.Node(bias=bias)
        self.nodes.append(node)
        if input:
            self.inputs.append(node)
        if output:
            self.outputs.append(node)
        if not (input or output):
            self.hidden_nodes.append(node)
        return node

    def activate(self, input_values: np.array([])):
        if input_values.size != len(self.inputs):
            raise ValueError("Given input does not fit inputs of network.")

        for i, input in enumerate(np.nditer(input_values)):
            self.inputs[i].value = input
        for node in self.nodes:
            node.get_next_value()
        for node in self.nodes:
            node.finish_activation()

        out = []
        for output_node in self.outputs:
            out.append(output_node.value)

        return out

    def mutate_all_values(self, deviation=0.1):
        for node in self.hidden_nodes + self.outputs:
            node.bias += self.get_random(deviation)
        for edge in self.edges:
            edge.weight += self.get_random(deviation)

    # getting an offspring of genome and applying all mutations
    def get_offspring(self):
        offspring = Genome.from_dict(self.as_dict())
        offspring.mutate_all_values()
        offspring.add_random_edge(chance=0.6)
        offspring.split_random_edge(chance=0.5)

        return offspring

    def as_dict(self):
        nodes = dict()
        for i, node in enumerate(self.nodes):
            nodes[str(i)] = {
                "bias": node.bias,
                "is_input": node in self.inputs,
                "is_output": node in self.outputs,
                "is_hidden": node in self.hidden_nodes,
            }

        edges = dict()
        for i, edge in enumerate(self.edges):
            edges[str(i)] = {
                "weight": edge.weight,
                "start": self.nodes.index(edge.start),
                "end": self.nodes.index(edge.end),
            }

        return {
            "nodes": nodes,
            "edges": edges,
            "input_count": len(self.inputs),
            "output_count": len(self.outputs),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_dict(cls, dict: dict):
        genome = Genome(dict["input_count"], dict["output_count"])

        for i in range(dict["node_count"]):
            if dict["nodes"][str(i)]["is_hidden"]:
                genome.add_node(dict["nodes"][str(i)]["bias"])

        for i in range(dict["edge_count"]):
            genome.add_edge(
                genome.nodes[dict["edges"][str(i)]["start"]],
                genome.nodes[dict["edges"][str(i)]["end"]],
                dict["edges"][str(i)]["weight"],
            )

        return genome

    @staticmethod
    def get_random(standard_deviation=1.0, mean_value=0.0):
        return random.normalvariate(mean_value, standard_deviation)
