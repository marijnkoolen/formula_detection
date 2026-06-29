"""
visualisation.py — Graphviz-based visualisation helpers.

Provides two unrelated kinds of visualisations:

1. Transition-probability graphs: given a nested mapping of node ->
   node -> probability (as produced elsewhere for pre/post phrase
   transitions), render a directed graph of the nodes reachable from the
   "<PHRASE>" start node, with edges labelled by transition probability.
2. Collatex variant graphs: render a collatex VariantGraph (the result of
   collating multiple text witnesses) as an SVG graphviz Digraph, showing
   both the regular token-sequence edges and the near-match (variant)
   edges.
"""
import re
from collections import defaultdict
from typing import List

import graphviz
import matplotlib.pyplot as plt
from graphviz import Digraph


def get_start_node(transition_probs):
    """Find the "<PHRASE>" start node in a transition-probability mapping.

    Iterates over the keys of ``transition_probs``, sorted by key length
    ascending, and returns the first node whose label is "<PHRASE>". A
    node's label is the node itself if it is a string, or its last
    element if it is a tuple/sequence.

    Args:
        transition_probs: Mapping of source node to a mapping of target
            node to transition probability.

    Returns:
        The start node (matching label "<PHRASE>"), or None if no such
        node is found.
    """
    for source in sorted(transition_probs.keys(), key=lambda x: len(x)):
        source_label = source if isinstance(source, str) else source[-1]
        if source_label == '<PHRASE>':
            print('get_start_node - source in transition_probs:', source in transition_probs)
            return source
    return None


def get_connected_non_var_nodes(transition_probs, curr_node, exclude_var: bool = False,
                                seen_nodes: List[str] = None,
                                debug: int = 0):
    """Recursively collect nodes reachable from a node in a transition graph.

    Performs a depth-first traversal of ``transition_probs`` starting at
    ``curr_node``, collecting every node reached, optionally skipping
    nodes whose label starts with "<VAR" (variable/variant placeholder
    nodes).

    Args:
        transition_probs: Mapping of source node to a mapping of target
            node to transition probability.
        curr_node: The node to start the traversal from.
        exclude_var: If True, do not traverse into (or include) nodes
            whose label starts with "<VAR".
        seen_nodes: List of nodes already visited, used to avoid cycles
            and re-visiting; mutated in place across the recursion. If
            None, initialised from ``connected_nodes``.
        debug: Verbosity level for diagnostic printing; > 0 enables it.

    Returns:
        A list of nodes connected to (reachable from) ``curr_node``,
        including ``curr_node`` itself.
    """
    connected_nodes = [curr_node]
    if seen_nodes is None:
        seen_nodes = [node for node in connected_nodes]
    elif curr_node not in seen_nodes:
        seen_nodes.append(curr_node)
    if debug > 0:
        print('get_connected_non_var_nodes - seen_nodes:', seen_nodes)
    if debug > 0:
        print('get_connected_non_var_nodes - curr_node:', curr_node)
    if curr_node not in transition_probs:
        return connected_nodes
    for next_node in transition_probs[curr_node]:
        if debug > 0:
            print('\tget_connected_non_var_nodes - next_node:', next_node)
            print('\t\ttrans_prob:', transition_probs[curr_node][next_node])
        node_label = next_node if isinstance(next_node, str) else next_node[-1]
        if next_node in seen_nodes:
            if debug > 0:
                print('get_connected_non_var_nodes - skipping previously seen next_node', next_node)
            continue
        if exclude_var and node_label.startswith('<VAR'):
            continue
        next_connected = get_connected_non_var_nodes(transition_probs, next_node,
                                                     exclude_var=exclude_var,
                                                     seen_nodes=seen_nodes, debug=debug)
        for node in next_connected:
            if node not in connected_nodes:
                connected_nodes.append(node)
    return connected_nodes


def get_transition_probs_graph(phrase, transition_probs, direction: str, exclude_var: bool = False,
                               debug: int = 0):
    """Build a graphviz Digraph of transition probabilities for a phrase.

    Starting from the "<PHRASE>" node for the given ``direction`` (pre or
    post), restricts the graph to nodes reachable via
    ``get_connected_non_var_nodes``, then renders nodes and edges with
    edge labels showing the transition probability. For direction "pre",
    edges are drawn target -> source so that the graph reads in the
    natural left-to-right order of preceding context; for "post" edges
    are drawn source -> target.

    Args:
        phrase: The phrase string to use as the label for the "<PHRASE>"
            node.
        transition_probs: Mapping with "pre" and "post" keys, each a
            nested mapping of source node to target node to transition
            probability.
        direction: Either "pre" or "post", selecting which sub-mapping of
            ``transition_probs`` to visualise.
        exclude_var: If True, exclude variable/variant placeholder nodes
            (label starting with "<VAR") from the graph.
        debug: Verbosity level for diagnostic printing; > 0 enables it.

    Returns:
        A graphviz.Digraph (PNG format) representing the transition graph.

    Raises:
        ValueError: If ``direction`` is not "pre" or "post".
    """
    if direction not in {'pre', 'post'}:
        raise ValueError('direction must be "pre" or "post"')
    dot = Digraph(comment=f'Transition probabilities for phrase "{phrase}"', format='png')
    dot.graph_attr['rankdir'] = 'LR'
    node_map = {}

    start_node = get_start_node(transition_probs[direction])
    if debug > 0:
        print('get_transition_probs_graph - start_node:', start_node)
    include_nodes = get_connected_non_var_nodes(transition_probs[direction], start_node,
                                                exclude_var=exclude_var, debug=debug)

    # for source in sorted(transition_probs[direction].keys(), key=lambda x: len(x)):
    for source in include_nodes:
        source_label = source if isinstance(source, str) else source[-1]
        if source not in node_map:
            node_map[source] = str(len(node_map))
            if source_label == '<PHRASE>':
                source_label = phrase
            dot.node(node_map[source], source_label)
        source_id = node_map[source]
        for target in transition_probs[direction][source]:
            if target not in node_map:
                node_map[target] = str(len(node_map))
                target_label = target if isinstance(target, str) else target[-1]
                dot.node(node_map[target], target_label)
            target_id = node_map[target]
            if direction == 'pre':
                dot.edge(target_id, source_id, label=f"{transition_probs[direction][source][target]: >.2f}")
            else:
                dot.edge(source_id, target_id, label=f"{transition_probs[direction][source][target]: >.2f}")
    return dot


# visualize the variant graph into SVG format
def display_variant_graph_as_svg(graph, output, show_labels: bool = True):
    """Render a collatex VariantGraph as a graphviz SVG Digraph.

    Adds a node per graph vertex (either a simple label, or — unless
    ``output`` is "svg_simple" — an HTML table listing each distinct
    reading and the witnesses that share it, ranked via
    VariantGraphRanking), an edge per regular token-sequence connection,
    and a dashed edge per near-match (fuzzy variant) connection labelled
    with its similarity weight. Vertices that share the same rank are
    grouped into a same-rank subgraph so they are aligned in the layout.

    Args:
        graph: A collatex VariantGraph (with ``.graph`` and
            ``.near_graph`` networkx graphs) to visualise.
        output: Output mode; "svg_simple" renders plain text labels,
            anything else renders the detailed HTML table per node.
        show_labels: If False, omit edge labels on the regular
            token-sequence edges.

    Returns:
        A graphviz.Digraph (SVG format) representing the variant graph.
    """
    from collatex.core_classes import VariantGraphRanking

    a = graphviz.Digraph(format="svg", graph_attr={'rankdir': 'LR'})
    counter = 0
    mapping = {}
    ranking = VariantGraphRanking.of(graph)

    # add nodes
    for n in graph.graph.nodes():
        counter += 1
        mapping[n] = str(counter)
        if output == "svg_simple":
            label = n.label
            if label == '':
                label = '#'
            a.node(mapping[n], label=label)
        else:
            rank = ranking.byVertex[n]
            readings = ["<TR><TD ALIGN='LEFT'><B>" + n.label + "</B></TD><TD ALIGN='LEFT'>exact: " + str(
                rank) + "</TD></TR>"]
            reverse_dict = defaultdict(list)
            for key, value in n.tokens.items():
                reverse_dict["".join(
                    re.sub(r'>', r'&gt;', re.sub(r'<', r'&lt;', item.token_data["t"])) for item in value)].append(
                    key)
            for key, value in sorted(reverse_dict.items()):
                reading = (
                    "<TR><TD ALIGN='LEFT'><FONT FACE='Bukyvede'>{}</FONT></TD><TD ALIGN='LEFT'>{}</TD></TR>").format(
                    key, ', '.join(value))
                readings.append(reading)
            a.node(mapping[n], label='<<TABLE CELLSPACING="0">' + "".join(readings) + '</TABLE>>')

    # add regular (token sequence) edges
    for u, v, edgedata in graph.graph.edges(data=True):
        # print('regular edges ', u, v, edgedata)
        label = edgedata['label']
        if show_labels is False:
            label = None
        a.edge(mapping[u], mapping[v], label=label)

    # add near-match edges
    # TODO: Show all near edges (currently), or just the top one?
    for u, v, edgedata in graph.near_graph.edges(data=True):
        # print('near-match edges ', u, v, edgedata)
        label = str('{:3.2f}'.format(edgedata['weight']))
        a.edge(mapping[u], mapping[v], style='dashed', label=label)
    # Add rank='same' information
    for key, value in ranking.byRank.items():
        # print(key, value)
        # print(key, value, len(value))
        # print(key, set(value), len(set(value)))
        tmp = graphviz.Digraph(graph_attr={'rank': 'same'})
        for n in [mapping[item] for item in value]:
            tmp.node(n)
        a.subgraph(tmp)
    # diagnostic, not for production
    # dot = a.draw(prog='dot')
    # print(dot.decode(encoding='utf-8'))
    # # display using the IPython SVG module
    return a
