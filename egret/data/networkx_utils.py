import networkx
from .model_data import ModelData
from pprint import pprint
from .data_utils import zip_items


def get_networkx_graph(model_data: ModelData, handle_reverse_pairs='error') -> networkx.Graph:
    assert handle_reverse_pairs in {'error', 'remove', 'keep'}
    model_data = model_data.clone_in_service()
    graph = networkx.Graph()
    buses = dict(model_data.elements(element_type='bus'))
    branch_attrs = model_data.attributes(element_type='branch')
    bus_pairs = zip_items(branch_attrs['from_bus'], branch_attrs['to_bus'])

    if handle_reverse_pairs == 'error':
        unique_bus_pairs = dict((val, None) for idx, val in bus_pairs.items())
        for fb, tb in unique_bus_pairs.keys():
            assert (tb, fb) not in unique_bus_pairs
    elif handle_reverse_pairs == 'remove':
        unique_bus_pairs = dict()
        for fb, tb in bus_pairs.values():
            if (tb, fb) not in unique_bus_pairs:
                unique_bus_pairs[(fb, tb)] = None
    else:
        unique_bus_pairs = dict((val, None) for idx, val in bus_pairs.items())

    for bus_name in buses.keys():
        graph.add_node(bus_name)

    for from_bus, to_bus in unique_bus_pairs.keys():
        graph.add_edge(from_bus, to_bus)

    return graph
