def get_unique_elements(list1, list2):
    """
    This function returns a list of elements that are in list1 but not in list2.
    
    :param list1: The first list.
    :param list2: The second list.
    :return: A list of elements that are in list1 but not in list2.
    """
    # Use list comprehension to filter elements
    unique_elements = [element for element in list1 if element not in list2]
    return unique_elements


def hamming(edge_first, non_edge_first, edge_second, non_edge_second):
    mismatching = len(edge_first & non_edge_second) + len(non_edge_first & edge_second)
    return mismatching


