def get_state_mapping_from_list(states_list):
    return list(set([(x[0], x[1]) for x in states_list.tolist()]))
