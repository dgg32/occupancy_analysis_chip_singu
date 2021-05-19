
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
numeral_ind_dict = {}
alpha_ind_dict = {}
v = 6
cannon_base_list = ['A', 'C', 'G', 'T', 'E', 'M']
# [center, upper left, up, upper right, left, right, lower left, down, lower right]
direction_dict = {"horiz": [4, 5], 'vert': [2, 7], '': [1, 2, 3, 4, 5, 6, 7, 8]}
for (a, b) in itt.combinations('ACGT', 2):
    if (a in alpha_ind_dict.keys()):
        alpha_ind_dict[a].append(v)
    else:
        alpha_ind_dict[a] = [v]

    if (b in alpha_ind_dict.keys()):
        alpha_ind_dict[b].append(v)
    else:
        alpha_ind_dict[b] = [v]
    cannon_base_list.append(a + b)
    v += 1
for key in alpha_ind_dict.keys():
    alpha_ind_dict[key] = list(set(alpha_ind_dict[key]))
v = 6
for (a, b) in itt.combinations([0, 1, 2, 3], 2):
    if (a in numeral_ind_dict.keys()):
        numeral_ind_dict[a].append(v)
    else:
        numeral_ind_dict[a] = [v]

    if (b in numeral_ind_dict.keys()):
        numeral_ind_dict[b].append(v)
    else:
        numeral_ind_dict[b] = [v]
    if v in numeral_ind_dict.keys():
        numeral_ind_dict[v].extend([a, b])
    else:
        numeral_ind_dict[v] = [a, b]

    v += 1
for key in numeral_ind_dict.keys():
    numeral_ind_dict[key] = list(set(numeral_ind_dict[key]))


def neighbor_to_matches(neighbor_arr, mixed_splits=True, split_type=''):
    out = np.zeros(neighbor_arr.shape[0])
    type_vals = np.unique(neighbor_arr[:, 0])
    print(type_vals)
    for i in direction_dict[split_type]:
        out += (neighbor_arr[:, 0] == neighbor_arr[:, i]).astype(int)
    if mixed_splits:
        type_vals = np.unique(neighbor_arr[:, 0])
        print(type_vals)
        for w in type_vals:
            if w in numeral_ind_dict.keys():
                t_bool = neighbor_arr[:, 0] == w
                to_check = numeral_ind_dict[w]
                print(w, to_check)
                for t in to_check:
                    for i in direction_dict[split_type]:
                        out[t_bool.ravel()] += (neighbor_arr[t_bool.ravel(), i] == t).astype(int)

    return out

def filter_neighbors_by_base(neighbor_arr, base_ind='Null', filter_mixed=False, include_mixed=False):
    if filter_mixed:
        neighbor_base_bool = (neighbor_arr[:, 0] < 6)
        if base_ind != 'Null':
            neighbor_base_bool = (neighbor_arr[:, 0] == base_ind)*neighbor_base_bool
    else:
        if base_ind == 'Null':
            return neighbor_arr
        neighbor_base_bool = (neighbor_arr[:, 0] == base_ind)
        if include_mixed:
            if base_ind in numeral_ind_dict.keys():
                for mixed_ind in numeral_ind_dict[base_ind]:
                    neighbor_base_bool = np.logical_or(neighbor_base_bool, neighbor_arr[:, 0] == mixed_ind)
    neighbor_arr = neighbor_arr[neighbor_base_bool.ravel()]
    return neighbor_arr


def neighbor_to_single_counting(neighbor_arr):
    sampling = np.random.choice([False, True], neighbor_arr.shape[0], p=[0.8, 0.2])
    return neighbor_arr[sampling]

def get_match_vals(called_neighbor_group, sim_neighbor_group, mixed_splits, split_type):
    called_matches = neighbor_to_matches(called_neighbor_group, mixed_splits, split_type)
    sim_matches = neighbor_to_matches(sim_neighbor_group, mixed_splits, split_type)
    print(sim_matches)
    s_mean = sim_matches.mean()
    print(mixed_splits, split_type, s_mean)
    r_mean = called_matches.mean()
    if s_mean > 0:
        perc = 100*(float(r_mean - s_mean) / float(s_mean))
    else:
        perc = -1
    return perc, r_mean, s_mean

def generate_concordances(real_neighbor_arr, sim_neighbor_arr, base_list, tag, filter_mixed, count_mixed_splits):
    excess_percentage = {}
    sim_means = {}
    real_means = {}
    called_neighbor_group = filter_neighbors_by_base(real_neighbor_arr, base_ind='Null', filter_mixed=filter_mixed,
                                                     include_mixed=count_mixed_splits)
    sim_neighbor_group = filter_neighbors_by_base(sim_neighbor_arr, base_ind='Null', filter_mixed=filter_mixed,
                                                  include_mixed=count_mixed_splits)
    for split_type in direction_dict.keys():
        excess_percentage['_'.join(["Total", tag, split_type])], \
        real_means['_'.join(["Total", tag, split_type])], \
        sim_means['_'.join(["Total", tag, split_type])], =  get_match_vals(called_neighbor_group, sim_neighbor_group,
                                                                           count_mixed_splits, split_type)

    for l, b in enumerate(base_list):
        base_called_neighbor_groups = filter_neighbors_by_base(real_neighbor_arr, l, filter_mixed,
                                                               include_mixed=count_mixed_splits)
        base_sim_neighbor_groups = filter_neighbors_by_base(sim_neighbor_arr, l, filter_mixed,
                                                            include_mixed=count_mixed_splits)
        for split_type in direction_dict.keys():
            print(b, count_mixed_splits)
            excess_percentage['_'.join([b, tag, split_type])], \
            real_means['_'.join([b, tag, split_type])], \
            sim_means['_'.join([b, tag, split_type])], = get_match_vals(base_called_neighbor_groups,
                                                                        base_sim_neighbor_groups,
                                                                        count_mixed_splits, split_type)
    return excess_percentage, real_means, sim_means


def reduce_to_valid(called_neighbor_groups, sim_neighbor_groups):
    # removes DNBs with mixed or empty neighbors
    n_call_bool = ((called_neighbor_groups == 4).sum(axis=1) == 0) * ((called_neighbor_groups == 5).sum(axis=1) == 0)
    called_neighbor_groups = called_neighbor_groups[n_call_bool]
    sim_call_bool = ((sim_neighbor_groups == 4).sum(axis=1) == 0) * ((sim_neighbor_groups == 5).sum(axis=1) == 0)
    sim_neighbor_groups = sim_neighbor_groups[sim_call_bool]
    return called_neighbor_groups, sim_neighbor_groups

def append_dict_entries(target_dict, source_dict):
    # adds source dictionary entries to target_dict
    for key in source_dict.keys():
        if key in target_dict:
            target_dict[key].append(source_dict[key])
        else:
            target_dict[key] = [source_dict[key]]
    return target_dict

def list_dict_to_average(list_dict):
    # takes mean of each list
    out_dict = {}
    for key in list_dict.keys():
        out_dict[key] = np.nan_to_num(np.mean(list_dict[key]))
    return out_dict

def add_traditional_neighbor_metric(called_neighbor_group, sim_neighbor_group):
    called_out = {}
    sim_out = {}
    called_neighbor_group = called_neighbor_group[called_neighbor_group[:, 0] < 4, :]
    sim_neighbor_group = sim_neighbor_group[sim_neighbor_group[:, 0] < 4, :]
    print(np.equal(called_neighbor_group[:, 1:], called_neighbor_group[:, 0].reshape(called_neighbor_group.shape[0], 1)))
    called_out['Total'] = 100.0*(np.equal(called_neighbor_group[:, 0].reshape(called_neighbor_group.shape[0], 1),
                                    called_neighbor_group[:, 1:]).sum(axis=1) > 0).astype(int).mean()
    sim_out['Total'] = 100.0*(np.equal(sim_neighbor_group[:, 0].reshape(sim_neighbor_group.shape[0], 1), sim_neighbor_group[:,
                                                                                                   1:]).sum(axis=1) > 0).astype(int).mean()
    return called_out, sim_out

def run(call_arr, nieghbor_arr):
    if type(call_arr) == str:
        called_bases = np.load(call_arr)
        neighbors_arr = np.load(nieghbor_arr)
    else:
        neighbors_arr = nieghbor_arr - 1
        called_bases = call_arr
    excess_percentage = {}
    sim_means = {}
    real_means = {}
    perc_valid = {'real': [], 'sim': []}
    base_comps = {}
    has_neighbor_sim = {}
    has_neighbor_real = {}
    # real_neighbors = neighbor_to_single_counting(neighbors_arr)
    real_neighbors = neighbors_arr
    edge_mask = (real_neighbors.min(axis=1) > -1)
    for i in range(called_bases.shape[-1]):

        cycle_calls = np.argmax(called_bases[:, :, i], axis=1)
        cycle_calls[called_bases[:, :, i].sum(axis=1) == 0] = 4
        cycle_calls[called_bases[:, :, i].sum(axis=1) > 2] = 5
        ind = 6
        for (j, k) in itt.combinations([0, 1, 2, 3], 2):
            chan1 = called_bases[:, j, i] == 1
            chan2 = called_bases[:, k, i] == 1
            cycle_calls[chan1.ravel()*chan2.ravel()] = ind
            ind += 1

        base_comp = []
        # find proportions
        for l, b in enumerate(cannon_base_list):
            proportion = float((cycle_calls == l).sum()) / cycle_calls.shape[0]
            base_comp.append(proportion)
            if b not in base_comps.keys():
                base_comps[b] = []
            base_comps[b].append(proportion * 100)

        called_neighbor_groups = cycle_calls[real_neighbors][edge_mask]
        # generate simulated array
        for l in range(5):
            simulated_calls = np.random.choice(np.arange(ind).tolist(), cycle_calls.shape, p=base_comp)
            if l == 0:
                sim_neighbor_groups = simulated_calls[real_neighbors]
            else:
                sim_neighbor_groups = np.concatenate((sim_neighbor_groups, simulated_calls[real_neighbors]), axis=0)
        has_neigh_r, has_neigh_sim = add_traditional_neighbor_metric(called_neighbor_groups, sim_neighbor_groups)
        has_neighbor_real = append_dict_entries(has_neighbor_real, has_neigh_r)
        has_neighbor_sim = append_dict_entries(has_neighbor_sim, has_neigh_sim)
        raw_excess_percentage, raw_real_means, raw_sim_means = generate_concordances(called_neighbor_groups,
                                                                                     sim_neighbor_groups,
                                                                                     cannon_base_list,
                                                                                     'Raw_Including_Mixed',
                                                                                     False, True)
        excess_percentage = append_dict_entries(excess_percentage, raw_excess_percentage)
        real_means = append_dict_entries(real_means, raw_real_means)
        sim_means = append_dict_entries(sim_means, raw_sim_means)
        raw_no_mixed_excess_percentage, raw_no_mixed_real_means, raw_no_mixed_sim_means = generate_concordances(called_neighbor_groups, 
                                                                                                                sim_neighbor_groups,
                                                                                                                cannon_base_list, 
                                                                                                                'Raw_Excluding_Mixed',
                                                                                                                False,
                                                                                                                False)
        excess_percentage = append_dict_entries(excess_percentage, raw_no_mixed_excess_percentage)
        real_means = append_dict_entries(real_means, raw_no_mixed_real_means)
        sim_means = append_dict_entries(sim_means, raw_no_mixed_sim_means)
        sim_count = sim_neighbor_groups.shape[0]
        called_count = called_neighbor_groups.shape[0]
        called_neighbor_groups, sim_neighbor_groups = reduce_to_valid(called_neighbor_groups, sim_neighbor_groups)
        for l, b in enumerate(cannon_base_list):
            base_tag = b+'_Valid'
            proportion = float((called_neighbor_groups[:, 0] == l).sum()) / called_neighbor_groups.shape[0]
            if base_tag not in base_comps.keys():
                base_comps[base_tag] = []
            base_comps[base_tag].append(proportion * 100)
        perc_valid['real'].append(100.0*float(called_neighbor_groups.shape[0])/float(called_count))
        perc_valid['sim'].append(100.0*float(sim_neighbor_groups.shape[0])/float(sim_count))
        valid_excess_percentage, valid_real_means, valid_sim_means = generate_concordances(called_neighbor_groups,
                                                                                           sim_neighbor_groups,
                                                                                           cannon_base_list,
                                                                                           'Valid_Including_Mixed',
                                                                                           False, True)
        excess_percentage = append_dict_entries(excess_percentage, valid_excess_percentage)
        real_means = append_dict_entries(real_means, valid_real_means)
        sim_means = append_dict_entries(sim_means, valid_sim_means)
        valid_no_mixed_excess_percentage, valid_no_mixed_real_means, valid_no_mixed_sim_means = generate_concordances(
            called_neighbor_groups,
            sim_neighbor_groups,
            cannon_base_list,
            'Valid_Excluding_Mixed',
            False, False)
        excess_percentage = append_dict_entries(excess_percentage, valid_no_mixed_excess_percentage)
        real_means = append_dict_entries(real_means, valid_no_mixed_real_means)
        sim_means = append_dict_entries(sim_means, valid_no_mixed_sim_means)

    excess_out = list_dict_to_average(excess_percentage)
    real_means = list_dict_to_average(real_means)
    sim_means = list_dict_to_average(sim_means)
    percentage_out = list_dict_to_average(base_comps)
    perc_valid = list_dict_to_average(perc_valid)
    has_neighbor_sim = list_dict_to_average(has_neighbor_sim)['Total']
    has_neighbor_real = list_dict_to_average(has_neighbor_real)['Total']
    return real_means, sim_means, excess_out, percentage_out, perc_valid, has_neighbor_real, has_neighbor_sim