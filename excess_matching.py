
import numpy as np
import matplotlib.pyplot as plt

numeral_ind_dict = {}
alpha_ind_dict = {}
v = 6
cannon_base_list = ['A', 'C', 'G', 'T', 'E', 'M']
for i, a in enumerate('ACGT'):
    for j, b in enumerate('ACGT'):
        numeral_ind_dict[i].append(v)
        alpha_ind_dict[a].append(v)
        cannon_base_list.append(a+b)
        v += 1

def neighbor_to_matches(neighbor_arr):
    return np.equal(neighbor_arr[:, 0].reshape(neighbor_arr.shape[0], 1),
                    neighbor_arr[:, 1:].reshape(neighbor_arr.shape[0], 8)).astype(int).sum(axis=1)

def filter_neighbors_by_base(neighbor_arr, base_ind):
    neigbor_base_bool = (neighbor_arr[:, 0] == base_ind)
    neighbor_arr = neighbor_arr[neigbor_base_bool.ravel()]
    return neighbor_arr

def run(call_arr, nieghbor_arr):
    neighbors_arr = nieghbor_arr - 1
    called_bases = call_arr
    perc_valid = {'real': 0.0, 'sim': 0.0}
    excess_percentage = {'All': [], 'Total_NonN': [], 'A': [], 'C': [], 'G': [], 'T': [], 'E': [], 'M': [],
                         'A_NonN': [], 'C_NonN': [], 'G_NonN': [], 'T_NonN': []}
    sim_means = {'All': [], 'Total_NonN': [], 'A': [], 'C': [], 'G': [], 'T': [], 'E': [], 'M': [],
                'A_NonN': [], 'C_NonN': [], 'G_NonN': [], 'T_NonN': []}
    real_means = {'All': [], 'Total_NonN': [], 'A': [], 'C': [], 'G': [], 'T': [], 'E': [], 'M': [],
                         'A_NonN': [], 'C_NonN': [], 'G_NonN': [], 'T_NonN': []}
    base_comps = {'A': [], 'C': [], 'G': [], 'T': [], 'E': [], 'M': [],
                  'A_NonN': [], 'C_NonN': [], 'G_NonN': [], 'T_NonN': []}
    real_neighbors = neighbors_arr
    edge_mask = (real_neighbors.min(axis=1) > -1)
    for i in range(called_bases.shape[-1]):
        cycle_calls = np.argmax(called_bases[:, :, i], axis=1)
        cycle_calls[called_bases[:, :, i].sum(axis=1) == 0] = 4
        cycle_calls[called_bases[:, :, i].sum(axis=1) > 2] = 5
        ind = 6
        # for j in [0, 1, 2, 3]:
        #     for k in [0, 1, 2, 3]:
        #         if j != k:
        #             chan1 = called_bases[called_bases[:, j, i], :, i] == 1
        #             chan2 = called_bases[called_bases[:, k, i], :, i] == 1
        #             cycle_calls[chan1*chan2] = ind
        #             ind += 1

        base_comp = []
        for l, b in enumerate('ACGTEM'):
            proportion = float((cycle_calls[edge_mask] == l).sum()) / cycle_calls[edge_mask].shape[0]
            base_comp.append(proportion)
            # if b not in base_comps.keys():
            #     base_comps[b] = []
            base_comps[b].append(proportion * 100)
        print(base_comp)

        called_neighbor_groups = cycle_calls[real_neighbors]
        called_matches = neighbor_to_matches(called_neighbor_groups)
        simulated_mean = []
        for i in range(5):
            simulated_calls = np.random.choice(np.arange(ind+1).tolist(), cycle_calls.shape, p=base_comp)
            if i == 0:
                sim_neighbor_groups = simulated_calls[real_neighbors]
            else:
                sim_neighbor_groups = np.concatenate((sim_neighbor_groups, simulated_calls[real_neighbors]), axis=0)
        simulated_matches = neighbor_to_matches(sim_neighbor_groups)
        sim_count = simulated_matches.shape[0]
        simulated_mean = np.mean(simulated_matches)
        sim_means['All'] = simulated_mean
        called_mean = np.mean(called_matches[edge_mask])
        called_count = called_matches[edge_mask].shape[0]
        real_means['All'] = called_mean
        print(simulated_mean, called_mean)
        excess_percentage['All'].append(float(called_mean - simulated_mean) / float(simulated_mean))
        # cycle_calls[n_call_bool == 0] = 4
        print(sim_neighbor_groups.shape)
        print(called_neighbor_groups.shape)
        print(edge_mask.shape)
        for i, b in enumerate('ACGTEM'):
            base_called_neighbor_groups = filter_neighbors_by_base(called_neighbor_groups[edge_mask], i)
            base_sim_neighbor_groups = filter_neighbors_by_base(sim_neighbor_groups, i)
            called_matches = neighbor_to_matches(base_called_neighbor_groups)
            simulated_matches = neighbor_to_matches(base_sim_neighbor_groups)
            simulated_mean = simulated_matches.mean()
            called_mean = called_matches.mean()
            sim_means[b] = simulated_mean
            real_means[b] = called_mean
            print(b, simulated_mean, called_mean)
            excess_percentage[b].append(float(called_mean - simulated_mean) / float(simulated_mean))

        called_neighbor_groups = cycle_calls[real_neighbors][edge_mask]

        n_call_bool = ((called_neighbor_groups == 4).sum(axis=1) == 0)*((called_neighbor_groups == 5).sum(axis=1) == 0)
        called_neighbor_groups = called_neighbor_groups[n_call_bool]
        sim_call_bool = ((sim_neighbor_groups == 4).sum(axis=1) == 0)*((sim_neighbor_groups == 5).sum(axis=1) == 0)
        sim_neighbor_groups = sim_neighbor_groups[sim_call_bool]
        perc_valid['real'] = 100.0*float(called_neighbor_groups.shape[0])/float(called_count)
        perc_valid['sim'] = 100.0*float(sim_neighbor_groups.shape[0])/float(sim_count)
        # for i, b in enumerate('ACGT'):
        #     proportion = float((called_neighbor_groups[:, 0] == i).sum()) / called_neighbor_groups[:, 0].shape[0]
        #         #     base_comp.append(proportion)
        #         #     base_comps[b + '_NonN'].append(proportion * 100)
        # print(base_comp)
        # simulated_calls = np.random.choice([0, 1, 2, 3], cycle_calls.shape, p=base_comp)
        # sim_neighbor_groups = simulated_calls[real_neighbors]
        called_matches = neighbor_to_matches(called_neighbor_groups)
        simulated_matches = neighbor_to_matches(sim_neighbor_groups)
        simulated_mean = simulated_matches.mean()
        called_mean = called_matches.mean()
        sim_means['Total_NonN'] = simulated_mean
        real_means['Total_NonN'] = called_mean
        plt.hist(called_matches, alpha=0.5, label='Real NonN', bins=8)
        plt.hist(simulated_matches, alpha=0.5, label='Simulated', bins=8)
        plt.legend()
        print(simulated_mean, called_mean)
        excess_percentage['Total_NonN'].append(float(called_mean - simulated_mean) / float(simulated_mean))

        for l, b in enumerate('ACGT'):
            base_called_neighbor_groups = filter_neighbors_by_base(called_neighbor_groups, l)
            base_sim_neighbor_groups = filter_neighbors_by_base(sim_neighbor_groups, l)
            called_matches = neighbor_to_matches(base_called_neighbor_groups)
            simulated_matches = neighbor_to_matches(base_sim_neighbor_groups)
            simulated_mean = simulated_matches.mean()
            called_mean = called_matches.mean()
            sim_means[b+'_NonN'] = simulated_mean
            real_means[b+'_NonN'] = called_mean
            print(b+'_NonN', simulated_mean, called_mean)
            base_comps[b + '_NonN'].append(float(base_called_neighbor_groups.shape[0])/float(called_neighbor_groups.shape[0]) * 100)
            excess_percentage[b+'_NonN'].append(float(called_mean - simulated_mean) / float(simulated_mean))
    excess_out = {'All': 0, 'Total_NonN': 0, 'A': 0, 'C': 0, 'G': 0, 'T': 0,
                  'A_NonN': 0, 'C_NonN': 0, 'G_NonN': 0, 'T_NonN': 0, 'E': 0, 'M': 0}
    percentage_out = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'E': 0, 'M': 0,
                      'A_NonN': 0, 'C_NonN': 0, 'G_NonN': 0, 'T_NonN': 0}

    for key in excess_out.keys():
        average_excess = np.mean(excess_percentage[key])
        excess_out[key] = average_excess * 100.0
        if key in percentage_out.keys():
            percentage_out[key] = np.round(np.mean(base_comps[key]), 2)
    return real_means, sim_means, excess_out, percentage_out, perc_valid