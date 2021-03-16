import numpy as np
import os
import sys, getopt
import NAN_support_lib as sup
import NAN_neuron_network as cin


class SFRC_network_builder:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/'

        print('Main working directory: ' + str(self.main_Path))
        print('Networks directory: ' + str(self.network_Path))
        print('Data directory: ' + str(self.data_Path))

    def get_neuron_func_ranges(self,res_size,input_size,output_size):

        '''
        DEFINES NEURON POPULATION RANGES

        :param res_size: NUMBER OF NEURONS IN RECURRENT NETWORK (N)
        :param input_size: NUMBER OF NEURONS IN INPUT LAYER
        :param output_size: NUMBER OF NEURONS IN OUTPUT LAYER

        :return: LIST OF RANGES
        '''

        res_range = [0,res_size,res_size]
        inp_range = [res_size,res_size+input_size,input_size]
        out_range = [res_size+input_size,res_size+input_size+output_size,output_size]

        return ['res_range', 'inp_range', 'out_range'], [res_range, inp_range, out_range]

    def get_neuron_degs(self,w_mask):

        '''
        
        :param w_mask: CONNECTION MASK MATRIX
        
        :return: INPUT DEGREE, OUPTUT DEGREE, TOTAL DEGREE COUNTS PER NEURON
        '''

        out_degs = np.sum(w_mask,axis=0,dtype=np.int)
        in_degs = np.sum(w_mask, axis=1,dtype=np.int)
        total_degs = np.add(out_degs,in_degs)

        return in_degs,out_degs,total_degs


    def compute_degree_based_probability(self,deg_arr):

        '''
        
        :param deg_arr: NEURON DEGREE COUNT VECTOR
        
        :return: PROBABILITY PER NEURON BASED ON DEGREE COUNT
        '''

        total_degs = np.sum(deg_arr)
        probs = np.divide(deg_arr,total_degs)

        return probs


    def SF_reservoir_mask_generator(self, num_neurons_total, out_conn_num, starter_neuron_num):

        '''
        
        :param num_neurons_total: NUMBER OF NEURONS IN RECURRENT NETWORK
        :param out_conn_num: LIMIT OF OUTWARD CONNECTIONS PER NEURON
        :param starter_neuron_num: SEED NUMBER OF NEURONS FOR RICH CLUB
        
        :return: 
        '''

        out_con_ini = starter_neuron_num -1
        num_total_connections = out_conn_num*num_neurons_total

        w_mask = np.zeros((num_neurons_total, num_neurons_total), dtype=np.float32)

        uncreated_neurons = []
        created_neurons_all = []

        for i in range(0, num_neurons_total):
            uncreated_neurons.append(i)

        for j in range(0, starter_neuron_num):
            uncreated_neurons.remove(j)
            created_neurons_all.append(j)

        # initialize network with starter neurons assume they are all excitatory
        for n in range(0, starter_neuron_num):
            output_conns_ini = np.random.permutation(starter_neuron_num)
            # print('shape of output_conns_ini',np.shape(output_conns_ini))
            if len(np.where(output_conns_ini[0:out_con_ini] == n)[0]) > 0:
                output_conns_final = output_conns_ini[0:out_con_ini + 1]
            else:
                output_conns_final = output_conns_ini[0:out_con_ini]

            w_mask[output_conns_final, n] = 1.0

            ### get rid of self connections
            w_mask[np.arange(num_neurons_total), np.arange(num_neurons_total)] = 0.0

        ### LOOP THAT ADDS NEW NEURONS AND ALSO INTRODUCES OUTGOING CONNECTIONS TO FROM EXISTING NEURONS
        print('ADDING NEURONS')
        for n in range(starter_neuron_num, num_neurons_total):
            # print('ADDING NEURON: ' + str(n))
            # randomly selects neuron to create
            neuron_created_idx = np.random.randint(0, len(uncreated_neurons), size=1)[0]
            neuron_created = uncreated_neurons[neuron_created_idx]

            # does some accounting of created uncreateed neurons
            uncreated_neurons.remove(neuron_created)
            created_neurons_all.append(neuron_created)

            # computes probabilities of connection between new neuron and all other existing neurons
            # by default if neuron has no connections it doesn't exist and has 0 probability to have connections
            in_degs, out_degs, total_degs = self.get_neuron_degs(w_mask=w_mask)
            deg_probs = self.compute_degree_based_probability(total_degs)
            conn_idx = []

            for kkk in range(0, 5):
                neuron_conn_status = np.clip(np.ceil(np.subtract(deg_probs, np.random.random(size=len(deg_probs)))),0,1)
                conn_idx.extend(np.where(neuron_conn_status == 1)[0])
                conn_idx = np.unique(np.asarray(conn_idx)).tolist()
                if len(np.where(conn_idx == neuron_created)[0]) > 0:
                    conn_idx.remove(neuron_created)

            conn_idx = conn_idx[0:out_con_ini]

            w_mask[conn_idx, neuron_created] = 1.0

            ### get rid of self connections
            w_mask[np.arange(num_neurons_total), np.arange(num_neurons_total)] = 0.0

        print('TIEING OUT STARTER NEURONS WITH CONNECTIONS...WILL TAKE ATLEAST A FEW MINUTES')
        running_conn_sum = np.sum(w_mask)
        while(running_conn_sum<num_total_connections):

            for j in range(0, num_neurons_total):

                # print('TIE OUT NEURON NEURON: ' + str(j) + '_____running num of conns: '+str(np.sum(w_mask)))
                connections_left = int(num_total_connections-np.sum(w_mask))
                if connections_left==0:
                    break

                in_degs, out_degs, total_degs = self.get_neuron_degs(w_mask=w_mask)
                deg_probs = self.compute_degree_based_probability(total_degs)
                conn_idx = []
                conn_idx.extend(np.where(w_mask[:, j] == 1.0)[0].tolist())

                if connections_left>1:
                    for kkk in range(0, 1):
                        neuron_conn_status = np.clip(np.ceil(np.subtract(deg_probs, np.random.random(size=len(deg_probs)))),0,1)
                        conn_idx.extend(np.where(neuron_conn_status == 1)[0])
                        conn_idx = np.unique(np.asarray(conn_idx)).tolist()

                elif connections_left==1:
                    for kkk in range(0, 20):
                        neuron_conn_status = np.clip(np.ceil(np.subtract(deg_probs, np.random.random(size=len(deg_probs)))),0,1)
                        conn_idx.extend(np.where(neuron_conn_status == 1)[0])
                        conn_idx = np.unique(np.asarray(conn_idx)).tolist()

                if len(conn_idx)<=connections_left:
                    conn_idx = conn_idx
                else:
                    conn_idx = conn_idx[0:connections_left]

                w_mask[conn_idx, j] = 1.0

            running_conn_sum = np.sum(w_mask)

        in_degs, out_degs, total_degs = self.get_neuron_degs(w_mask=w_mask)
        assert (len(np.where(total_degs == 0)[0]) == 0)

        return w_mask


    def create_mask(self,reservoir_size,input_size,output_size,starter_neuron_num,reservoir_connection_density=0.1,input_output_connection_density=0.1):

        '''

        :param reservoir_size: NUMBER OF NEURONS IN RECURRENT NETWORK (N)
        :param input_size: NUMBER OF NEURONS IN INPUT LAYER
        :param output_size: NUMBER OF NEURONS IN OUTPUT LAYER
        :param starter_neuron_num: RECURRENT NETWORK SEED NUMBER OF NEURONS FOR RICH CLUB
        :param reservoir_connection_density: RECURRENT NETWORK CONNECITON DENSITY 
        :param input_output_connection_density: CONNECTION DENSITY BETWEEN INPUT LAYER AND RECURRENT NETWORK

        :param 
        :return: MATRIX OF SHAPE (N,N) WITH {1,0}, MASK FOR NETWORK CONNECTIONS
        '''

        assert (reservoir_size > 0)
        assert (input_size > 0)
        assert (output_size > 0)

        # GENERATE RESERVOIR BLOCK - NON-ZERO
        num_of_post_conns = int(reservoir_connection_density*reservoir_size)

        block_0_0 = self.SF_reservoir_mask_generator(num_neurons_total=reservoir_size, out_conn_num=num_of_post_conns, starter_neuron_num=starter_neuron_num+1)

        # GENERATE BLOCK 0_1 RESERVOIR INPUT - NON-ZERO
        block_0_1_rows = np.random.randint(0, reservoir_size,
                                           size=int(reservoir_size * input_size * input_output_connection_density))
        block_0_1_cols = np.random.randint(0, input_size,
                                           size=int(reservoir_size * input_size * input_output_connection_density))

        block_0_1 = np.zeros((reservoir_size, input_size), dtype=np.float32)
        block_0_1[block_0_1_rows, block_0_1_cols] = 1.0

        # GENERATE BLOCK 2_0 OUTPUT RESERVOIR - NON-ZERO
        block_2_0_rows = np.random.randint(0, output_size,
                                           size=int(output_size * reservoir_size * input_output_connection_density))
        block_2_0_cols = np.random.randint(0, reservoir_size,
                                           size=int(output_size * reservoir_size * input_output_connection_density))

        block_2_0 = np.zeros((output_size, reservoir_size), dtype=np.float32)
        block_2_0[block_2_0_rows, block_2_0_cols] = 1.0

        # GENERATE BLOCK 1_0 INPUT RESERVOIR - ZERO
        block_1_0 = np.zeros((input_size, reservoir_size), dtype=np.float32)

        # GENERATE BLOCK 1_1 INPUT INPUT - ZERO
        block_1_1 = np.zeros((input_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 0_2 RESERVOIR OUTPUT - ZERO
        block_0_2 = np.zeros((reservoir_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 1_2 INPUT OUTPUT - ZERO
        block_1_2 = np.zeros((input_size, output_size), dtype=np.float32)

        # GENERATE BLOCK 2_1 OUTPUT INPUT - ZERO
        block_2_1 = np.zeros((output_size, input_size), dtype=np.float32)

        # GENERATE BLOCK 2_2 OUTPUT OUTPUT - ZERO
        block_2_2 = np.zeros((output_size, output_size), dtype=np.float32)


        block_col_0 = np.concatenate([block_0_0,block_1_0,block_2_0],axis=0)
        block_col_1 = np.concatenate([block_0_1, block_1_1, block_2_1], axis=0)
        block_col_2 = np.concatenate([block_0_2, block_1_2, block_2_2], axis=0)

        mask = np.concatenate([block_col_0,block_col_1,block_col_2],axis=1)

        return mask


    def create_weight_matrix(self,reservoir_size,input_size,output_size, num_res_exc_neurons,num_res_inh_neurons,res_exc_conn_dist, res_inh_conn_dist,input_conn_max,output_conn_max,skew=1.0):

        '''

        :param reservoir_size: NUMBER OF NEURONS IN RECURRENT NETWORK (N)
        :param input_size: NUMBER OF NEURONS IN INPUT LAYER
        :param output_size: NUMBER OF NEURONS IN OUTPUT LAYER
        :param num_res_exc_neurons: NUMBER OF EXCITATORY NEURONS IN RECURRENT NETWORK
        :param num_res_inh_neurons: NUMBER OF INHIBITORY NEURONS IN RECURRENT NETWORK
        :param res_exc_conn_dist: EXCITATORY CONNECTION WEIGHT DISTRIBUTION (MEAN, VARIANCE)
        :param res_inh_conn_dist: INHIBITORY CONNECTION WEIGHT DISTRIBUTION -(MEAN, VARIANCE)
        :param input_conn_max: MAXIMUM INPUT CONNECTION WEIGHT, UNIFORM DISTRIBUTION
        :param output_conn_max: MAXIMUM OUTPUT CONNECTION WEIGHT, UNIFORM DISTRIBUTION
        :param skew: 1.0 CONSTANT

        :return: WEIGHT MATRIX OF SHAPE (N,N)
        '''

        assert (reservoir_size > 0)
        assert (input_size > 0)
        assert (output_size > 0)

        # GENERATE RESERVOIR BLOCK - NON-ZERO - lognormal distributions
        block_0_0 = np.concatenate([np.random.lognormal(res_exc_conn_dist[0],res_exc_conn_dist[1],size=(reservoir_size,num_res_exc_neurons)),-skew*np.random.lognormal(res_inh_conn_dist[0],res_inh_conn_dist[1],size=(reservoir_size,num_res_inh_neurons))],axis=1)

        # GENERATE BLOCK 0_1 RESERVOIR INPUT - NON-ZERO
        block_0_1 = input_conn_max*np.random.random((reservoir_size, input_size))

        # GENERATE BLOCK 2_0 OUTPUT RESERVOIR - NON-ZERO
        block_2_0 = output_conn_max*np.random.random((output_size, reservoir_size))

        # GENERATE BLOCK 1_0 INPUT RESERVOIR - ZERO
        block_1_0 = np.zeros((input_size, reservoir_size))

        # GENERATE BLOCK 1_1 INPUT INPUT - ZERO
        block_1_1 = np.zeros((input_size, input_size))

        # GENERATE BLOCK 0_2 RESERVOIR OUTPUT - ZERO
        block_0_2 = np.zeros((reservoir_size, output_size))

        # GENERATE BLOCK 1_2 INPUT OUTPUT - ZERO
        block_1_2 = np.zeros((input_size, output_size))

        # GENERATE BLOCK 2_1 OUTPUT INPUT - ZERO
        block_2_1 = np.zeros((output_size, input_size))

        # GENERATE BLOCK 2_2 OUTPUT OUTPUT - ZERO
        block_2_2 = np.zeros((output_size, output_size))

        block_col_0 = np.concatenate([block_0_0, block_1_0, block_2_0], axis=0)
        block_col_1 = np.concatenate([block_0_1, block_1_1, block_2_1], axis=0)
        block_col_2 = np.concatenate([block_0_2, block_1_2, block_2_2], axis=0)

        weights = np.concatenate([block_col_0, block_col_1, block_col_2], axis=1).astype(np.float32)

        return weights


    def create_network_v1(self,net_num,a,b,c,d,w,mask,neuron_ranges_names, neuron_ranges_inds):

        '''

        :param net_num: NETWORK NUMBER
        :param a: IZHIKEVICH NEURON PARAMETER VECTOR
        :param b: IZHIKEVICH NEURON PARAMETER VECTOR
        :param c: IZHIKEVICH NEURON PARAMETER VECTOR
        :param d: IZHIKEVICH NEURON PARAMETER VECTOR
        :param w: WEIGHT MATRIX
        :param mask: CONNECTION MASK MATRIX
        :param neuron_ranges_names: NEURON RANGES LIST NAMES
        :param neuron_ranges_inds: NEURON RANGES LIST

        :return: SAVED FILE FOR NETWORK
        '''

        net = cin.izhikevich_network()

        names,data = net.initialize_network_parameters_v1(a,b,c,d,w,mask)

        net_name = 'Network_'+str(net_num)
        sup.save_tf_nontf_data(names=names,data=data,names_nontf=neuron_ranges_names,data_nontf=neuron_ranges_inds,filename=net_name,savePath=self.network_Path)

        print('Saved '+str(net_name)+' to '+str(self.network_Path))


    def build_one_network_w_IE_neurons_v1(self, net_num, starter_neuron_num,num_neurons_in_res=1000,
                                          num_input_neurons=100,
                                          num_output_neurons=100,
                                          res_con_density=0.1,
                                          input_output_conn_density=1.0,
                                          res_exc_w_dist=[1.0,1.0],
                                          res_inh_w_dist=[-1.0,1.0],
                                          w_dist_skew = 1.0,
                                          inp_w_max=5.0,
                                          out_w_max=0.5,
                                          num_res_exc_neurons=800,
                                          num_res_inh_neurons=200,
                                          a_vec=[0.02, 0.1, 0.02, 0.02], b_vec=[0.2, 0.2, 0.2, 0.2],
                                          c_vec=[-65.0, -65.0, -65.0, -65.0], d_vec=[8.0, 2.0, 8.0, 8.0]):

        a = np.concatenate([a_vec[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                            a_vec[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                            a_vec[2] * np.ones(num_input_neurons, dtype=np.float32),
                            a_vec[3] * np.ones(num_output_neurons, dtype=np.float32)])

        b = np.concatenate([b_vec[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                            b_vec[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                            b_vec[2] * np.ones(num_input_neurons, dtype=np.float32),
                            b_vec[3] * np.ones(num_output_neurons, dtype=np.float32)])

        c = np.concatenate([c_vec[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                            c_vec[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                            c_vec[2] * np.ones(num_input_neurons, dtype=np.float32),
                            c_vec[3] * np.ones(num_output_neurons, dtype=np.float32)])
        d = np.concatenate([d_vec[0] * np.ones(num_res_exc_neurons, dtype=np.float32),
                            d_vec[1] * np.ones(num_res_inh_neurons, dtype=np.float32),
                            d_vec[2] * np.ones(num_input_neurons, dtype=np.float32),
                            d_vec[3] * np.ones(num_output_neurons, dtype=np.float32)])

        mask = self.create_mask(reservoir_size=num_neurons_in_res, input_size=num_input_neurons,
                              output_size=num_output_neurons, reservoir_connection_density=res_con_density,
                              input_output_connection_density=input_output_conn_density,starter_neuron_num=starter_neuron_num)

        weights = self.create_weight_matrix(num_neurons_in_res, num_input_neurons, num_output_neurons,num_res_exc_neurons=num_res_exc_neurons,num_res_inh_neurons=num_res_inh_neurons,
                                          res_exc_conn_dist=res_exc_w_dist, res_inh_conn_dist=res_inh_w_dist,
                                          input_conn_max=inp_w_max,
                                          output_conn_max=out_w_max,
                                             skew=w_dist_skew)

        neuron_ranges_names, neuron_ranges_inds = self.get_neuron_func_ranges(res_size=num_neurons_in_res,
                                                                            input_size=num_input_neurons,
                                                                            output_size=num_output_neurons)


        self.create_network_v1(net_num, a, b, c, d, weights, mask, neuron_ranges_names, neuron_ranges_inds)

        log_filename = 'Network_' + str(net_num) + '_Log.txt'
        log_fn = os.path.abspath(os.path.join(self.network_Path, log_filename))
        with open(log_fn, 'w') as f:
            f.write('LOG___NETWORK_' + str(net_num) + '\n\n')

            f.write('NETWORK STATS:\n\n')
            f.write('   num_neurons_in_res:     ' + str(num_neurons_in_res) + '\n')
            f.write('   num_input_neurons:     ' + str(num_input_neurons) + '\n')
            f.write('   num_output_neurons:     ' + str(num_output_neurons) + '\n')
            f.write('   res_con_density:     ' + str(res_con_density) + '\n')
            f.write('   input_output_conn_density:     ' + str(input_output_conn_density) + '\n')
            f.write('   res_exc_w_dist:     ' + str(res_exc_w_dist) + '\n')
            f.write('   res_inh_w_dist:     ' + str(res_inh_w_dist) + '\n')
            f.write('   inp_w_max:     ' + str(inp_w_max) + '\n')
            f.write('   out_w_max:     ' + str(out_w_max) + '\n')
            f.write('   num_res_exc_neurons:     ' + str(num_res_exc_neurons) + '\n')
            f.write('   num_res_inh_neurons:     ' + str(num_res_inh_neurons) + '\n')
            f.write('   a_vec:     ' + str(a_vec) + '\n')
            f.write('   b_vec:     ' + str(b_vec) + '\n')
            f.write('   c_vec:     ' + str(c_vec) + '\n')
            f.write('   d_vec:     ' + str(d_vec) + '\n')



##### RUN CODE #####
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["net_num="])


    except getopt.GetoptError:
        print('Incorrect number of arguments')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--net_num':
            NET_NUM = int(arg)

        else:
            print('Error occured...')
            sys.exit()

    # INITIAL WEIGHT DISTRIBUTION LOGNORMAL MEAN AND VARIANCE RANGES ENSURING SUB-CRITIAL NETWORK INITIALIZATION
    log_average_ranges = [-2.5, -2.0]
    log_var_ranges = [0.1, 1.1]

    initialization_not_satisfied = True
    while initialization_not_satisfied:

        log_ave_w = log_average_ranges[0] + np.absolute(
            (log_average_ranges[1] - log_average_ranges[0])) * np.random.random()
        log_var_w = log_var_ranges[0] + np.absolute(
            (log_var_ranges[1] - log_var_ranges[0])) * np.random.random()

        if log_ave_w<=log_average_ranges[1] and log_ave_w>=log_average_ranges[0] and log_var_w<=log_var_ranges[1] and log_var_w>=log_var_ranges[0]:
            initialization_not_satisfied = False

    bn = SFRC_network_builder()
    bn.build_one_network_w_IE_neurons_v1(net_num=NET_NUM, res_exc_w_dist=[log_ave_w, log_var_w], res_inh_w_dist=[log_ave_w, log_var_w],
                                         w_dist_skew=1.0, inp_w_max=3.35, out_w_max=3.35, res_con_density=0.12,
                                         input_output_conn_density=1.0,starter_neuron_num=1)



if __name__ == '__main__':
    main(sys.argv[1:])

