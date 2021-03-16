import sys, getopt
import numpy as np
import os
import time
import multiprocessing as mp
import NAN_support_lib as sup



class multi_process_bf:
    def __init__(self,num_processes):

        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/'

        print('Main working directory: ' + str(self.main_Path))
        print('Networks directory: ' + str(self.network_Path))
        print('Data directory: ' + str(self.data_Path))

        # Define IPC managers
        manager1 = mp.Manager()

        # Define lists (queue) for tasks and computation results
        self.data_feed1 = manager1.Queue()

        self.status1 = manager1.Queue()

        self.processes1 = []

        for i in range(num_processes):
            # Set process name
            process_name = 'Pb%i' % i

            # Create the process, and connect it to the worker function
            new_process = mp.Process(target=self.processInterval, args=(process_name, self.data_feed1, self.status1))

            # Add new process to the list of processes
            self.processes1.append(new_process)

            # Start the process
            new_process.start()


    def process_interval(self,signal, data_filename, network_filename,window, offset, save_filename):

        '''
            LOADS DATA FOR PROCESSING INTO POOL QUEUE
        '''

        self.data_feed1.put([signal, data_filename, network_filename, window, offset, save_filename])


    def processInterval(self, process_name, data_feed, status):

        '''
        
        :param process_name: NUMBER OF PROCESS/POOL WORKER
        :param data_feed: LIST OF [
                                    signal
                                    , data_filename
                                    , network_filename
                                    , window
                                    , offset
                                    , save_filename
                                    ]
        :param status: LIST OF BF FILES PROCESSED AND SAVED
        :return: 
        '''

        '''
                    data_feed data format
                     process_command, interval_sec_START, interval_sec_END, t_ms_window,w

        '''

        print('[%s] AFF process launched, waiting for data' % process_name)

        while True:
            data = data_feed.get()

            # Ft = data[1]
            data_filename = data[1]
            network_filename = data[2]
            window = data[3]
            offset = data[4]
            save_filename = data[5]


            if data[0] == -1:
                print('[%s] AFF process terminated' % process_name)
                status.put(1)
                break
            elif data[0] == 0:
                print('[%s] AFF process completed round, ready for next batch ' % process_name)
                status.put(-1)
            else:

                print('OPENING_' + str(data_filename) + '_w_'+str(network_filename))
                time_series, net_data = self.open_data(data_filename, network_filename)

                st = time.time()
                names_bf, data_bf = self.compute_branching_factors(time_series_data=time_series, net_data=net_data,
                                                                   window=window, offset=offset)
                print('BF_COMPUTED_IN_' + str(time.time() - st))
                sup.save_non_tf_data(names_bf, data_bf, save_filename, self.data_Path)
                print('DATA_SAVED_to_'+str(save_filename))

                status.put(save_filename)


    def convert_spike_data_into_dense_matrix_form(self, spike_inds, spike_matrix_shape):

        '''
        
        :param spike_inds: LIST OF SPIKE INDICES (NEURON NUM, SPIKE TIME) 
        :param spike_matrix_shape: (NEURON NUM, 1000)
        
        :return: BINARY MATRIX WITH SHAPE (NEURON NUM, 1000)
        '''

        out = np.zeros((spike_matrix_shape[0], spike_matrix_shape[1]))
        out[spike_inds[:, 0], spike_inds[:, 1]] = 1
        return out


    def preprocess_spike_slices(self, compressed_spike_slices, spike_matrix_shape):

        '''
        
        :param compressed_spike_slices: LIST OF LISTS OF SPIKE INDICES (NEURON NUM, SPIKE TIME) 
        :param spike_matrix_shape: (NEURON NUM, 1000)
        
        :return: BINARY MATRIX WITH SHAPE (NEURON NUM, SIMULATION FILE ms RUNTIME)
        '''

        for i in range(0, len(compressed_spike_slices)):
            if i == 0:
                spike_slice = self.convert_spike_data_into_dense_matrix_form(compressed_spike_slices[i],
                                                                             spike_matrix_shape)
                single_spike_matrix = spike_slice
            else:
                spike_slice = self.convert_spike_data_into_dense_matrix_form(compressed_spike_slices[i],
                                                                             spike_matrix_shape)
                single_spike_matrix = np.concatenate([single_spike_matrix, spike_slice], axis=1)

        return single_spike_matrix


    def open_data(self, data_filename, network_filename):

        '''
            OPENS SIMULATION DATAFILE AND NETWORK DATAFILE
        '''
        print('open data path')
        print(self.data_Path)

        # UNPACK SPIKE DATA
        names, data = sup.unpack_file(data_filename, dataPath=self.data_Path)

        # UNPACK NETWORK
        names_net, data_net = sup.unpack_file(network_filename, dataPath=self.network_Path)

        return [names, data], [names_net, data_net]


    def compute_time_iteration_range(self, window, offset, all_t):

        '''
        
        :param window: ms WINDOW SIZE FOR COUNTING SPIKES BEFORE AND AFTER NEURON SPIKES AT SOME t ms
        :param offset: ALWAYS KEPT CONSTANT AT 0 ms
        :param all_t: MAXIMUM T ms
        
        :return: ms TIME RANGES FOR INCLUDING SPIKES BEFORE AND AFTER NEURON SPIKES
        '''

        margin = window + offset

        forward_ranges = [1 + offset, 1 + margin]
        backward_ranges = [-margin, -offset]
        t_ranges = [margin, all_t - margin]

        return t_ranges, forward_ranges, backward_ranges


    def compute_branching_factors(self, time_series_data, net_data, window, offset,
                                  data_save_ms_dur=1000):

        '''


        :param time_series_data: SPIKE DATA CONTAINING NEURON AND TIME OF EACH SPIKE
        :param net_data: DATA FROM NETWORK FILE
        :param cust_masks:  list of binary 0,1 arrays to select neurons forming popoulation for which compute BF 
        :param window: ms WINDOW SIZE FOR COUNTING SPIKES BEFORE AND AFTER NEURON SPIKES AT SOME t ms
        :param offset: ALWAYS KEPT CONSTANT AT 0 ms
        
        :return: BRANCHING FACTOR OVER TIME AT ms RESOLUTION AND AVERAGED OVER EACH second
        '''

        cust_masks = [np.concatenate([np.ones(800, dtype=np.float32),
                                    np.ones(200, dtype=np.float32),
                                    np.zeros(100, dtype=np.float32),
                                    np.zeros(100, dtype=np.float32)])]


        cust_inds = []
        for m in range(0, len(cust_masks)):
            cust_inds.append(np.where(cust_masks[m] == 1.0)[0])


        data_t = time_series_data[1][time_series_data[0].index('time_sec_spikes')]

        names_net = net_data[0]
        data_net = net_data[1]

        # EXTRACT CONNECTION MASK FOR NETWORK
        # AXIS 0 IS POSTSYNAPTIC NEURONS, AXIS 1 IS PRESYNAPTIC INPUTS
        W_mask = data_net[names_net.index('W_mask')]
        A = data_net[names_net.index('A')]
        num_neurons = len(A)


        st = time.time()
        # GLUE ALL 1 SEC TIME SLICES HORIZONTALLY
        all_spikes = self.preprocess_spike_slices(compressed_spike_slices=data_t,
                                                  spike_matrix_shape=[num_neurons, data_save_ms_dur])
        print('Assembled data for processing in ' + str(time.time() - st) + ' sec')

        # COMPUTE RANGES
        t_inds, forward_inds, backward_inds = self.compute_time_iteration_range(window, offset,
                                                                                np.shape(all_spikes)[1])

        network_BF_per_t_list = []
        t_series_list = []
        t_sec = []
        temp_BF_agg = []
        BF_ave_per_sec_list = []
        for i in range(0, len(cust_masks)):
            network_BF_per_t_list.append([])
            t_series_list.append([])
            temp_BF_agg.append([])
            BF_ave_per_sec_list.append([])
            t_sec.append([])

        # ITERATE THROUGH SPIKE DATA AND COMPUTE BRANCHING 1.OUT 2.IN COMPONENTS FOR EACH TIMESTEP FOR EACH NEURON
        for t in range(t_inds[0], t_inds[1]):
            # GET FORWARD WINDOW SPIKES
            forward_spike_union_at_t = np.clip(
                np.sum(all_spikes[:, t + forward_inds[0]:t + forward_inds[1]], axis=1), 0, 1)
            # GET BACKWARD WINDOW SPIKES
            backward_spike_union_at_t = np.clip(
                np.sum(all_spikes[:, t + backward_inds[0]:t + backward_inds[1]], axis=1), 0, 1)
            # GET CURRENT T SPIKES
            spikes_at_t = all_spikes[:, t]
            # COMPUTE BRANCHING OUT VALUES, NEED TO TRANSPOSE W_mask FOR THIS CASE
            unfiltered_branch_out_values = np.matmul(np.transpose(W_mask), forward_spike_union_at_t)
            filtered_branch_out_values = np.multiply(unfiltered_branch_out_values, spikes_at_t)
            # COMPUTE BRANCHING IN VALUES
            unfiltered_branch_in_values = np.matmul(W_mask, backward_spike_union_at_t)
            filtered_branch_in_values = np.multiply(unfiltered_branch_in_values, spikes_at_t)

            ## APPLY CUSTOM FILTER TO SELECT BRANCHING COMPUTATION FOR SELECT NEURONAL GROUPS
            for m in range(0, len(cust_masks)):

                record_bf = False

                b_out_sum = np.sum(filtered_branch_out_values[cust_inds[m]])
                b_in_sum = np.sum(filtered_branch_in_values[cust_inds[m]])

                if b_out_sum == 0 and b_in_sum == 0:
                    network_branching_factor_at_t = -1.0
                elif b_out_sum > 0 and b_in_sum == 0:
                    network_branching_factor_at_t = -2.0
                    print('ERROR: GOT NON-ZERO BRANCH OUT WITH ZERO BRANCH IN....')
                else:
                    # SUM BRANCHING OUTS AND INS OF ALL NEURONS AND COMPUTE NETWORK BRANCHING FACTOR
                    network_branching_factor_at_t = np.divide(b_out_sum, b_in_sum)
                    record_bf = True

                # APPEND TO TIME SERIES
                if np.sum(spikes_at_t[cust_inds[m]]) > 0 and record_bf == True:
                    network_BF_per_t_list[m].append(network_branching_factor_at_t)
                    t_series_list[m].append(t)

                # RECORD FOR PER SECOND AVERAGED DATA
                if t % 1000 == 0:
                    BF_ave_per_sec_list[m].append(np.average(temp_BF_agg[m]))
                    temp_BF_agg[m] = []
                    t_sec[m].append(t / 1000)
                    if np.sum(spikes_at_t[cust_inds[m]]) > 0 and record_bf == True:
                        temp_BF_agg[m].append(network_branching_factor_at_t)

                elif t == t_inds[1] - 1:
                    BF_ave_per_sec_list[m].append(np.average(temp_BF_agg[m]))
                    t_sec[m].append(np.ceil(t / 1000))
                else:
                    if np.sum(spikes_at_t[cust_inds[m]]) > 0 and record_bf == True:
                        temp_BF_agg[m].append(network_branching_factor_at_t)

        return ['network_BF_per_t_list', 't_series_list', 'BF_ave_per_sec_list', 't_sec', 'cust_masks'], [network_BF_per_t_list, t_series_list, BF_ave_per_sec_list,
                            t_sec, cust_masks]


    def kill_workers(self, process_count):

        '''
            SHUTS OFF ALL POOL WORKERS
        '''

        for i in range(0, process_count):
            print('KILL SWITCH SENT FOR PROCESS ' + str(i))
            self.process_interval(signal=-1,data_filename='', network_filename='',window=1, offset=1, save_filename='')

        list_of_save_files = []
        sum1 = 0
        while sum1 != process_count:
            temp = self.status1.get()
            if type(temp) == int:

                sum1 = sum1 + temp
            elif type(temp)==str:
                list_of_save_files.append(temp)
            else:
                print('Found other stuff in queue....check')

        return list_of_save_files


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "net_num=", "sim_run_time_s=", "start_sp=", "end_sp=", "pool_size="])
    except getopt.GetoptError:
        print('Incorrect arguments')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ver_num':
            ver = int(arg)

        elif opt == '--net_num':
            net = int(arg)

        elif opt == '--sim_run_time_s':
            times_end = int(arg)
            times_strt = 0

        elif opt == '--start_sp':
            start_sp = int(arg)

        elif opt == '--end_sp':
            end_sp = int(arg)

        elif opt == '--pool_size':
            num_processes = int(arg)

        else:
            print('Error, exiting')
            sys.exit()


    # delta_t parameter for branching factor algorithm = 9ms
    window = 9
    # offset t parameter for branching factor algorithm = 0ms
    offset = 0

    sp_arr = start_sp+np.arange(end_sp-start_sp)

    print('SIM_VER: ' + str(ver))
    print('NETWORK_NUMBER: ' + str(net))
    print('FULL_SIMULATION_TIME_RANGE: ' + str(times_strt)+'_' + str(times_end))
    print('SP_RANGE: ' + str(start_sp)+'_' + str(end_sp))
    print('POOL_SIZE: ' + str(num_processes))

    print('Fixed parameters: ')
    print('delta_t: ' + str(window)+'ms')
    print('t_offset: ' + str(offset)+'ms')



    bf = multi_process_bf(num_processes=num_processes)
    print('All pool workers launched')


    for j in range(0,len(sp_arr)):
        # FILE NAMES
        data_filename = 'ver_'+str(ver)+'/Sim_Data_for_Net_' + str(net) + '_ver_' + str(ver) + '_' + str(times_strt) + str(times_end)+'_sp_'+str(sp_arr[j])
        network_filename = 'Network_' + str(net)
        save_filename = 'ver_'+str(ver)+'/BF_Data_for_Net_' + str(net) + '_ver_' + str(ver) + '_' + str(times_strt) + str(times_end)+'_DT_'+str(window)+'_sp_'+str(sp_arr[j])


        bf.process_interval(signal=1, data_filename=data_filename,network_filename=network_filename,window=window,offset=offset,save_filename=save_filename)


    saved_filenames = bf.kill_workers(process_count=num_processes)

    print('Files Saved:')
    for i in range(0,len(saved_filenames)):
        print(str(i)+'. ' + saved_filenames[i])


if __name__ == '__main__':
    main(sys.argv[1:])
