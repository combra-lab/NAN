import sys, getopt
import numpy as np
import os
import time
import multiprocessing as mp
import powerlaw as pl
import NAN_support_lib as sup


class multi_process:
    def __init__(self,num_processes, cont_Ft, n_by_spike_time, n_inputs):

        # Define IPC managers
        manager1 = mp.Manager()

        # Define lists (queue) for tasks and computation results
        self.data_feed1 = manager1.Queue()

        self.status1 = manager1.Queue()

        self.spike_to_avalanche_dict = manager1.dict()

        self.processes1 = []

        for i in range(num_processes):
            # Set process name
            process_name = 'Pb%i' % i

            # Create the process, and connect it to the worker function
            new_process = mp.Process(target=self.process_input_for_avalanches, args=(process_name, cont_Ft, n_by_spike_time, n_inputs, self.data_feed1, self.status1))

            # Add new process to the list of processes
            self.processes1.append(new_process)

            # Start the process
            new_process.start()


    def add_inputs_to_parallel_feed(self,signal, interval_ms_START, current_time_ms, neuron_processed):

        self.data_feed1.put([signal, interval_ms_START, current_time_ms, neuron_processed])



    def process_input_for_avalanches(self, process_name, cont_Ft, n_by_spike_time, n_inputs, data_feed, status):

        '''
        PROCESSES SPIKE DATA TO ASSEMBLE AVALANCHE
                    
            data_feed data format
             process_command, interval_sec_START, interval_sec_END, t_ms_window,w

        '''

        print('[%s] Avalanche processor launched, waiting for data' % process_name)

        while True:
            data = data_feed.get()

            if data[0] == -1:
                print('[%s] Avalanche process terminated' % process_name)
                status.put(1)
                break
            elif data[0] == 0:
                print('[%s] Avalanche process completed round, ready for next batch ' % process_name)
                status.put(-1)
            else:

                interval_ms_START = data[1]
                current_time_ms = data[2]
                post_neuron = data[3]

                tt1=time.time()

                idx_spike_strt = interval_ms_START-cont_Ft[0]
                idx_spike_end = current_time_ms - cont_Ft[0]

                if idx_spike_strt<0:
                    idx_spike_strt=0

                interval_times_ms = cont_Ft[idx_spike_strt:idx_spike_end]

                unfiltered_spiked_input_neurons_list = n_by_spike_time[idx_spike_strt:idx_spike_end]

                inputs_for_current_post_neuron = n_inputs[post_neuron]

                time_neuron_key_list = []
                for i in range(0,len(unfiltered_spiked_input_neurons_list)):
                    filtered_temp = np.intersect1d(unfiltered_spiked_input_neurons_list[i],inputs_for_current_post_neuron)
                    time_neuron_key_list.extend(1000000*interval_times_ms[i]+filtered_temp)

                if len(time_neuron_key_list)>0:
                    avalanche_candidates=[]
                    for j in range(0,len(time_neuron_key_list)):
                        avalanche_candidates.extend(self.spike_to_avalanche_dict[str(time_neuron_key_list[j])])

                    final_avalanche_set = np.unique(np.asarray(avalanche_candidates)).tolist()

                    self.spike_to_avalanche_dict[str(1000000*current_time_ms+post_neuron)] = final_avalanche_set
                    status.put(0)

                elif len(time_neuron_key_list)==0:
                    status.put(str(1000000*current_time_ms+post_neuron))



class compute_avalanche_distribution:
    def __init__(self,ver):

        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/'


        self.status_of_comp = True


    def assemble_spike_data_for_interval(self,filename,interval_sec=[0,-1]):

        '''
        EXTRACTS SPIKE DATA FROM SAVED SIMULATION FILES
        '''

        names, data = sup.unpack_file(filename=filename, dataPath=self.data_Path)

        if interval_sec == [0, -1]:
            interval_sec = [0, len(data[0])]

        ### ASSEMBLE DATA
        for i in range(interval_sec[0], interval_sec[1]):
            inds_t_n = data[1][i]
            times = inds_t_n[:, 1]
            neurons = inds_t_n[:, 0]

            if i == interval_sec[0]:
                Ft = (i * 1000) + times.reshape((1, -1))
                Fn = neurons.reshape((1, -1))
            else:
                Ft = np.concatenate([Ft, (i * 1000) + times.reshape((1, -1))], axis=1)
                Fn = np.concatenate([Fn, neurons.reshape((1, -1))], axis=1)

        print('Spike Data Assembled for Interval '+str(interval_sec[0])+'_-_'+str(interval_sec[1]))

        return np.squeeze(Ft), np.squeeze(Fn)

    def str_splitter_into_intervals(self,str):

        idx_split = str.find('_')
        int1 = int(str[0:idx_split])
        int2 = int(str[idx_split+1:len(str)])

        return [int1,int2]

    def filter_spike_series_for_neurons(self, Ft, Fn, neuron_pop):

        '''
        FILTERS OUT SPIKES FOR GIVEN NEURONAL POPULATION
        '''

        if neuron_pop == 'RES':
            neuron_interval = [0, 1000]
        elif neuron_pop == 'EXC':
            neuron_interval = [0, 800]
        elif neuron_pop == 'INH':
            neuron_interval = [800, 1000]
        elif neuron_pop == 'ALL':
            neuron_interval = [np.amin(Fn), np.amax(Fn) + 1]
            print('Number of Unique Neurons Found = ' + str(neuron_interval[1]))
        else:
            neuron_interval = self.str_splitter_into_intervals(neuron_pop)


        n_idx = np.intersect1d(np.where(np.squeeze(Fn)>=neuron_interval[0])[0],np.where(np.squeeze(Fn)<neuron_interval[1])[0])

        fFt = np.squeeze(Ft)[n_idx]
        fFn = np.squeeze(Fn)[n_idx]

        print('Spike Data Filtered for Neuron Population: ' + str(neuron_pop))

        sort_idx = np.argsort(fFt)

        sFt = fFt[sort_idx]
        sFn = fFn[sort_idx]

        print('Spike Data Sorted by Time')

        return sFt,sFn

    def assemble_neurons_by_spike_time(self,Ft,Fn,starting_time_ms):

        '''
        
        :param Ft: TIME DATA OF EACH SPIKE, LIST OF 100 LISTS EACH CONTAINING SPIKE TIME DATA FOR 1 SECOND OF SIMULATION TIME
        :param Fn: NEURON DATA OF EACH SPIKE, LIST OF 100 LISTS EACH CONTAINING SPIKE NEURON DATA FOR 1 SECOND OF SIMULATION TIME
        :param starting_time_ms: STARTING TIME OF DATA
        
        :return: LIST OF NONREPEATING ORDERED TIMES IN WHICH SPIKES OCCURED OVER ALL SPIKE DATA (cont_Ft) 
                AND LIST WITH INDEX CORRESPONDING TO (cont_Ft) WITH A LIST OF NEURONS THAT FIRED AT EACH TIME STEP 
        '''

        print('starting_time_ms',starting_time_ms)

        new_Ft = []
        new_Fn = []
        cur_time = starting_time_ms
        time_t_spiked_neuron_list = []
        for t in range(0,len(Ft)):

            if cur_time>Ft[t]:
                print('Error Ft not sorted, got less than current time at t:' +str(t))
                print('current time: '+str(cur_time))
                print('Ft[t]: '+str(Ft[t]))
                sys.exit()
            else:
                if cur_time==Ft[t]:
                    time_t_spiked_neuron_list.append(Fn[t])
                else:


                    new_Ft.append(cur_time)
                    new_Fn.append(time_t_spiked_neuron_list)

                    if cur_time + 1 != Ft[t]:
                        for i in range(cur_time+1,Ft[t]):
                            new_Ft.append(i)
                            new_Fn.append([])

                    cur_time = Ft[t]
                    time_t_spiked_neuron_list = []
                    time_t_spiked_neuron_list.append(Fn[t])

        new_Ft.append(cur_time)
        new_Fn.append(time_t_spiked_neuron_list)

        print('Assembled neurons by spike time')

        return new_Ft,new_Fn

    def assemble_neurons_by_postsyn_neuron(self,filename,neuron_pop):

        '''
        ASSEMBLES LIST OF INPUT NEURONS FOR EACH NEURON INTO A LIST
        '''

        names, data = sup.unpack_file(filename=filename, dataPath=self.network_Path)

        W_mask = data[names.index('W_mask')]

        if neuron_pop == 'RES':
            neuron_interval = [0, 1000]
        elif neuron_pop == 'EXC':
            neuron_interval = [0, 800]
        elif neuron_pop == 'INH':
            neuron_interval = [800, 1000]
        elif neuron_pop == 'ALL':
            neuron_interval = [0, np.shape(W_mask)[1]]
            print('Number of Unique Neurons Found = ' + str(neuron_interval[1]))
        else:
            neuron_interval = self.str_splitter_into_intervals(neuron_pop)


        list_of_inputs = []
        for i in range(0,np.shape(W_mask)[1]):
            input_neurons = np.where(W_mask[i,neuron_interval[0]:neuron_interval[1]]==1)[0]
            list_of_inputs.append(input_neurons)

        print('Assembled neurons by postsyn neuron')

        return list_of_inputs


    def data_loader(self,spike_filename,net_filename,interval_sec,neuron_pop):

        '''
        LOADS AND ASSEMBLES NETWORK AND SIMULATION DATA FOR ASSEMBLING NEURONAL AVALANCHES
        '''

        ## assembles processed spike data
        Ft, Fn = self.assemble_spike_data_for_interval(filename=spike_filename, interval_sec=interval_sec)
        Ft_f, Fn_f = self.filter_spike_series_for_neurons(Ft=Ft, Fn=Fn, neuron_pop=neuron_pop)

        cont_Ft,n_by_spike_time = self.assemble_neurons_by_spike_time(Ft=Ft_f,Fn=Fn_f,starting_time_ms=interval_sec[0]*1000)  ### takes sorted Ft, does not work for unsorted Ft
        n_inputs = self.assemble_neurons_by_postsyn_neuron(filename=net_filename,neuron_pop=neuron_pop)

        return Ft_f, Fn_f,cont_Ft,n_by_spike_time,n_inputs


    def initialize_workers(self,num_processes,cont_Ft, n_by_spike_time, n_inputs):

        self.ss = multi_process(num_processes, cont_Ft=cont_Ft, n_by_spike_time=n_by_spike_time, n_inputs=n_inputs)
        print('All pool workers launched')


    def kill_workers(self,process_count):
        for i in range(0, process_count):
            print('KILL SWITCH SENT FOR PROCESS '+str(i))
            self.ss.add_inputs_to_parallel_feed(signal=-1,interval_ms_START=1,current_time_ms=1,neuron_processed=1)

        sum1 = 0
        while sum1 != process_count:
            temp = self.ss.status1.get()
            if type(temp) == int:

                sum1 = sum1 + temp
            else:
                print('Found other stuff in queue....check')

        self.status_of_comp = False


    def compute_avalanches(self,ver,net,Ft_f,Fn_f,cont_Ft,n_by_spike_time,interval_to_process,neuron_pop,delta_interval=6):

        '''
        
        :param ver: SIMULATION VERION 
        :param net: NETWORK VERSION
        :param Ft_f: TIME DATA OF EACH SPIKE, LIST OF 100 LISTS EACH CONTAINING SPIKE TIME DATA FOR 1 SECOND OF SIMULATION TIME
        :param Fn_f: NEURON DATA OF EACH SPIKE, LIST OF 100 LISTS EACH CONTAINING SPIKE NEURON DATA FOR 1 SECOND OF SIMULATION TIME
        :param cont_Ft: LIST OF NONREPEATING ORDERED TIMES IN WHICH SPIKES OCCURED OVER ALL SPIKE DATA 
        :param n_by_spike_time: INDEX CORRESPONDS TO cont_Ft WITH A LIST OF NEURONS THAT FIRED AT EACH TIME STEP 
        :param interval_to_process: TIME INTERVAL TO PROCESS USED TO PULL OPEN FILE 
        :param neuron_pop: 'RES' THE RECURRENT NETWORK POPULATION, KEPT CONSTANT
        :param delta_interval: 9 ms KEPT CONSTANT
        :return: 
        '''

        log_filename = 'LOG_Avalanche_Distribution_Calculator_net' + str(net) + '_ver_' + str(ver) + '_interval_' + str(
            interval_to_process[0]) + str(interval_to_process[1]) + '_neurons_' + str(neuron_pop) + '.txt'
        log_fn = os.path.abspath(os.path.join(self.data_Path, log_filename))
        with open(log_fn, 'w') as f:
            f.write('LOG_FOR_AVALANCHE_DISTRIBUTION_FOR_NETWORK_' + str(net) + '_VER_' + str(ver) + '\n\n')
            f.write(' PARAMETERS OF COMPUTATION:'+'\n')
            f.write('   interval processed: ' + str(interval_to_process[0]) + '_-_' + str(interval_to_process[1]) + '\n')
            f.write('   delta_interval: ' + str(delta_interval) + '\n')
            f.write('   neuron_pop: ' + str(neuron_pop) + '\n\n')

            avalanche_count = 0

            assert(len(cont_Ft)==len(n_by_spike_time))
            print('len of cont_Ft', len(cont_Ft), cont_Ft[0],cont_Ft[-1])
            print('len of n_by_spike_time', len(n_by_spike_time),n_by_spike_time[0],n_by_spike_time[-1])

            print('Starting Avalanche Distrubtion Computation')
            ### first time step initialize new avalanche for every spike encountered
            if len(n_by_spike_time[0]) > 0:
                for i in range(0,len(n_by_spike_time[0])):
                    self.ss.spike_to_avalanche_dict[str(1000000*cont_Ft[0]+n_by_spike_time[0][i])] = [avalanche_count]
                    avalanche_count = avalanche_count + 1

            print('Completed First Timestep')
            ### after first timestep
            for t in range(1,len(cont_Ft)):
                if t%10000==0:
                    print('On Step ' + str(t))

                ### check if any neurons spiked at this time step (remember since cont_Ft includes all ms in interval, not just in Ft_f)
                if len(n_by_spike_time[t])==0:
                    continue

                for n in range(0,len(n_by_spike_time[t])):
                    start_time_ms = cont_Ft[t]-delta_interval

                    self.ss.add_inputs_to_parallel_feed(signal=1,interval_ms_START=start_time_ms,current_time_ms=cont_Ft[t],neuron_processed=n_by_spike_time[t][n])

                counter = 0
                sum1 = 0
                while sum1 != len(n_by_spike_time[t]):
                    temp = self.ss.status1.get()
                    if type(temp) == int:
                        sum1 = sum1 + 1
                    elif type(temp) == str:
                        self.ss.spike_to_avalanche_dict[temp] = [avalanche_count]
                        avalanche_count = avalanche_count+1
                        sum1 = sum1 + 1

                    counter = counter + 1


            print('Finished Computing Avalanches.')
            print('Avalanches Found____'+str(avalanche_count))

            f.write(' RESULTS:' + '\n')
            f.write('   num of avalanches:' + str(avalanche_count) + '\n')

            print('Assembling avalanche list')


            final_avalanche_dict = dict(self.ss.spike_to_avalanche_dict)

            print('Converted Shared avalanche dict to serialized local version')

            avalanche_list = []
            for i in range(0,len(Ft_f)):
                key = str(1000000*Ft_f[i]+Fn_f[i])
                avalanche_list.extend(self.ss.spike_to_avalanche_dict[key])
                if not np.array_equal(np.asarray(self.ss.spike_to_avalanche_dict[key]),np.asarray(final_avalanche_dict[key])):
                    print('Dictionaries dont agree for key:   '+str(key))
                    print(self.ss.spike_to_avalanche_dict[key])
                    print(final_avalanche_dict[key])

            print('Compiled Avalanche list')


            return avalanche_list

    def powerlaw_fit(self,avalanche_list_in):

        '''
        FITS POWERLAW DISTRIBUTION TO AVALANCHE SIZE DISTRIBUTION
        
        :param avalanche_list_in: ASSEMBLED AVALANCHES SIZES LIST 
        :return: POWERLAW SCALING EXPONENT 
        '''

        XMIN = 1
        XMAX = 1000 # SIZE OF RECURRENT NETWORK - KEPT FIXED FOR ALL SIMULATIONS

        ava_arr = np.asarray(avalanche_list_in)
        x1, y1 = np.unique(ava_arr, return_counts=True)
        fit = pl.Fit(data=y1, xmin=XMIN, xmax=XMAX)

        return fit.alpha




def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "net_num=", "sim_run_time_s=", "start_sp=", "end_sp=", "pool_size="])
    except getopt.GetoptError:
        print('Incorrect number of arguments')
        print('Arguments are: ver, net, times, af_params_list')
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
            print('Something is wrong')
            sys.exit()

    neuron_pop = 'RES'
    window = 9

    print('VERSION: ' + str(ver))
    print('NETWORK: ' + str(net))
    print('SIM_FILE_TIMES_RANGE: ' + str(times_strt)+'_' + str(times_end))
    print('NEURON_POP: ' + str(neuron_pop))
    print('POOL_SIZE: ' + str(num_processes))
    print('DELTA_INTERVAL: ' + str(window))


    INTERVALS = [[15, 25], [40, 50], [65, 75],[89,99]]
    TIME = [20,50,70,95]
    sp_arr = start_sp + np.arange(end_sp - start_sp)

    for j in range(0,len(sp_arr)):
        # FILE NAMES
        spike_name = 'ver_'+str(ver)+'/Sim_Data_for_Net_' + str(net) + '_ver_' + str(ver) + '_' + str(times_strt) + str(times_end)+'_sp_'+str(sp_arr[j])
        net_name = 'Network_' + str(net)
        save_filename = 'ver_'+str(ver)+'/AVA_Data_for_Net_' + str(net) + '_ver_' + str(ver) + '_' + str(times_strt) + str(times_end)+'_DT_'+str(window)+'_sp_'+str(sp_arr[j])

        alpha_l = []
        time_sec_l = []
        for i in range(0,len(INTERVALS)):

            ca = compute_avalanche_distribution(ver=ver)

            INTERVAL_times_strt = INTERVALS[i][0]
            INTERVAL_times_end = INTERVALS[i][1]

            Ft_filtered,Fn_filtered,cont_Ft,n_by_spike_time,n_inputs = ca.data_loader(spike_filename=spike_name,net_filename=net_name,interval_sec=[INTERVAL_times_strt, INTERVAL_times_end],neuron_pop=neuron_pop)

            ca.initialize_workers(num_processes=num_processes,cont_Ft=cont_Ft, n_by_spike_time=n_by_spike_time,n_inputs=n_inputs)

            '''
            ver,net,cont_Ft,n_by_spike_time,orig_interval,interval_to_process,process_count,neuron_pop,delta_interval=6
            '''

            avalanche_list_out = ca.compute_avalanches(ver=ver, net=net, Ft_f=Ft_filtered, Fn_f=Fn_filtered, cont_Ft=cont_Ft, n_by_spike_time=n_by_spike_time,
                                  interval_to_process=[INTERVAL_times_strt, INTERVAL_times_end], neuron_pop=neuron_pop,delta_interval=window)

            ca.kill_workers(process_count=num_processes)

            alpha = ca.powerlaw_fit(avalanche_list_out)
            alpha_l.append(alpha)
            time_sec_l.append(TIME[i]+(sp_arr[j]*100))

            while ca.status_of_comp:
                time.sleep(1)

        sup.save_non_tf_data(
            names=['time_sec_l','alpha_l'], data=[time_sec_l,alpha_l], filename=save_filename, savePath=ca.data_Path)



if __name__ == '__main__':
    main(sys.argv[1:])

