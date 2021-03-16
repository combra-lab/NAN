import sys, getopt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import time
import NAN_neuron_network as nani
import NAN_astrocyte_network as nastro
import NAN_STDP as nasti
import NAN_support_lib as sup
import NAN_saver as ps


















class run_Izhikevich_network:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/'

    def initialize_sic_mask(self, sic_mask_temp, astro_connectivity_density,input_neuron_ind_range,output_neuron_ind_range):

        '''
        
        :param sic_mask_temp: GENERAL MASK FROM NETWORK
        :param astro_connectivity_density: PERCENTAGE OF TRIPARTITE SYNAPSES FROM ALL SYNAPSES IN NETWORK
        :param input_neuron_ind_range: INDEX RANGE FOR INPUT NEURONS
        :param output_neuron_ind_range: INDEX RANGE FOR OUTPUT NEURONS
        :return: 
        '''

        sic_mask_temp[:,800:output_neuron_ind_range[1]] = 0
        sic_mask_temp[input_neuron_ind_range[0]:output_neuron_ind_range[1],:] = 0

        existing_syn_idx = np.where(sic_mask_temp == 1.0)
        sel_ast_idx = np.random.permutation(np.arange(len(existing_syn_idx[0])))[
                      0:int(len(existing_syn_idx[0]) * astro_connectivity_density)]
        sic_mask = np.zeros(np.shape(sic_mask_temp), dtype=np.float32)
        sic_mask[existing_syn_idx[0][sel_ast_idx], existing_syn_idx[1][sel_ast_idx]] = 1.0

        return sic_mask


    def run_net_v1(self
                   , save_ver
                   , net_num
                   , run_time_s
                   , astro_stats
                   , t_ref_control_ranges_list_mat
                   , t_ref_control_period
                   , stdp_time_range
                   , sum_v_b_trgt
                   , input_mean=0
                   , input_var=1
                   , stdp_tau_pls=10
                   , stdp_tau_mns=10
                   , integration_t_step=0.25
                   , stdp_Ap=1.0
                   , stdp_Am=1.0
                   , CA_THRSH=0.2
                   , save_data_mod=500
                   , glut_lvl=0.2
                   , w_min_max=[-10,10]
                   , astro_connectivity_density=0.65
                   ):

        # -SEEDSET- SET RANDOM SEED SEED
        np.random.seed()
        rnd_sd = np.random.randint(0, 1000)
        np.random.seed(rnd_sd)
        ######################## SET SEED MANUALLY ######################
        # SET SEED FOR UR NEAR-CRITICAL STABILIZATION:
        # np.random.seed(474)
        # SET SEED FOR SFRC NEAR-CRITICAL STABILIZATION:
        # np.random.seed(497)
        ######################## SET SEED MANUALLY ######################

        if len(t_ref_control_ranges_list_mat) > 0:
            t_ref_last = -2000

        # CREATE CONTROL ARR FOR TIME RANGE WHEN SIC IS DRIVEN BY GCH-I
        if len(t_ref_control_ranges_list_mat)>0:
            if np.asarray(t_ref_control_ranges_list_mat).ndim==1:
                t_ref_control_ranges_list_mat_ARR = np.expand_dims(np.asarray(t_ref_control_ranges_list_mat),axis=0)
            elif np.asarray(t_ref_control_ranges_list_mat).ndim==2:
                t_ref_control_ranges_list_mat_ARR = np.asarray(t_ref_control_ranges_list_mat)
            assert(t_ref_control_ranges_list_mat_ARR.ndim==2)
        else:
            print('ERROR---> INCORRECT SIZE FOR SIC_RANGES')

        ### CREATE CONTROL ARR FOR TIME RANGE WHEN STDP IS ACTIVE
        if len(stdp_time_range) > 0:
            if np.asarray(stdp_time_range).ndim == 1:
                stdp_time_range_ARR = np.expand_dims(np.asarray(stdp_time_range), axis=0)
            elif np.asarray(stdp_time_range).ndim == 2:
                stdp_time_range_ARR = np.asarray(stdp_time_range)
            assert (stdp_time_range_ARR.ndim == 2)
        else:
            print('ERROR---> INCORRECT SIZE FOR STDP_RANGES')

        if len(t_ref_control_ranges_list_mat) > 0:
            adder_mat = np.concatenate([np.expand_dims(np.zeros(np.shape(t_ref_control_ranges_list_mat_ARR)[0]), axis=1),
                            np.expand_dims(2000 * np.ones(np.shape(t_ref_control_ranges_list_mat_ARR)[0]), axis=1)],
                           axis=1)

            assert(np.shape(t_ref_control_ranges_list_mat_ARR)==np.shape(np.asarray(adder_mat)))
            t_ref_control_ranges_list_mat_exp = np.add(t_ref_control_ranges_list_mat_ARR,adder_mat)


        tf.reset_default_graph()

        ##### LOAD NETWORK DATA/STRUCTURES #####
        self.data_Path = self.data_Path + '/ver_'+str(save_ver)+'/'
        sup.check_create_save_dir(self.data_Path)
        net_name = 'Network_'+str(net_num)

        # INITIALIZE PARALLEL PROCESSES FOR SAVING
        smart_saver = ps.multi_process(num_processes=2, save_path=self.data_Path)

        nanis = nani.izhikevich_network()
        names,data,num_of_neurons,orig_names,orig_data = nanis.open_network(net_name=net_name,netPath=self.network_Path)

        A = data[names.index('A')]
        B = data[names.index('B')]
        C = data[names.index('C')]
        D = data[names.index('D')]
        V = data[names.index('V')]
        U = data[names.index('U')]
        I = data[names.index('I')]
        W = data[names.index('W')]
        W_mask = data[names.index('W_mask')]
        V_store = data[names.index('V_store')]
        S = data[names.index('S')]
        S_store = data[names.index('S_store')]
        TF_INDS = data[names.index('tf_inds')]
        ext_I_ph = data[names.index('ext_I_ph')]

        # DEFINE NEURON POPULATION RANGES
        reservoir_neuron_ind_range = orig_data[orig_names.index('res_range')]
        input_neuron_ind_range = orig_data[orig_names.index('inp_range')]
        output_neuron_ind_range = orig_data[orig_names.index('out_range')]

        # TF OPS FOR MEASURING AVERAGE W OF NETWORK
        W_exc_mean = tf.divide(tf.reduce_sum(tf.multiply(W,W_mask)[0:1000,0:800]),tf.reduce_sum(W_mask[0:1000,0:800]))
        W_inh_mean = tf.divide(tf.reduce_sum(tf.multiply(W,W_mask)[0:1000, 800:1000]), tf.reduce_sum(W_mask[0:1000, 800:1000]))

        #### CREATE SIC MASK FROM W_MASK
        gen_mask = orig_data[orig_names.index('W_mask')]
        sic_mask_temp = np.multiply(np.ones(np.shape(gen_mask)),gen_mask)
        sic_mask = self.initialize_sic_mask(sic_mask_temp=sic_mask_temp,
                                            astro_connectivity_density=astro_connectivity_density,
                                            input_neuron_ind_range=input_neuron_ind_range,
                                            output_neuron_ind_range=output_neuron_ind_range)

        ## EXC MASK CONSTRUCTION
        exc_mask = np.multiply(np.ones(np.shape(gen_mask)),gen_mask)
        exc_mask[:, 800:1000] = 0
        exc_mask[:, input_neuron_ind_range[0]:output_neuron_ind_range[1]] = 0
        exc_mask[input_neuron_ind_range[0]:output_neuron_ind_range[1], :] = 0

        ## INH MASK CONSTRUCTION
        inh_mask = np.multiply(np.ones(np.shape(gen_mask)),gen_mask)
        inh_mask[:, 0:800] = 0
        inh_mask[:, input_neuron_ind_range[0]:output_neuron_ind_range[1]] = 0
        inh_mask[input_neuron_ind_range[0]:output_neuron_ind_range[1], :] = 0

        ## RES MASK CONSTRUCTION
        res_mask = np.multiply(np.ones(np.shape(gen_mask)),gen_mask)
        res_mask[:, input_neuron_ind_range[0]:output_neuron_ind_range[1]] = 0
        res_mask[input_neuron_ind_range[0]:output_neuron_ind_range[1], :] = 0

        # Check that all types of connections are present
        assert (np.sum(sic_mask) > 0)
        assert (np.sum(inh_mask) > 0)
        assert (np.sum(exc_mask) > 0)
        assert (np.sum(res_mask) > 0)


        ## ASTRO INITIALIZATIONS
        num_of_astros = astro_stats[0]
        SIC_amp_w_percent = astro_stats[1]
        astro_W_dims = astro_stats[2]
        orig_W_shp = [np.shape(sic_mask)[0],np.shape(sic_mask)[1]]

        nastros = nastro.gpu_astro(num_astros=num_of_astros, max_syns_per_astro=1200*1200)
        t_ref, SIC, SIC_mask, t_ref_ph, t_ph,INH_mask, RES_mask, EXC_mask = nastros.initialize_SIC_manual_driver_parameters(num_astros=num_of_astros, sic_mask=sic_mask,inh_mask=inh_mask,res_mask=res_mask,exc_mask=exc_mask)


        ## GRAPH CONSTRUCTION
        update_V, update_U, store_V = nanis.update_neuron_states(v=V,v_store=V_store,u=U,a=A,b=B,input=I,integration_step_size=integration_t_step)
        compute_spikes = nanis.register_spikes(v=V,s=S)
        spike_reset_V, spike_reset_U = nanis.reset_spiked_neurons(s=S,c=C,d=D,v=V,u=U)
        store_spikes_by_t = nanis.spike_store(s_store=S_store,s=S,c=C,tf_inds=TF_INDS,neurons=num_of_neurons)
        reset_I = nanis.reset_input_vector(i=I)
        propagate_spikes_to_I = nanis.propagate_spikes(i=I,w=W,w_mask=W_mask,spikes=S)
        add_external_input_to_I = nanis.add_external_input(i=I, ext_i=ext_I_ph)


        ####### VARIABLE FOR NEURON-ASTROCYTE INTERACTIONS
        # PLACEHOLDER
        vertical_ones = tf.constant(np.ones((np.shape(sic_mask)[0],1),dtype=np.float32),dtype=tf.float32,shape=[np.shape(sic_mask)[0],1])
        # MATRIX (N,N) FOR AGGREGATING SPIKES AT SYNAPSE LEVEL
        W_rates = tf.Variable(np.zeros(np.shape(sic_mask),dtype=np.float32),dtype=tf.float32,expected_shape=[np.shape(sic_mask)[0],np.shape(sic_mask)[1]])


        seconds_normalizer_ph = tf.placeholder(dtype=tf.float32)
        # COMPUTE SPIKE TRANSMISSION RATE PER SYNAPSE USING W_rates SUM OF SPIKES
        get_cur_rates = tf.assign(W_rates,tf.divide(W_rates,seconds_normalizer_ph))
        # RESETS W_RATES STORAGE OF AGGREGATE SPIKE COUNTS
        reset_W_rates = tf.assign(W_rates,tf.scalar_mul(0.0,W_rates))

        # ADD NEW SPIKES TO SUM OF SPIKES IN GIVEN SECOND
        store_synaptic_transmissions = nanis.gather_spike_rates_at_synapse_level_all(W_rates=W_rates, spikes=S,
                                                                                     vertical_ones_vec=vertical_ones)

        ###### GCH-I ASTRO MODEL ########
        # GLUTAMATE CONCENTRATION MATRIX FOR EACH SYNAPSE AND EACH SPIKE CONTRIBUTION
        syn_spike_mat = tf.multiply(SIC_mask, tf.scalar_mul(glut_lvl,tf.matmul(vertical_ones, tf.reshape(S, [1, -1]))))
        ## INITIALIZE GCH-I V_B WEIGHTS
        v_b = nastros.initialize_astro_v_b(sum_v_b_trgt=sum_v_b_trgt, sic_mask_np=sic_mask)


        syn_spike_mat_reshaped_for_astro_input = tf.reshape(syn_spike_mat, [1, -1])
        sic_mask_reshaped_for_astro_input = tf.reshape(SIC_mask, [1, -1])
        v_b_mat_reshaped_for_astro_input = tf.reshape(v_b, [1, -1])


        ###### BIOLOGICAL ASTRO MODEL GRAPH #####
        ip3_state, ca_state, h_state, ip3_store, ca_store = nastros.initialize_vars()

        # COMPUTE NEXT CA STATE
        new_ca_var_states = nastros.run_ca_state_transition(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                          L1_h_state=h_state)
        # STORE CA STATE
        store_ca_var_states = tf.assign(ca_store, new_ca_var_states)

        # COMPUTE NEXT H STATE
        new_h_var_states = nastros.run_h_state_transition(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                        L1_h_state=h_state)
        # UPDATE H STATE
        update_h_var_states = tf.assign(h_state, new_h_var_states)

        # COMPUTE NEXT IP3 STATE
        new_ip3_var_states = nastros.run_ip3_state_transition_v1(L1_ip3_state=ip3_state, L1_ca_state=ca_state,
                                                               syn_inp=syn_spike_mat_reshaped_for_astro_input,
                                                               v_beta_var=v_b_mat_reshaped_for_astro_input,
                                                               input_morph=sic_mask_reshaped_for_astro_input)
        # UPDATE IP3 STATE
        update_ip3_var_states = tf.assign(ip3_state, new_ip3_var_states)
        # UPDATE CA STATE
        update_ca_var_states = tf.assign(ca_state, ca_store)


        #### SYNAPTIC SPIKE RATES RECORDING OPS #### START
        get_RES_synaptic_spike_rates = tf.divide(tf.reduce_sum(tf.multiply(RES_mask,W_rates)), tf.reduce_sum(RES_mask))
        #### SYNAPTIC SPIKE RATES RECORDING OPS #### END

        # OP ADDING SIC TO POST-SYNAPTIC NEURONS
        add_SIC_input = nanis.propagate_SIC_uniformly(i=I,SIC_mask=SIC_mask,sic_vec=SIC,scalar=SIC_amp_w_percent,dims=[num_of_astros,astro_W_dims,orig_W_shp])

        # EVALUATES NEXT VALUE OF SIC WAVE
        update_SIC_state = nastros.update_SIC_states(SIC=SIC, t_ref=t_ref, t=t_ph)
        update_t_ref = nastros.set_t_ref_manual(t_ref=t_ref,new_t_ref=t_ref_ph)

        ######## STDP GRAPH #######
        # CONSTRUCT MASKS FOR DIFFERENT NEURON POPULATIONS
        aaa = orig_data[orig_names.index('A')] ## filler
        trace_mask_in = np.zeros(np.shape(aaa), dtype=np.float32)
        trace_mask_in[reservoir_neuron_ind_range[0]:reservoir_neuron_ind_range[1]] = 1.0

        trace_mask_in_exc = np.zeros(np.shape(aaa), dtype=np.float32)
        trace_mask_in_exc[0:800] = 1.0

        trace_mask_in_inh = np.zeros(np.shape(aaa), dtype=np.float32)
        trace_mask_in_inh[800:1000] = 1.0

        # INITIALIZE STDP CLASS
        nastis = nasti.stdp(tau_plus=stdp_tau_pls,tau_minus=stdp_tau_mns,A_plus=stdp_Ap,A_minus=stdp_Am,min_w=w_min_max[0],max_w=w_min_max[1])
        names_stdp, structs_stdp = nastis.initialize_stdp_parameters_v0(trace_mask_gen=trace_mask_in,trace_mask_exc=trace_mask_in_exc,trace_mask_inh=trace_mask_in_inh)

        trc_pre = structs_stdp[names_stdp.index('trc_pre')]
        trc_pst = structs_stdp[names_stdp.index('trc_pst')]
        trc_msk_g = structs_stdp[names_stdp.index('trc_msk_gen')]
        trc_msk_e = structs_stdp[names_stdp.index('trc_msk_exc')]
        trc_msk_i = structs_stdp[names_stdp.index('trc_msk_inh')]

        ## STDP TF OPS ##
        update_pre_trace = nastis.update_trace_pre_v0(trace_pre=trc_pre,spike_vec=S)
        update_post_trace = nastis.update_trace_post_v0(trace_post=trc_pst, spike_vec=S)
        update_W_exc,delta_W_exc = nastis.update_exc_weights(W=W,spike_vec=S,trace_pre=trc_pre,trace_post=trc_pst,W_mask=W_mask,trace_mask_gen=trc_msk_g,trace_mask_exc=trc_msk_e)
        update_W_inh,delta_W_inh = nastis.update_inh_weights(W=W, spike_vec=S, trace_pre=trc_pre, trace_post=trc_pst, W_mask=W_mask, trace_mask_gen=trc_msk_g, trace_mask_inh=trc_msk_i)
        clip_W = nastis.clip_weights(W=W,trace_mask_gen=trc_msk_g,trace_mask_exc=trc_msk_e,trace_mask_inh=trc_msk_i)

        ### DATA COLLECTION LISTS
        time_mats = []
        SR_res_list = []
        t_ref_inp_list = []
        ca_list = []
        w_exc_mean_list = []
        w_inh_mean_list = []
        instantaneous_astro_rate = []
        instantaneous_astro_rate_time_ms = []


        f_inst = 0
        initial_time = 0
        final_time = run_time_s

        ### INITIALIZE TF SESSION ###
        # -SEEDSET-
        tf.set_random_seed(1)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # SAVE SIMULATION PARAMETERS
        sup.save_non_tf_data(['run_time_s', 'input_mean', 'input_var', 'astro_stats', 't_ref_control_ranges_list_mat',
                                't_ref_control_period', 'stdp_time_range', 'sum_v_b_trgt', 'CA_THRSH','glut_lvl', 'save_data_mod', 'stdp_tau_pls', 'stdp_tau_mns', 'stdp_Ap',
                                'stdp_Am', 'NP_SEED', 'TF_SEED'],
                               [run_time_s, input_mean, input_var, astro_stats, t_ref_control_ranges_list_mat,
                                t_ref_control_period, stdp_time_range, sum_v_b_trgt, CA_THRSH,glut_lvl,save_data_mod, stdp_tau_pls, stdp_tau_mns, stdp_Ap, stdp_Am,
                                rnd_sd, 1],

                               filename='Sim_STATS_PARAMS_for_Net_' + str(net_num) + '_ver_' + str(save_ver) + '_' + str(
                                   initial_time) + str(final_time),savePath=self.data_Path)


        save_data_mod_COUNTER = 0
        save_start_sec = 0

        # MAIN SIMULATION LOOP - OUTER LOOP ITERATES EVERY SECOND - INNER LOOP ITERATES THROUGH 1000 MS FOR EACH SECOND
        for sec in range(initial_time,final_time):
            st0 = time.time()

            ## MS LOOP
            for t in range(0,1000):

                sess.run(reset_I)

                ## GENERATE RANDOM INPUT FOR INPUT NEURONS
                random_vec = 10.0*np.random.normal(0,1,size=reservoir_neuron_ind_range[2]+input_neuron_ind_range[2]+output_neuron_ind_range[2])

                mask_vector = np.concatenate([np.zeros(reservoir_neuron_ind_range[2], dtype=np.float32),
                                              np.ones(input_neuron_ind_range[2]).astype(
                                                  np.float32), np.zeros(output_neuron_ind_range[2], dtype=np.float32)])

                input_vec = np.multiply(random_vec,mask_vector)

                # ADDS INPUT TO I VARIABLE
                sess.run(add_external_input_to_I, feed_dict={ext_I_ph: input_vec})

                ### ADD SIC INPUT TO I
                sess.run(add_SIC_input)

                # COMPUTE WHICH NEURONS SPIKED
                sess.run(compute_spikes)

                # SAVE SPIKES AT SYNAPSE LEVEL TO GET RATES AFTER
                sess.run(store_synaptic_transmissions)

                # RESETS V, U VARIABLES
                sess.run(spike_reset_V)
                sess.run(spike_reset_U)

                ## SPIKE SAVER FOR EACH NEURON AND TIME
                sess.run(store_spikes_by_t[t])

                # PROPAGATE PRE-SYNAPTIC SPIKES TO POST-SYNAPTIC NEURONS
                sess.run(propagate_spikes_to_I)

                # UPDATE NEURON STATES
                for i in range(0,int(1.0/integration_t_step)):
                    sess.run(store_V)
                    sess.run(update_V)
                    sess.run(update_U)

                #### COMPUTE GCH-I ASTROCYTE
                sess.run(store_ca_var_states)
                sess.run(update_h_var_states)
                sess.run(update_ip3_var_states)
                sess.run(update_ca_var_states)

                ## STORE GCHI-I CA
                ca_list.append(sess.run(ca_state)[0])

                # UPDATE SIC STATES
                sess.run(update_SIC_state,feed_dict={t_ph: (sec*1000+t)*np.ones(1,dtype=np.float32)})

                # APPLY STDP TO NETWORK WEIGHTS
                if len(np.prod(np.sign(np.subtract(stdp_time_range_ARR,((sec*1000+t)))),axis=1)) > np.sum(np.prod(np.sign(np.subtract(stdp_time_range_ARR,((sec*1000+t)))),axis=1)):
                    sess.run(update_W_exc)
                    sess.run(update_W_inh)
                    sess.run(clip_W)

                ### STDP TRACE OPS
                sess.run(update_pre_trace)
                sess.run(update_post_trace)

                ## UPDATE CONTROL PARAMETER FOR SIC STIMULATION DEPENDENT ON GCH-I CA CONCENTRATION LEVELS
                if len(np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                               axis=1)) > np.sum(
                        np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))), axis=1)):

                    # CHECK IF PREVIOUS SIC WAVES IS OVER
                    if ((sec*1000+t)-t_ref_last)>t_ref_control_period:
                        # CHECK IF [CA] IS ABOVE THRESHOLD TO TRIGGER NEXT SIC WAVE INTO SYNAPSES
                        if sess.run(ca_state)>CA_THRSH:
                            t_ref_inp = (sec*1000+t)*np.ones(1,dtype=np.float32)
                            t_ref_inp_list.append(sec*1000+t)

                            if len(np.where(
                                            np.prod(np.sign(
                                                np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                                    axis=1) == -1)[0]) > 0:
                                search_array = np.where(
                                    np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                            axis=1) == -1)[0]
                            else:
                                search_array = np.where(
                                    np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                            axis=1) == 0)[0]
                            range_idx = search_array[0]
                            current_range = t_ref_control_ranges_list_mat_ARR[range_idx]

                            ## MEASURE GCH-I CA WAVE INSTANTANEOUS FREQUENCY
                            if t_ref_last>current_range[0]:
                                f_inst = 1000/((sec*1000+t)-t_ref_last)
                                instantaneous_astro_rate.append(f_inst[0])
                                instantaneous_astro_rate_time_ms.append((sec*1000+t))



                            t_ref_last = t_ref_inp.astype(dtype=np.float32)
                        else:
                            t_ref_inp = (sec * 1000 + t + 1) * np.ones(1, dtype=np.float32)

                        ### COMPUTE TEMPORAL AVERAGE OF GCH-I CA WAVE FREQUENCY
                        if len(np.where(
                            np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                    axis=1) == -1)[0])>0:
                            search_array = np.where(
                                np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                        axis=1) == -1)[0]
                        else:
                            search_array = np.where(
                                np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_ARR, ((sec * 1000 + t)))),
                                        axis=1) == 0)[0]

                        range_idx = search_array[0]
                        current_range = t_ref_control_ranges_list_mat_ARR[range_idx]


                elif len(np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_exp, ((sec * 1000 + t)))),
                               axis=1)) == np.sum(
                        np.prod(np.sign(np.subtract(t_ref_control_ranges_list_mat_exp, ((sec * 1000 + t)))), axis=1)):

                    t_ref_inp = (sec*1000+t+1)*np.ones(1,dtype=np.float32)

                sess.run(update_t_ref,feed_dict={t_ref_ph: t_ref_inp})


            time_mats.append(sess.run(tf.where(tf.equal(S_store,1.0))))

            #### SAVE AVERAGE WEIGHT
            w_exc_mean_list.append(sess.run(W_exc_mean))
            w_inh_mean_list.append(sess.run(W_inh_mean))

            # RECORD SYNAPTIC SPIKE TRANSMISSION RATES
            sess.run(get_cur_rates, feed_dict={seconds_normalizer_ph: 1.0})
            SR_res_list.append(sess.run(get_RES_synaptic_spike_rates))
            sess.run(reset_W_rates)


            st1 = time.time()
            print('Second '+str(sec)+' Completed in ' + str(st1-st0) + ' secs')


            if (sec+1)%save_data_mod==0:
                # SAVE ALL OTHER DATA
                smart_saver.save_data(signal=1
                                      ,names = ['save_sec_range', 'SR_res_list', 't_ref_inp_list', 'ca_list',
                         'instantaneous_astro_rate', 'instantaneous_astro_rate_time_ms',
                          'w_exc_mean_list', 'w_inh_mean_list']
                                      , data = [[save_start_sec, sec + 1], SR_res_list, t_ref_inp_list, ca_list,
                          instantaneous_astro_rate, instantaneous_astro_rate_time_ms,
                          w_exc_mean_list, w_inh_mean_list]
                                      ,save_filename='Sim_Data_for_Net_' + str(net_num) + '_ver_' + str(save_ver) + '_' + str(
                        initial_time) + str(final_time) + '_sp_'+str(save_data_mod_COUNTER) + '_additional')



                SR_res_list = []
                t_ref_inp_list = []
                ca_list = []
                instantaneous_astro_rate = []
                instantaneous_astro_rate_time_ms = []
                w_exc_mean_list = []
                w_inh_mean_list = []

                # SAVE SPIKE DATA
                smart_saver.save_data(signal=1
                                      , names=['save_sec_range','time_sec_spikes']
                                      , data=[[save_start_sec,sec+1],time_mats]
                                      , save_filename='Sim_Data_for_Net_' + str(net_num) + '_ver_' + str(
                        save_ver) + '_' + str(
                        initial_time) + str(final_time) + '_sp_' + str(save_data_mod_COUNTER)
                                      )


                time_mats = []

                save_start_sec = (sec + 1)
                save_data_mod_COUNTER += 1

        sess.close()

        if len(time_mats) == 0:
            # SAVE SPIKE DATA
            sup.save_non_tf_data(['save_sec_range','time_sec_spikes'],[[save_start_sec, sec+1],time_mats],filename='Sim_Data_for_Net_'+str(net_num)+'_ver_'+str(save_ver) + '_' + str(initial_time)+str(final_time)+ '_sp_'+str(save_data_mod_COUNTER),savePath=self.data_Path)

            # SAVE ALL OTHER DATA
            sup.save_non_tf_data(['save_sec_range', 'SR_res_list', 't_ref_inp_list', 'ca_list',
                         'instantaneous_astro_rate', 'instantaneous_astro_rate_time_ms',
                          'w_exc_mean_list', 'w_inh_mean_list'], [[save_start_sec, sec + 1], SR_res_list, t_ref_inp_list, ca_list,
                          instantaneous_astro_rate, instantaneous_astro_rate_time_ms, w_exc_mean_list, w_inh_mean_list],
                                 filename='Sim_Data_for_Net_'+str(net_num)+'_ver_'+str(save_ver) + '_' + str(initial_time)+str(final_time)+ '_sp_'+str(save_data_mod_COUNTER) +'_additional',savePath=self.data_Path)


        ### TERMINATE PARALLEL PROCESSES
        saved_file_names = smart_saver.kill_workers(process_count=2)

        for i in range(0,len(saved_file_names)):
            print('SAVED: '+str(saved_file_names[i]))

        # SAVE LOG FILE OF SIM
        log_filename = 'Sim_Log_for_Net_' + str(net_num)+'_ver_'+str(save_ver) + '_Log.txt'
        log_fn = os.path.abspath(os.path.join(self.data_Path, log_filename))
        with open(log_fn, 'w') as f:
            f.write('LOG_FOR_SIM_FOR_NETWORK_' + str(net_num) + '_VER_'+ str(save_ver) +'\n\n')
            f.write('   integration_time_step: ' + str(integration_t_step) + '\n')
            f.write('   run_time_s: ' + str(run_time_s) + '\n')
            f.write('   input_mean: ' + str(input_mean) + '\n')
            f.write('   input_var: ' + str(input_var) + '\n\n')

            f.write('   ASTRO STATS: \n')
            f.write('   num_astros:  ' + str(astro_stats[0]) + '\n')
            f.write('   amp_scalar:  ' + str(astro_stats[1]) + '\n')
            f.write('   W_orig_shape:  ' + str(astro_stats[2]) + '\n')

            if len(t_ref_control_ranges_list_mat) > 0:
                f.write('   sic_start_of_times_in_ms:  ' + str(t_ref_control_ranges_list_mat_ARR[:,0]) + '\n')
                f.write('   sic_end_of_times_in_ms:  ' + str(t_ref_control_ranges_list_mat_ARR[:,1]) + '\n')
                f.write('   sic_mod_in_ms:  ' + str(t_ref_control_period) + '\n\n')
            else:
                f.write('   sic_start_of_times_in_ms:  N/A \n')
                f.write('   sic_end_of_times_in_ms:  N/A \n')
                f.write('   sic_mod_in_ms:  N/A \n\n')

            f.write('sum_v_b_trgt:    '+str(sum_v_b_trgt))

            f.write('   STDP STATS: \n')


            if len(stdp_time_range) > 0:
                f.write('   stdp start times ms:   ' + str(stdp_time_range_ARR[:,0]) + '\n')
                f.write('   stdp end times ms:   ' + str(stdp_time_range_ARR[:,1]) + '\n')
            else:
                f.write('   stdp start times ms:   N/A \n')
                f.write('   stdp end times ms:   N/A \n')

            f.write('   stdp_tau_pls:   ' + str(stdp_tau_pls) + '\n')
            f.write('   stdp_tau_mns:   ' + str(stdp_tau_mns) + '\n')
            f.write('   stdp_tau_pls_approx_mult:   ' + str(nastis.tau_plus_decay)  + '\n')
            f.write('   stdp_tau_mns_approx_mult:   ' + str(nastis.tau_minus_decay)  + '\n')




def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "net_num=", "sim_run_time_s=", "net_type="])
    except getopt.GetoptError:
        print('Incorrect arguments')

        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ver_num':
            VER_NUM = int(arg)

        elif opt == '--net_num':
            NET_NUM = int(arg)

        elif opt == '--sim_run_time_s':
            RUN_FOR_SECS = int(arg)

        elif opt == '--net_type':
            NET_TYPE = str(arg)

        else:
            print('Error, exiting')
            sys.exit()

    ####### UNCOMMENT IF USING GPU #######
    # GPU_NUM = input('GPU? ')
    #
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)
    # print('GPU: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))
    # --------------------------------

    print('SIM_VER: ' + str(VER_NUM))
    print('NETWORK_NUMBER: ' + str(NET_NUM))
    print('SIMULATION_RUN_TIME: ' + str(RUN_FOR_SECS))
    print('NET_TYPE: ' + str(NET_TYPE))

    if NET_TYPE=='SFRC':
        astro_sic_amplitude = 1.1
        as_l = [1, astro_sic_amplitude, [1200, 1200]]

        # SUM OF ASTROCYTE INPUT WEIGHTS ENABLING NETWORK STABILIZATION AT NEAR-CRITICALITY
        SUM_V_B_TRGT = 223.0
        # FOR SUB-CRITICAL DYNAMICS, INCREASE SUM_V_B_TRGT
        # FOR SUPER-CRITICAL DYNAMICS, DECREASE SUM_V_B_TRGT

    elif NET_TYPE=='UR':
        astro_sic_amplitude = 1.6
        as_l = [1, astro_sic_amplitude, [1200, 1200]]

        # SUM OF ASTROCYTE INPUT WEIGHTS ENABLING NETWORK STABILIZATION AT NEAR-CRITICALITY
        SUM_V_B_TRGT = 108.0
        # FOR SUB-CRITICAL DYNAMICS, INCREASE SUM_V_B_TRGT
        # FOR SUPER-CRITICAL DYNAMICS, DECREASE SUM_V_B_TRGT

    print('SIC AMPLITUDE: ' + str(astro_sic_amplitude))

    ## SIC APPLICATION STARTS 4 SECONDS INTO SIMULATION UNTIL END
    TRC_RANGE = [
                  [100000 * 1000, 200000 * 1000],   # FILLER
                  [4000, RUN_FOR_SECS * 1000]       # SIC RANGE
                  ]

    STDP_TIME_RANGE = [
                        [100000 * 1000, 200000 * 1000],   # FILLER
                        [1001, RUN_FOR_SECS * 1000]       # SIC RANGE
                        ]


    run_net = run_Izhikevich_network()
    run_net.run_net_v1(save_ver=VER_NUM,
                       net_num=NET_NUM,
                       run_time_s=RUN_FOR_SECS,
                       astro_stats=as_l,
                       t_ref_control_ranges_list_mat=TRC_RANGE,
                       t_ref_control_period=5000,
                       stdp_time_range=STDP_TIME_RANGE,
                       sum_v_b_trgt=SUM_V_B_TRGT,
                       save_data_mod=100)


if __name__ == '__main__':
    main(sys.argv[1:])


