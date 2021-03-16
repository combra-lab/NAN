import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
import NAN_support_lib as sup



class izhikevich_network:
    def __init__(self):
        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/'

        print('Main working directory: ' + str(self.main_Path))
        print('Networks directory: ' + str(self.network_Path))
        print('Data directory: ' + str(self.data_Path))

    def initialize_network_parameters_v1(self,a,b,c,d,w,w_mask):

        '''
        INPUTS:
        a,b,c,d VECTORS 
        w WEIGHT MATRIX/CONNECTIVITY MATRIX : axis=0 is output neurons, axis=1 is input neurons
        
        :return: TENSORFLOW VARIABLES: A,B,C,D,V,U,W,I 
        '''

        assert(np.shape(a)==np.shape(b))
        assert (np.shape(a) == np.shape(c))
        assert (np.shape(a) == np.shape(d))
        assert (np.shape(a) == np.shape(b))
        assert ((np.shape(a)[0],np.shape(a)[0]) == np.shape(w))
        assert (np.shape(w) == np.shape(w_mask))

        A = tf.Variable(a, dtype=tf.float32, expected_shape=[len(a)], name='A')
        B = tf.Variable(b, dtype=tf.float32, expected_shape=[len(b)], name='B')
        C = tf.Variable(c, dtype=tf.float32, expected_shape=[len(c)], name='C')
        D = tf.Variable(d, dtype=tf.float32, expected_shape=[len(d)], name='D')

        W = tf.Variable(w, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='W')
        W_mask = tf.Variable(w_mask, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='W_mask')

        V = tf.Variable(-65.0*np.ones(np.shape(a),dtype=np.float32), dtype=tf.float32, expected_shape=[len(d)], name='V')
        U = tf.Variable(np.multiply(b,-65.0*np.ones(np.shape(a),dtype=np.float32)), dtype=tf.float32, expected_shape=[len(d)], name='U')
        V_store = tf.Variable(np.zeros(np.shape(a),dtype=np.float32), dtype=tf.float32, expected_shape=[len(d)], name='V_store')

        I = tf.Variable(np.zeros(np.shape(a),dtype=np.float32))

        S = tf.Variable(np.zeros(len(a), dtype=np.float32), dtype=tf.float32, expected_shape=[len(a)], name='S')

        S_store = tf.Variable(np.zeros((len(a), 1000), dtype=np.float32), dtype=tf.float32,
                              expected_shape=[len(a), 1000],
                              name='S_store')

        ### generate constants list
        tf_inds = []
        for i in range(0,1000):
            ind = []
            for j in range(0,len(a)):
                ind.append([j,i])
            tf_inds.append(tf.constant(ind))


        return ['A','B','C','D','V','U','I','W','W_mask','V_store','S','S_store','tf_inds'],[A,B,C,D,V,U,I,W,W_mask,V_store,S,S_store,tf_inds]

    def reinitialize_network_parameters_v1(self, a, b, c, d, v, u, i, w, w_mask):
        '''
        INPUTS:
        a,b,c,d VECTORS 
        w WEIGHT MATRIX/CONNECTIVITY MATRIX : axis=0 is output neurons, axis=1 is input neurons

        :return: TENSORFLOW VARIABLES: A,B,C,D,V,U,W,I 
        '''

        assert (np.shape(a) == np.shape(b))
        assert (np.shape(a) == np.shape(c))
        assert (np.shape(a) == np.shape(d))
        assert (np.shape(a) == np.shape(b))
        print(np.shape(a))
        print(np.shape(w))
        assert ((np.shape(a)[0], np.shape(a)[0]) == np.shape(w))
        assert (np.shape(w) == np.shape(w_mask))

        A = tf.Variable(a, dtype=tf.float32, expected_shape=[len(a)], name='A')
        B = tf.Variable(b, dtype=tf.float32, expected_shape=[len(b)], name='B')
        C = tf.Variable(c, dtype=tf.float32, expected_shape=[len(c)], name='C')
        D = tf.Variable(d, dtype=tf.float32, expected_shape=[len(d)], name='D')

        W = tf.Variable(w, dtype=tf.float32, expected_shape=[np.shape(w)[0], np.shape(w)[1]], name='W')
        W_mask = tf.Variable(w_mask, dtype=tf.float32, expected_shape=[np.shape(w_mask)[0], np.shape(w_mask)[1]], name='W_mask')

        V = tf.Variable(v, dtype=tf.float32, expected_shape=[len(v)], name='V')
        U = tf.Variable(u, dtype=tf.float32,expected_shape=[len(u)], name='U')
        V_store = tf.Variable(np.zeros(np.shape(a), dtype=np.float32), dtype=tf.float32, expected_shape=[len(d)],
                              name='V_store')


        I = tf.Variable(i)

        S = tf.Variable(np.zeros(len(a), dtype=np.float32), dtype=tf.float32, expected_shape=[len(a)], name='S')

        S_store = tf.Variable(np.zeros((len(a), 1000), dtype=np.float32), dtype=tf.float32, expected_shape=[len(a), 1000],
                        name='S_store')

        ext_I_ph = tf.placeholder(dtype=tf.float32,shape=[len(i)],name='ext_I_ph')

        ### generate constants list
        tf_inds = []
        for i in range(0, 1000):
            ind = []
            for j in range(0, len(a)):
                ind.append([j, i])
            tf_inds.append(tf.constant(ind))

        return ['A','B','C','D','V','U','I','W','W_mask','V_store','S','S_store','tf_inds','ext_I_ph'],[A,B,C,D,V,U,I,W,W_mask,V_store,S,S_store,tf_inds,ext_I_ph]


    def update_neuron_states(self,v,v_store,u,a,b,input,integration_step_size):

        '''
        
        :param v: IZHIKEVICH NEURON VOLTAGE VARIABLE
        :param v_store: IZHIKEVICH NEURON VOLTAGE STORAGE VARIABLE
        :param u: IZHIKEVICH NEURON RECOVERY VARIABLE
        :param a: IZHIKEVICH NEURON PARAMETER
        :param b: IZHIKEVICH NEURON PARAMETER
        :param input: INPUT CURRENT VECTOR FOR NEURONS 
        :param integration_step_size: INTEGRATION STEP IN ms
        
        :return: V AND U VARIABLE STATES FOR THIS TIMESTEP BASED ON INPUT FROM PREVIOUS TIMESTEP
        '''

        dv = tf.scalar_mul(integration_step_size,tf.add(140.0,tf.add_n([tf.scalar_mul(0.04,tf.pow(v,2.0)),tf.scalar_mul(5.0,v),tf.negative(u),input])))
        du = tf.scalar_mul(integration_step_size,tf.multiply(a,tf.subtract(tf.multiply(b,v_store),u)))


        return tf.assign(v,tf.add(v,dv)),tf.assign(u,tf.add(u, du)),tf.assign(v_store,v)


    def register_spikes(self,v,s):

        '''
        COMPUTES SPIKES FROM NEURON VOLTAGES
        
        :param v: IZHIKEVICH NEURON VOLTAGE VARIABLE
        :param s: NEURON SPIKE VECTOR
        
        :return: SPIKE VECTOR FOR THIS ms TIMESTEP
        '''

        # 0 FOR SPIKE, 1 FOR NO SPIKE
        inv_spike_vec = tf.ceil(tf.clip_by_value(tf.subtract(30.0,v),0.0,1.0))

        # 1 FOR SPIKE, 0 FOR NO SPIKE
        return tf.assign(s,tf.add(1.0,tf.negative(inv_spike_vec)))


    def spike_store(self,s_store,s,c,tf_inds,neurons):

        '''
        STORES SPIKES
        
        :param s_store: SPIKE STORAGE MATRIX 
        :param s: NEURON SPIKE VECTOR
        :param c: IZHIKEVICH NEURON PARAMETER - PLACEHOLDER
        :param tf_inds: INDEX LIST FOR STORING EACH SPIKE BASED ON NEURON AND ms TIME
        :param neurons: NUM OF NEURONS IN NETWORK
        
        :return: UPDATES STORAGE WITH THIS TIMESTEP'S SPIKES
        '''

        spike_save_ops = []
        for i in range(0,1000):
            spike_save_ops.append(tf.assign(s_store,tf.add(tf.multiply(s_store, tf.add(1.0,tf.scatter_nd(tf_inds[i], tf.clip_by_value(c,-1.0,0.0), shape=[neurons, 1000]))), tf.scatter_nd(tf_inds[i],s,shape=[neurons,1000]))))

        return spike_save_ops


    def reset_spiked_neurons(self,s,c,d,v,u):

        '''
        
        :param s: NEURON SPIKE VECTOR
        :param c: IZHIKEVICH NEURON PARAMETER
        :param d: IZHIKEVICH NEURON PARAMETER
        :param v: IZHIKEVICH NEURON VOLTAGE VARIABLE
        :param u: IZHIKEVICH NEURON RECOVERY VARIABLE
        
        :return: RESET VARIABLES V,U FOR ALL NEURONS THAT SPIKED IN PREVIOUS ms TIMESTEP
        '''

        # 0 FOR SPIKE, 1 FOR NO SPIKE
        inv_spike_vec = tf.abs(tf.subtract(s,1.0))

        new_v = tf.add(tf.multiply(inv_spike_vec,v),tf.multiply(s,c))
        new_u = tf.add(tf.multiply(inv_spike_vec,u),tf.add(tf.multiply(s,u),tf.multiply(s,d)))

        return tf.assign(v,new_v),tf.assign(u,new_u)


    def reset_input_vector(self,i):

        '''
        
        :param i: INPUT CURRENT VECTOR FOR NEURONS
        
        :return: ZEROED OUT INPUT CURRENT VECTOR
        '''

        return tf.assign(i,tf.scalar_mul(0.0,i))


    def add_external_input(self,i,ext_i):

        '''
        
        :param i: INPUT CURRENT VECTOR FOR NEURONS
        :param ext_i: INPUT CURRENT VECTOR FOR NEURONS WITH EXTERNAL SIMULATION
        
        :return: INPUT CURRENT VECTOR FOR NEURONAL NETWORK WITH EXTERNAL INPUT INCLUDED FOR THIS ms TIMESTEP
        '''

        return tf.assign(i,tf.add(i,ext_i))


    def propagate_spikes(self,i,w,w_mask,spikes):

        '''
        
        :param i: INPUT CURRENT VECTOR FOR NEURONS
        :param w: WEIGHT MATRIX FOR NEURON NETWORK CONNECTIONS
        :param w_mask: MASK FOR NETWORK CONNECTIONS
        :param spikes: NEURON SPIKES FROM PREVIOUS ms TIMESTEP
        
        :return: INPUT CURRENT VECTOR FOR NEURONAL NETWORK WITH SPIKES FROM PRESYNAPTIC NEURONS INCLUDED FOR THIS ms TIMESTEP
        '''

        return tf.assign(i,tf.add(i,tf.squeeze(tf.matmul(tf.multiply(w,w_mask),tf.expand_dims(spikes,axis=1)))))


    def gather_spike_rates_at_synapse_level_all(self,W_rates,spikes, vertical_ones_vec):

        return tf.assign(W_rates, tf.add(W_rates, tf.matmul(vertical_ones_vec, tf.reshape(spikes, [1, -1]))))


    def propagate_SIC_uniformly(self, i, SIC_mask, sic_vec, scalar, dims):

        '''

        :param i: INPUT CURRENT VECTOR FOR NEURONS
        :param SIC_mask: MASK FOR TRIPARTITE CONNECTIONS
        :param sic_vec: SIC WAVE VALUE
        :param scalar: AMPLITUDE OF SIC WAVE
        :param dims:  precomputed dimnsions of: [0] num_astrocytes, 
                                                    [1] [0][1] shape_tuple_of_reshaped_W*_matrix, 
                                                    [2] [0][1] shape_tuple_of_original_shaped_W 
        
        :return: INPUT CURRENT VECTOR FOR NEURONAL NETWORK WITH SIC STIMULATION INCLUDED FOR THIS ms TIMESTEP
        '''

        SIC_mask_rshp = tf.reshape(SIC_mask, shape=[dims[0], dims[1][0], dims[1][1]])

        exp_sic_vec = tf.tile(tf.expand_dims(tf.expand_dims(sic_vec, axis=1), axis=2),
                              tf.constant([1, dims[1][0], dims[1][1]]))

        i_sic = tf.reduce_sum(tf.reshape(tf.multiply(SIC_mask_rshp, tf.scalar_mul(scalar,exp_sic_vec)),
                                                                                 shape=[dims[2][0], dims[2][1]]),
                              axis=1)

        return tf.assign(i, tf.add(i, i_sic))


    def open_network(self,net_name,netPath):

        '''
        
        :param net_name: INT VALUE
        :param netPath: PATH TO NETWORK
        
        :return: NETWORK INITIALIZED AS TENSORFLOW DATA STRUCTURES: LIST OF STRINGS, LIST OF TF DATA STRUCTURES, A SCALAR, LIST OF STRINGS, NUMPY DATA STRUCTURE
        '''

        names,data = sup.unpack_file(net_name,netPath)

        names_t,data_t = self.reinitialize_network_parameters_v1(a=data[0],b=data[1],c=data[2],d=data[3],v=data[4],u=data[5],i=data[6],w=data[7],w_mask=data[8])

        num_neurons = len(data[0])

        return names_t,data_t,num_neurons,names,data

