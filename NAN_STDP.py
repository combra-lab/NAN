import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class stdp:
    def __init__(self,
                 a_plus = 0.1,
                 a_minus = 0.12,
                 tau_plus = 10,
                 tau_minus = 10,
                 A_plus = 1.0, #0.001,
                 A_minus = 1.0, #0.001,
                 max_w = 10.0,
                 min_w = -10.0):

        # --------- STDP PARAMETERS -----------
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.max_w = max_w
        self.min_w = min_w

        # PRECOMPUTATIONS FOR FASTER IMPLEMENTATION
        tp = self.approximate_stdp_decay_multiplier(tau=tau_plus)
        self.tau_plus_decay = tp
        tm = self.approximate_stdp_decay_multiplier(tau=tau_minus)
        self.tau_minus_decay = tm


    def find_nearest(self, array, value):

        '''
        FINDS VALUE AND INDEX OF VALUE IN array CLOSEST TO value
        
        :param array: 1D ARRAY
        :param value: SCALAR
        :return: 
        '''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return array[idx], idx

    def approximate_stdp_decay_multiplier(self, tau):

        '''
        
        :param tau: STDP TAU SCALAR
        
        :return: SCALAR APPROXIMATING TAU DECAY RATE
        '''

        ratio_to_reach = (1 / np.e)

        x = np.linspace(0, tau, tau * 100)
        y = np.exp(-x / tau)

        approx_mults = np.linspace(0.01, 0.9999, 1000)
        state = np.ones(np.shape(approx_mults))
        for i in range(0, tau):
            state = np.multiply(approx_mults, state)

        na, ind = self.find_nearest(state, y[-1])

        approx_multiple = approx_mults[ind]

        return approx_multiple




    def initialize_stdp_parameters_v0(self, trace_mask_gen,trace_mask_exc,trace_mask_inh):

        '''
        INITIALIZE TENSORFLOW DATA STRUCTURES FOR STDP
        
        :param trace_mask_gen: MASK FOR ALL CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_exc: MASK FOR ALL EXCITATORY CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_inh: MASK FOR ALL INHIBITORY CONNECTIONS IN RECURRENT NETWORK
        
        :return: TENSORFLOW DATASTRUCTURES
        '''


        trc_pre = tf.Variable(np.zeros(np.shape(trace_mask_gen)), dtype=tf.float32, expected_shape=[len(trace_mask_gen)], name='trc_pre')
        trc_pst = tf.Variable(np.zeros(np.shape(trace_mask_gen)), dtype=tf.float32, expected_shape=[len(trace_mask_gen)], name='trc_pst')

        trc_msk_gen = tf.Variable(trace_mask_gen, dtype=tf.float32, expected_shape=[len(trace_mask_gen)], name='trc_pst_gen')
        trc_msk_exc = tf.Variable(trace_mask_exc, dtype=tf.float32, expected_shape=[len(trace_mask_gen)],
                                  name='trc_pst_exc')
        trc_msk_inh = tf.Variable(trace_mask_inh, dtype=tf.float32, expected_shape=[len(trace_mask_gen)],
                                  name='trc_pst_inh')

        return ['trc_pre', 'trc_pst', 'trc_msk_gen', 'trc_msk_exc', 'trc_msk_inh'], [trc_pre, trc_pst, trc_msk_gen, trc_msk_exc, trc_msk_inh]




    def update_trace_pre_v0(self,trace_pre,spike_vec):

        '''
        
        :param trace_pre: PRESYNAPTIC TRACE VALUES FROM PREVIOUS ms TIMESTEP
        :param spike_vec: SPIKE VECTOR
        
        :return: UPDATED PRESYNAPTIC TRACE VALUES FOR ALL NEURONS
        '''

        return tf.assign(trace_pre,tf.scalar_mul(self.tau_plus_decay,tf.add(tf.scalar_mul(self.a_plus,spike_vec),trace_pre)))

    def update_trace_post_v0(self,trace_post,spike_vec):

        '''
        
        :param trace_post: POSTSYNAPTIC TRACE VALUES FROM PREVIOUS ms TIMESTEP 
        :param spike_vec: SPIKE VECTOR
        
        :return: UPDATED POSTSYNAPTIC TRACE VALUES FOR ALL NEURONS
        '''

        return tf.assign(trace_post,tf.scalar_mul(self.tau_minus_decay,tf.add(tf.scalar_mul(self.a_minus,spike_vec),trace_post)))

    def update_exc_weights(self,W,spike_vec,trace_pre,trace_post,W_mask,trace_mask_gen,trace_mask_exc):

        '''
        STDP FOR EXCITATORY WEIGHTS

        :param W: NETWORK CONNECTION WEIGHT MATRIX
        :param spike_vec: NEURON SPIKES FROM PREVIOUS ms TIMESTEP
        :param trace_pre: TRACE VALUES OF PRESYNAPTIC NEURONS
        :param trace_post: TRACE VALUES OF POSTSYNAPTIC NEURONS
        :param W_mask: MASK FOR NETWORK CONNECTIONS
        :param trace_mask_gen: MASK FOR ALL CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_EXC: MASK FOR ALL EXCITATORY CONNECTIONS IN RECURRENT NETWORK
        
        :return: W MATRIX WITH UPDATED EXCITATORY WEIGHTS BASED ON TRACE VALUES FROM PREVIOUS ms TIMESTEP 
        '''

        pre_traces = tf.multiply(W_mask,tf.matmul(tf.expand_dims(tf.multiply(trace_mask_gen,spike_vec),axis=1),tf.expand_dims(tf.multiply(trace_mask_exc,trace_pre),axis=0)))

        post_traces = tf.multiply(W_mask,tf.matmul(tf.expand_dims(tf.multiply(trace_mask_gen,trace_post),axis=1),tf.expand_dims(tf.multiply(trace_mask_exc,spike_vec),axis=0)))

        delta_W = tf.subtract(tf.scalar_mul(self.A_plus,pre_traces),tf.scalar_mul(self.A_minus,post_traces))

        return tf.assign(W,tf.multiply(W_mask,tf.add(W,delta_W))),delta_W


    def update_inh_weights(self,W,spike_vec,trace_pre,trace_post,W_mask,trace_mask_gen,trace_mask_inh):

        '''
        STDP FOR INHIBITORY WEIGHTS
        
        :param W: NETWORK CONNECTION WEIGHT MATRIX
        :param spike_vec: NEURON SPIKES FROM PREVIOUS ms TIMESTEP
        :param trace_pre: TRACE VALUE OF PRESYNAPTIC NEURONS
        :param trace_post: TRACE VALUE OF POSTSYNAPTIC NEURONS
        :param W_mask: MASK FOR NETWORK CONNECTIONS
        :param trace_mask_gen: MASK FOR ALL CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_inh: MASK FOR ALL INHIBITORY CONNECTIONS IN RECURRENT NETWORK
        
        :return: W MATRIX WITH UPDATED EXCITATORY WEIGHTS BASED ON TRACE VALUES FROM PREVIOUS ms TIMESTEP
        '''

        pre_traces = tf.multiply(W_mask,tf.matmul(tf.expand_dims(tf.multiply(trace_mask_gen,spike_vec),axis=1),tf.expand_dims(tf.multiply(trace_mask_inh,trace_post),axis=0)))

        post_traces = tf.multiply(W_mask,tf.matmul(tf.expand_dims(tf.multiply(trace_mask_gen,trace_pre),axis=1),tf.expand_dims(tf.multiply(trace_mask_inh,spike_vec),axis=0)))

        delta_W = tf.subtract(tf.scalar_mul(self.A_plus,pre_traces),tf.scalar_mul(self.A_minus,post_traces))

        return tf.assign(W,tf.multiply(W_mask,tf.add(W,delta_W))),delta_W


    def clip_weights(self,W,trace_mask_gen,trace_mask_exc,trace_mask_inh):

        '''
        CLIPS WEIGHTS TO SET RANGE
        
        :param W: NETWORK CONNECTION WEIGHT MATRIX
        :param trace_mask_gen: MASK FOR ALL CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_exc: MASK FOR ALL EXCITATORY CONNECTIONS IN RECURRENT NETWORK
        :param trace_mask_inh: MASK FOR ALL INHIBITORY CONNECTIONS IN RECURRENT NETWORK
        
        :return: NETWOKR WEIGHT MATRIX W WITH CLIPPED WEIGHTS
        '''

        exc_mask = tf.matmul(tf.expand_dims(trace_mask_gen,axis=1),tf.expand_dims(trace_mask_exc,axis=0))
        inh_mask = tf.matmul(tf.expand_dims(trace_mask_gen,axis=1),tf.expand_dims(trace_mask_inh,axis=0))
        oth_mask = tf.abs(tf.subtract(tf.matmul(tf.expand_dims(trace_mask_gen,axis=1),tf.expand_dims(trace_mask_gen,axis=0)),1.0))

        exc_clip = tf.clip_by_value(tf.multiply(W,exc_mask),0.0,self.max_w)
        inh_clip = tf.clip_by_value(tf.multiply(W,inh_mask),self.min_w,0.0)



        return tf.assign(W,tf.add_n([exc_clip,inh_clip,tf.multiply(W,oth_mask)]))
