import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
import os
import NAN_support_lib as sup




class plotter:
    def __init__(self,VER, NET, SP_MIN, SP_MAX, SIM_DUR, LINEWIDTH):
        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = self.main_Path + '/dataFiles/ver_'+ str(VER) +'/'

        self.VER = VER
        self.NET = NET
        self.SP_MIN = SP_MIN
        self.SP_MAX = SP_MAX
        self.SIM_DUR = SIM_DUR
        self.BF_W = 9
        self.AVA_W = 9

        self.NEURON_POP = 0

        self.LINEWIDTH = LINEWIDTH


    def moving_average(self,a, n=3):
        ret = np.nancumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def moving_average_nan(self,a,a_nan_one, n=3):
        ret = np.nancumsum(a, dtype=float)
        ret_nans = np.cumsum(a_nan_one, dtype=float)

        ret[n:] = ret[n:] - ret[:-n]
        ret_nans[n:] = ret_nans[n:] - ret_nans[:-n]

        return ret[n - 1:] / (n-ret_nans[n - 1:])

    def BF_TM_PLOT(self):

        VER = self.VER
        NET = self.NET
        SP_MIN = self.SP_MIN
        SP_MAX = self.SP_MAX
        SIM_DUR = self.SIM_DUR
        BF_W = self.BF_W
        NEURON_POP = self.NEURON_POP

        t_ref_inp_list = []
        save_sec_range_global = []
        t_ms_list = []
        bf_ms_list = []
        bf_sec_list = []
        t_sec_list = []
        for i in range(SP_MIN, SP_MAX):
            print('Opening File: ' + str(i) + '/' + str(SP_MAX))

            file_bf = 'BF_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_DT_' + str(
                BF_W) + '_sp_' + str(i)
            names_bf, data_bf = sup.unpack_file(filename=file_bf, dataPath=self.data_Path)
            file_add = 'Sim_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_sp_' + str(
                i) + '_additional'
            names_add, data_add = sup.unpack_file(filename=file_add, dataPath=self.data_Path)

            t_ref_inp_list.extend(data_add[names_add.index('t_ref_inp_list')])
            save_sec_range = data_add[names_add.index('save_sec_range')]

            t_ms = data_bf[names_bf.index('t_series_list')]
            t_ms_f = np.add((save_sec_range[0] * 1000), t_ms[NEURON_POP])
            t_ms_list.extend(t_ms_f)
            bf_ms_list.extend(data_bf[names_bf.index('network_BF_per_t_list')][NEURON_POP])
            bf_sec_list.extend(data_bf[names_bf.index('BF_ave_per_sec_list')][NEURON_POP])
            t_sec = data_bf[names_bf.index('t_sec')]
            t_sec_f = np.add((save_sec_range[0]), t_sec[NEURON_POP])
            t_sec_list.extend(t_sec_f)

            if i == SP_MIN:
                save_sec_range_global.append(save_sec_range[0])
            elif i == (SP_MAX - 1):
                save_sec_range_global.append(save_sec_range[1])

        t_sec_arr = np.asarray(t_sec_list)
        bf_sec_arr = np.asarray(bf_sec_list)

        return t_sec_arr,bf_sec_arr


    def SR_TM_PLOT(self):

        VER = self.VER
        NET = self.NET
        SP_MIN = self.SP_MIN
        SP_MAX = self.SP_MAX
        SIM_DUR = self.SIM_DUR

        t_ref_inp_list = []
        save_sec_range_global = []
        sr_sec_list = []
        for i in range(SP_MIN, SP_MAX):
            print('Opening File: ' + str(i) + '/' + str(SP_MAX))

            file_add = 'Sim_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_sp_' + str(
                i) + '_additional'
            names_add, data_add = sup.unpack_file(filename=file_add, dataPath=self.data_Path)

            t_ref_inp_list.extend(data_add[names_add.index('t_ref_inp_list')])
            save_sec_range = data_add[names_add.index('save_sec_range')]

            sr_sec = data_add[names_add.index('SR_res_list')]
            sr_sec_list.extend(sr_sec)

            if i == SP_MIN:
                save_sec_range_global.append(save_sec_range[0])
            elif i == (SP_MAX - 1):
                save_sec_range_global.append(save_sec_range[1])

        t_sec_arr = np.asarray((self.SP_MIN * 100)+1+np.arange(len(sr_sec_list)))
        sr_sec_arr = np.asarray(sr_sec_list)

        return t_sec_arr,sr_sec_arr

    def W_TM_PLOT(self):

        VER = self.VER
        NET = self.NET
        SP_MIN = self.SP_MIN
        SP_MAX = self.SP_MAX
        SIM_DUR = self.SIM_DUR

        t_ref_inp_list = []
        save_sec_range_global = []
        w_exc_mean_list = []
        w_inh_mean_list = []
        for i in range(SP_MIN, SP_MAX):
            print('Opening File: ' + str(i) + '/' + str(SP_MAX))

            file_add = 'Sim_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_sp_' + str(
                i) + '_additional'
            names_add, data_add = sup.unpack_file(filename=file_add, dataPath=self.data_Path)

            t_ref_inp_list.extend(data_add[names_add.index('t_ref_inp_list')])
            save_sec_range = data_add[names_add.index('save_sec_range')]
            w_exc_mean_list.extend(data_add[names_add.index('w_exc_mean_list')])
            w_inh_mean_list.extend(data_add[names_add.index('w_inh_mean_list')])

            if i == SP_MIN:
                save_sec_range_global.append(save_sec_range[0])
            elif i == (SP_MAX - 1):
                save_sec_range_global.append(save_sec_range[1])

        t_sec_arr = np.asarray((self.SP_MIN * 100) + 1 + np.arange(len(w_exc_mean_list)))
        w_exc_mean_arr = np.asarray(w_exc_mean_list)
        w_inh_mean_arr = np.absolute(np.asarray(w_inh_mean_list))

        return t_sec_arr, w_exc_mean_arr, w_inh_mean_arr

    def CA_TM_PLOT(self):

        VER = self.VER
        NET = self.NET
        SP_MIN = self.SP_MIN
        SP_MAX = self.SP_MAX
        SIM_DUR = self.SIM_DUR

        t_ref_inp_list = []
        save_sec_range_global = []
        ca_list = []
        for i in range(SP_MIN, SP_MAX):
            print('Opening File: ' + str(i) + '/' + str(SP_MAX))

            file_add = 'Sim_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_sp_' + str(
                i) + '_additional'
            names_add, data_add = sup.unpack_file(filename=file_add, dataPath=self.data_Path)

            t_ref_inp_list.extend(data_add[names_add.index('t_ref_inp_list')])
            save_sec_range = data_add[names_add.index('save_sec_range')]
            ca_list.extend(data_add[names_add.index('ca_list')])

            if i == SP_MIN:
                save_sec_range_global.append(save_sec_range[0])
            elif i == (SP_MAX - 1):
                save_sec_range_global.append(save_sec_range[1])

        ca_ms_arr = np.asarray(ca_list)
        ca_sec_arr = np.squeeze(np.average(ca_ms_arr.reshape((-1,1000)),axis=1))
        t_sec_arr = np.asarray(save_sec_range_global[0] + np.arange(np.size(ca_sec_arr)))

        return t_sec_arr, ca_sec_arr


    def AVA_TM_PLOT(self):

        VER = self.VER
        NET = self.NET
        SP_MIN = self.SP_MIN
        SP_MAX = self.SP_MAX
        SIM_DUR = self.SIM_DUR
        DT = self.AVA_W

        time_sec_l = []
        alpha_l = []
        for i in range(SP_MIN, SP_MAX):
            print('Opening File: ' + str(i) + '/' + str(SP_MAX))

            file_name = 'AVA_Data_for_Net_' + str(NET) + '_ver_' + str(VER) + '_' + str(SIM_DUR) + '_DT_' + str(DT) + '_sp_' + str(i)
            names, data = sup.unpack_file(filename=file_name, dataPath=self.data_Path)

            time_sec_l.extend(data[names.index('time_sec_l')])
            alpha_l.extend(data[names.index('alpha_l')])

        return time_sec_l, alpha_l

    def plot_all_data(self, BF_t_sec_arr, BF_bf_sec_arr, SR_t_sec_arr, SR_sr_sec_arr, W_t_sec_arr, W_w_exc_mean_arr, W_w_inh_mean_arr, CA_t_sec_arr, CA_ca_sec_arr, AVA_t_sec_arr, AVA_alpha_arr):

        fig = plt.figure(figsize=(18,9))
        ax0 = fig.add_subplot(611)
        ax1 = fig.add_subplot(612, sharex=ax0)
        ax2 = fig.add_subplot(613, sharex=ax0)
        ax3 = fig.add_subplot(614, sharex=ax0)
        ax4 = fig.add_subplot(615, sharex=ax0)
        ax5 = fig.add_subplot(616, sharex=ax0)

        ax0.plot(BF_t_sec_arr, BF_bf_sec_arr, c='k', linewidth=self.LINEWIDTH)
        ax1.plot(SR_t_sec_arr, SR_sr_sec_arr, c='k', linewidth=self.LINEWIDTH)
        ax2.plot(W_t_sec_arr, W_w_exc_mean_arr, c='b', linewidth=self.LINEWIDTH)
        ax3.plot(W_t_sec_arr, W_w_inh_mean_arr, c='g', linewidth=self.LINEWIDTH)
        ax4.plot(CA_t_sec_arr, CA_ca_sec_arr, c='k', linewidth=self.LINEWIDTH)
        ax5.plot(AVA_t_sec_arr, AVA_alpha_arr, c='k', linewidth=self.LINEWIDTH)

        fig.suptitle('VER: '+str(self.VER)+'_NET: '+str(self.NET)+'_sp: '+str(self.SP_MIN)+'-'+str(self.SP_MAX))
        ax0.set_ylabel('Branching\nFactor')
        ax1.set_ylabel('Average\nNeuronal\nSpiking Rate\n (Hz)')
        ax2.set_ylabel('Average\nExcitatory\nWeight')
        ax3.set_ylabel('Average\nAbsolute\nInhibitory\nWeight')
        ax4.set_ylabel('Astrocyte\nCalcium\nConcentration')
        ax5.set_ylabel('Avalanche\nDistribution\nScaling\nExponent')
        ax5.set_xlabel('Time (Seconds)')

        fig_name = 'NAN_Simulation_Output.png'
        fig_fn = os.path.abspath(os.path.join(self.data_Path, fig_name))
        fig.savefig(fig_fn)

        # plt.show()



def main(argv):
    try:
        opts, args = getopt.getopt(argv, "", ["ver_num=", "net_num=", "sim_run_time_s=", "start_sp=", "end_sp="])
    except getopt.GetoptError:
        print('Incorrect arguments')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--ver_num':
            VER_IN = int(arg)

        elif opt == '--net_num':
            NET_IN = int(arg)

        elif opt == '--sim_run_time_s':
            times_end = int(arg)
            times_strt = 0

        elif opt == '--start_sp':
            SP_MIN_IN = int(arg)

        elif opt == '--end_sp':
            SP_MAX_IN = int(arg)

        else:
            print('Error, exiting')
            sys.exit()

    print('SIM_VER: ' + str(VER_IN))
    print('NETWORK_NUMBER: ' + str(NET_IN))
    print('FULL_SIM_TIME_RANGE: ' + str(times_strt)+'_' + str(times_end))
    print('SP_RANGE: ' + str(SP_MIN_IN)+'_' + str(SP_MAX_IN))

    LINEWIDTH_IN = 1.5

    SIM_DUR_IN = str(times_strt)+str(times_end)


    fp = plotter(VER=VER_IN,NET=NET_IN,SP_MIN=SP_MIN_IN,SP_MAX=SP_MAX_IN,SIM_DUR=SIM_DUR_IN, LINEWIDTH=LINEWIDTH_IN)

    bf_x, bf_y = fp.BF_TM_PLOT()
    sr_x, sr_y = fp.SR_TM_PLOT()
    w_x, w_y_exc, w_y_inh = fp.W_TM_PLOT()
    ca_x, ca_y = fp.CA_TM_PLOT()
    ava_x, ava_y = fp.AVA_TM_PLOT()

    fp.plot_all_data(
           BF_t_sec_arr = bf_x
         , BF_bf_sec_arr = bf_y
         , SR_t_sec_arr = sr_x
         , SR_sr_sec_arr = sr_y
         , W_t_sec_arr = w_x
         , W_w_exc_mean_arr = w_y_exc
         , W_w_inh_mean_arr = w_y_inh
         , CA_t_sec_arr = ca_x
         , CA_ca_sec_arr = ca_y
         , AVA_t_sec_arr=ava_x
         , AVA_alpha_arr=ava_y
                     )

if __name__ == '__main__':
    main(sys.argv[1:])




