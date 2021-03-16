import os
import multiprocessing as mp
import NAN_support_lib as sup


class multi_process:
    def __init__(self,num_processes,save_path):

        self.main_Path = os.getcwd()
        self.network_Path = self.main_Path + '/networks/'
        self.data_Path = save_path

        print('Main working directory: ' + str(self.main_Path))
        print('Networks directory: ' + str(self.network_Path))
        print('Data directory: ' + str(self.data_Path))

        # Define IPC managers
        manager1 = mp.Manager()

        # Define lists (queue) for tasks and computation results
        self.data_feed1 = manager1.Queue()

        self.status1 = manager1.Queue()

        self.processes1 = []


        # activate full branch saving processes
        for i in range(num_processes):
            # Set process name
            process_name = 'Pb%i' % i

            # Create the process, and connect it to the worker function
            new_process = mp.Process(target=self.saveData, args=(process_name, self.data_feed1, self.status1))

            # Add new process to the list of processes
            self.processes1.append(new_process)

            # Start the process
            new_process.start()


    def save_data(self,signal, names, data, save_filename):

        self.data_feed1.put([signal, names, data, save_filename])


    #### Saving functions ####
    def saveData(self, process_name, data_feed, status):

        '''
                    data_feed data format
                     process_command, interval_sec_START, interval_sec_END, t_ms_window,w

        '''

        print('[%s] SAVE PROCESS launched, waiting for data' % process_name)

        while True:
            data = data_feed.get()


            if data[0] == -1:
                print('[%s] SAVE PROCESS terminated' % process_name)
                status.put(1)
                break
            else:

                saveNames = data[1]
                saveData = data[2]
                save_filename = data[3]

                # SAVES DATA
                sup.save_non_tf_data(saveNames, saveData, filename=save_filename, savePath=self.data_Path)

                print('DATA_SAVED_to_'+str(save_filename))

                status.put(save_filename)


    def kill_workers(self, process_count):
        for i in range(0, process_count):
            print('KILL SWITCH SENT FOR PROCESS ' + str(i))
            self.save_data(signal=-1,names=[0],data=[0],save_filename='')

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