# BUILT-IN MODULES
import multiprocessing as mp
import time
import os.path
# EXTERNAL MODULES
from PyQt5.QtWidgets import QApplication
import numpy as np
import pickle
# MEDUSA-KERNEL MODULES
from medusa import components
from medusa import meeg
from medusa.bci import cvep_spellers as cvep
# MEDUSA MODULES
import resources, exceptions
import constants as mds_constants
from gui import gui_utils
# APP MODULES
from . import app_controller
from .app_constants import *
from .app_controller import AppController


class App(resources.AppSkeleton):
    """ Class that runs in a separate process to set up the app.

        This class will run in a separate process to represent the MEDUSA server
        side of the application. Its aim is to control the life cycle of the
        developed application, as well as to communicate with the main GUI of
        MEDUSA to print logs. The main() function is going to control life cycle
        by setting up the ``AppController`` (server for communicating with Unity
        clients): initializing the TCP server, opening up the Unity's .exe, and
        communicating with it. As here we do not have a GUI, the Manager thread
        via `manager_thread_worker()` will use the ``AppController`` to send and
        receive messages to and from Unity. This thread will be also devoted to
        process EEG signals, as it has access to all LSL workers.

        In this example, this App will start an Unity application that shows us
        the amount of EEG samples recorded by the LSL. The first thing the
        Unity app will do whenever is ready will be to wait for parameters.
        MEDUSA will send them immediately, according to the `settings.py` file.
        After an acknowledgment from Unity, the application starts by pressing
        the START button. Unity will request us an update with a rate according
        to the parameter `updates_per_min`. Whenever we receive a request,
        MEDUSA is going to answer it by sending the current number of recorded
        samples. Unity will listen for that and update its GUI.

        Attributes
        ----------
        app_controller : AppController
            Controller that helps us to communicate with Unity.
        queue_to_controller : queue.Queue
            Queue used to send messages to ``AppController``.
        queue_from_controller : queue.Queue
            Queue used to receive messages from ``AppController``.
    """

    def __init__(self, app_info, app_settings, medusa_interface,
                 app_state, run_state, working_lsl_streams_info):
        # Call superclass constructor
        super().__init__(app_info, app_settings, medusa_interface,
                         app_state, run_state, working_lsl_streams_info)
        self.eeg_worker_name = self.get_eeg_worker_name(
            working_lsl_streams_info)

        # Set attributes
        self.TAG = '[apps/cvep_speller/main] '
        self.app_controller = None
        # Queues to communicate with the app controller
        self.queue_to_controller = mp.Queue()
        self.queue_from_controller = mp.Queue()
        # Colors
        theme_colors = gui_utils.get_theme_colors('dark')
        self.log_color = theme_colors['THEME_TEXT_ACCENT']

        # Booleans
        self.process_required = False
        self.trainmodel_required = False

        # Load model if available
        self.cvep_model = None
        if self.app_settings.run_settings.mode == ONLINE_MODE:
            try:
                m_path = self.app_settings.run_settings.cvep_model_path
                with open(m_path, 'rb') as handle:
                    self.cvep_model = pickle.load(handle)
            except Exception as ex:
                self.handle_exception(ex)

        # Initialize c-VEP recorded data
        conf, comms = self.get_conf(self.app_settings.run_settings.mode)
        mode_ = 'train' if self.app_settings.run_settings.mode != \
                           ONLINE_MODE else 'test'
        target_ = list()
        for i in range(self.app_settings.run_settings.train_trials):
            target_.append([0, 0, 0])
        self.cvep_data = cvep.CVEPSpellerData(
            mode=mode_,
            paradigm_conf=conf,
            commands_info=comms,
            onsets=np.zeros((0,)),
            command_idx=np.zeros((0,)),
            unit_idx=np.zeros((0,)),
            level_idx=np.zeros((0,)),
            matrix_idx=np.zeros((0,)),
            cycle_idx=np.zeros((0,)),
            trial_idx=np.zeros((0,)),
            cvep_model=None,
            spell_result=[],
            fps_resolution=self.app_settings.run_settings.fps_resolution,
            spell_target=target_
        )

    def handle_exception(self, ex):
        if not isinstance(ex, exceptions.MedusaException):
            raise ValueError('Unhandled exception')
        if isinstance(ex, exceptions.MedusaException):
            # Take actions
            if ex.importance == 'critical':
                self.close_app(force=True)
                ex.set_handled(True)

    # ---------------------------- LSL transponder ----------------------------
    def check_lsl_config(self, working_lsl_streams_info):
        # Check if the have only one EEG
        count = 0
        for lsl_info in working_lsl_streams_info:
            if lsl_info['lsl_type'] == 'EEG':
                count += 1
        if count == 1:
            return True
        else:
            return False

    def check_settings_config(self, app_settings):
        """Check settings config.
        By default, this function check if unity path exits"""

        if not os.path.exists(app_settings.connection_settings.path_to_exe):
            raise exceptions.IncorrectSettingsConfig(
                "Incorrect path of Unity file: " +
                app_settings.connection_settings.path_to_exe)
        if app_settings.run_settings.mode == ONLINE_MODE:
            # Check if we are on online and no model is specified
            if app_settings.run_settings.cvep_model_path == '':
                raise exceptions.IncorrectSettingsConfig(
                    "Cannot run ONLINE mode if c-VEP model is missing")
            # Check if the model has been trained with the same sequence
            curr_seq = tuple(app_settings.matrices['train'][0].item_list[
                0].sequence)
            with open(app_settings.run_settings.cvep_model_path, 'rb') as h:
                cvep_model = pickle.load(h)
            trained_seq = list(cvep_model.methods['clf_method']['instance'].
                               fitted['sequences'].keys())[0]
            if curr_seq != trained_seq:
                raise exceptions.IncorrectSettingsConfig(
                    "It seems that the model (%s) has been trained using a "
                    "different sequence! Please, train a new model using the "
                    "desired sequence" %
                    app_settings.run_settings.cvep_model_path)




    def get_eeg_worker_name(self, working_lsl_streams_info):
        for lsl_info in working_lsl_streams_info:
            if lsl_info['lsl_type'] == 'EEG':
                return lsl_info['lsl_name']
        return None

    def get_lsl_worker(self):
        """Returns the LSL worker"""
        return self.lsl_workers[self.eeg_worker_name]

    # ---------------------------- LOG ----------------------------
    def send_to_log(self, msg):
        """ Styles a message to be sent to the main MEDUSA log. """
        self.medusa_interface.log(
            msg, {'color': self.log_color, 'font-style': 'italic'})

    # ---------------------------- MANAGER THREAD ----------------------------
    def manager_thread_worker(self):
        """ Manager thread worker that controls the application flow.

            To set up correctly the communication between MEDUSA and Unity, it
            is required to initialize things correctly. First, it waits MEDUSA
            to be ready by checking `run_state`. Then, it waits until the main()
            function instantiates the ``AppController``, and afterward initiates
            the server by calling `app_controller.start_server()`. In parallel,
            the main() function is opening up the Unity's application, so this
            thread waits until it is up. When it is up, then it sends the
            required parameters to Unity via the ``AppController`` and waits
            until Unity confirms us that everything is ready. When user presses
            the START button, it sends a `play` command to Unity via the
            ``AppController``. The rest of the code is intended to listen for
            pause and stop events to notify Unity about them.
        """
        TAG = '[apps/cvep_speller/App/manager_thread_worker]'

        # Function to close everything
        def close_everything():
            # Notify Unity that it must stop
            self.app_controller.stop()  # Send the stop signal to unity
            print(TAG, 'Close signal emitted to Unity.')

            # Wait until the Unity server notify us that the app is closed
            while self.app_controller.unity_state.value != UNITY_DOWN:
                time.sleep(0.1)
            print(TAG, 'Unity application closed!')

            # Close the main app and exit the loop
            self.stop = True
            self.close_app()

        # Wait until MEDUSA is ready
        print(TAG, "Waiting MEDUSA to be ready...")
        while self.run_state.value != mds_constants.RUN_STATE_READY:
            time.sleep(0.1)

        # Wait until the app_controller is initialized
        while self.app_controller is None: time.sleep(0.1)

        # Set up the TCP server and wait for the Unity client
        self.send_to_log('Setting up the TCP server...')
        self.app_controller.start_server()

        # Wait until UNITY is UP and send the parameters
        while self.app_controller.unity_state.value == UNITY_DOWN:
            time.sleep(0.1)
        self.app_controller.send_parameters()

        # Wait until UNITY is ready
        while self.app_controller.unity_state.value == UNITY_UP:
            time.sleep(0.1)
        self.send_to_log('Unity is ready to start')

        # If play is pressed
        while self.run_state.value == mds_constants.RUN_STATE_READY:
            time.sleep(0.1)
        if self.run_state.value == mds_constants.RUN_STATE_RUNNING:
            self.app_controller.play()

        # Check for an early stop
        if self.run_state.value == mds_constants.RUN_STATE_STOP:
            close_everything()

        # Loop
        while not self.stop:
            # Check for pause
            if self.run_state.value == mds_constants.RUN_STATE_PAUSED:
                self.app_controller.pause()
                while self.run_state.value == mds_constants.RUN_STATE_PAUSED:
                    time.sleep(0.1)
                # If resumed
                if self.run_state.value == mds_constants.RUN_STATE_RUNNING:
                    self.app_controller.resume()

            # Check for stop
            if self.run_state.value == mds_constants.RUN_STATE_STOP:
                close_everything()

            # Processing event
            if self.process_required:
                if self.cvep_model is None:
                    raise Exception('[cvep_speller] Cannot process the trial '
                                    'if the model has not been trained before!')
                # We need to wait until the signal from the last onset is
                # enough to extract the full epoch
                if not self.cvep_model.check_predict_feasibility(
                        self.get_current_dataset()):
                    print('[cvep_speller] Epoch length is not enough, '
                              'waiting for more samples...')
                else:
                    self.process_required = False
                    decoding = self.process_trial()

                    # Notify UNITY about the selected character
                    # todo: matrix, level, unit etc
                    # Aclaration:
                    # 1) [-1] to access the last cycle (max no. cycles)
                    # 2) [-1] to access the last and unique training seq
                    # 3) ['sorted_cmds'] to access the commands sorted by
                    # their probability of being selected
                    # 4) [0] to get the most probable command
                    # 5) ['coords'][0] to get the matrix index
                    #    ['item']['row'] to get the row inside the matrix
                    #    ['item']['col'] to get the col inside the matrix
                    coords_ = [
                        decoding['items_by_no_cycle'][-1][-1][
                            'sorted_cmds'][0]['coords'][0],
                        decoding['items_by_no_cycle'][-1][-1][
                                'sorted_cmds'][0]['item']['row'],
                        decoding['items_by_no_cycle'][-1][-1][
                                'sorted_cmds'][0]['item']['col']
                    ]
                    self.app_controller.notify_selection(
                            selection_coords=coords_,
                            selection_label=decoding['spell_result'][0]
                    )
        print(TAG, 'Terminated')

    def process_event(self, dict_event):
        """ Process any interesting event.

            These events may be called by the `manager_thread_worker` whenever
            Unity requests any kind-of processing. As we do not have any MEDUSA
            GUI, this function call be also directly called by the instance of
            ``AppController`` if necessary.

            In this case, the possible events, encoded in 'event_type' are:
                - 'request_samples': Unity requires us to send the current
                registered samples of the LSL stream.
                - 'close': Unity said it has been closed, so we need to close
                everything.
        """
        # self.send_to_log('Message from Unity: %s' % str(event))
        if dict_event["event_type"] == "train" or \
                dict_event["event_type"] == "test":
            # Onset information. E.g.: msg = {"event_type":"train",
            # "target":"C", "cycle":0,"onset":5393}
            self.append_trial_info(dict_event)
        elif dict_event["event_type"] == "processPlease":
            # Unity is requesting MEDUSA to process the previous trial
            self.process_required = True
        else:
            print(self.TAG, 'Unknown event_type %s' % dict_event["event_type"])

    # ---------------------------- MAIN PROCESS ----------------------------
    def main(self):
        """ Controls the main life cycle of the ``App`` class.

            First, changes the app state to powering on and sets up the
            ``AppController`` instance. Then, changes the app state to on. It
            waits until the TCP Server instantiated by the ``AppController`` is
            up, and afterward tells the ``AppController`` to open the Unity's
            .exe application, which is a blocking process. When the application
            is closed, this function changes the app state to powering off and
            shows a dialog to save the file (only if we have data available).
            Finally, it changes the app state to off and dies.
        """
        # 1 - Change app state to powering on
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_POWERING_ON)
        # 2 - Set up the controller that starts the TCP server
        self.app_controller = app_controller.AppController(
            callback=self,
            app_settings=self.app_settings,
            run_state=self.run_state)
        # 3 - Change app state to power on
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_ON)
        # 4 - Wait until server is UP, start the unity app and block the
        # execution until it is closed
        while self.app_controller.server_state.value == SERVER_DOWN:
            time.sleep(0.1)
        try:
            self.app_controller.start_application()
        except Exception as ex:
            self.handle_exception(ex)
            self.medusa_interface.error(ex)
        # while self.app_controller: time.sleep(1)  # For debugging Unity
        # 5 - Change app state to powering off
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_POWERING_OFF)
        # 6 - Save recording
        self.stop_working_threads()
        if self.get_lsl_worker().data.shape[0] > 0:
            qt_app = QApplication([])
            self.save_file_dialog = resources.SaveFileDialog(
                self.app_info['extension'])
            self.save_file_dialog.accepted.connect(self.on_save_rec_accepted)
            self.save_file_dialog.rejected.connect(self.on_save_rec_rejected)
            qt_app.exec()
        else:
            print(self.TAG, 'Cannot save because we have no data!!')
        # 7 - Change app state to power off
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_OFF)

    def close_app(self, force=False):
        """ Closes the ``AppController`` and working threads.
        """
        # Trigger the close event in the AppController. Returns True if
        # closed correctly, and False otherwise. If everything was
        # correct, stop the working threads
        if self.app_controller.close():
            self.stop_working_threads()
        self.app_controller = None

    # ---------------------------- SAVE DATA ----------------------------
    @exceptions.error_handler(scope='app')
    def on_save_rec_accepted(self):
        file_info = self.save_file_dialog.get_file_info()
        rec = self.get_current_recording(file_info)
        rec.save(file_info['path'])
        # Print a message
        self.medusa_interface.log('Recording saved successfully')
        # Close the app
        self.close()

    @exceptions.error_handler(scope='app')
    def on_save_rec_rejected(self):
        self.close()

    @exceptions.error_handler(scope='app')
    def get_eeg_data(self):
        # EEG data
        lsl_worker = self.get_lsl_worker()
        channels = meeg.EEGChannelSet()
        channels.set_standard_montage(lsl_worker.receiver.l_cha)
        times_, signal_ = lsl_worker.get_data()
        if times_.shape[0] != signal_.shape[0]:
            min_len = min(times_.shape[0], signal_.shape[0])
            print('[get_current_recording] Warning! timestamps (%i) and '
                  'signal (%i) did not have the same dimensions, trimmed '
                  'both to have %i samples.' % (times_.shape[0],
                                                signal_.shape[0],
                                                min_len)
                  )
            times_ = times_[:min_len]
            signal_ = signal_[:min_len, :]
        return times_, signal_, lsl_worker.receiver.fs, channels, \
               lsl_worker.receiver.name

    @exceptions.error_handler(scope='app')
    def get_current_recording(self, file_info=None):
        # EEG data
        times_, signal_, fs, channels, equip = self.get_eeg_data()
        eeg = meeg.EEG(times_, signal_, fs, channels, equipement=equip)
        # Recording
        subject_id = file_info['subject_id'] if file_info is not None else ''
        recording_id = file_info['recording_id'] if file_info is not None else ''
        description = file_info['description'] if file_info is not None else ''
        rec = components.Recording(
            subject_id=subject_id,
            recording_id=recording_id,
            description=description,
            date=time.strftime("%d-%m-%Y %H:%M", time.localtime())
        )
        rec.add_biosignal(eeg)
        rec.add_experiment_data(self.cvep_data)
        return rec

    @exceptions.error_handler(scope='app')
    def get_current_dataset(self):
        try:
            rec = self.get_current_recording()
            dataset = cvep.CVEPSpellerDataset(
                channel_set=rec.eeg.channel_set,
                fs=rec.eeg.fs
            )
            dataset.add_recordings(rec)
        except Exception as e:
            self.handle_exception(e)
            return None
        return dataset

    # ---------------------------- PROCESSING ----------------------------
    def append_trial_info(self, msg):
        # Common trial info
        self.cvep_data.cycle_idx = np.append(
            self.cvep_data.cycle_idx, msg["cycle"])
        self.cvep_data.onsets = np.append(
            self.cvep_data.onsets, msg["onset"])
        self.cvep_data.trial_idx = np.append(
            self.cvep_data.trial_idx, msg["trial"])
        self.cvep_data.matrix_idx = np.append(
            self.cvep_data.matrix_idx, msg["matrix_idx"])
        self.cvep_data.level_idx = np.append(
            self.cvep_data.level_idx, msg["level_idx"])
        self.cvep_data.unit_idx = np.append(
            self.cvep_data.unit_idx, msg["unit_idx"])

        # Information only present in Train mode
        if "command_idx" in msg:
            self.cvep_data.command_idx = np.append(
                self.cvep_data.command_idx, msg["command_idx"])

    def process_trial(self):
        """ This function processes only the last trial to get the selected
        command. Note that this method is not called in TRAIN_MODE.
        """
        if self.cvep_model is None:
            self.handle_exception(Exception('[cvep_speller] Cannot process the '
                                            'trial if the model has not been'
                                            ' trained before!'))

        # Get current data
        last_idx = self.cvep_data.trial_idx[-1]
        times_, signal_, fs, channels, equip = self.get_eeg_data()
        eeg = meeg.EEG(times_, signal_, fs, channels, equipement=equip)

        # Process the last trial
        decoding = self.cvep_model.predict(times=times_, signal=signal_,
                                           trial_idx=last_idx,
                                           exp_data=self.cvep_data,
                                           sig_data=eeg)
        return decoding

    def get_conf(self, mode):
        # TODO: nested matrices (units) are not implemented yet
        cvep_conf = []
        cvep_comms = []

        matrix_type = 'test' if mode != TRAIN_MODE else 'train'
        prev_idx = 0
        for m in self.app_settings.matrices[matrix_type]:
            # Matrix configuration
            no_comms = len(m.item_list)
            comms_list = list(range(prev_idx, prev_idx + no_comms))
            comms_list = list(map(str, comms_list))
            m_conf = [  # Units
                # Commands
                [
                    comms_list
                ]
            ]
            cvep_conf.append(m_conf)
            prev_idx = no_comms

            # Commands info
            matrix_comms = {}
            for idx, item in enumerate(m.item_list):
                matrix_comms[comms_list[idx]] = item.to_dict()
            cvep_comms.append(matrix_comms)
        return cvep_conf, cvep_comms
