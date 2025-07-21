# BUILT-IN MODULES
import multiprocessing as mp
import time
import os.path
# EXTERNAL MODULES
from PySide6.QtWidgets import QApplication
import numpy as np
import pickle
# MEDUSA-KERNEL MODULES
from medusa import components
from medusa import meeg, emg, nirs, ecg
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
    """ Main class of the application. For detailed comments about all
        functions, see the superclass code in resources module."""
    def __init__(self, app_info, app_settings, medusa_interface,
                 app_state, run_state, working_lsl_streams_info,
                 rec_info):
        # Call superclass constructor
        super().__init__(app_info, app_settings, medusa_interface,
                         app_state, run_state, working_lsl_streams_info,
                         rec_info)

        # Set attributes
        self.TAG = '[apps/cvep_speller/main] '
        self.app_controller = None
        self.app_name = app_info["name"]
        # Queues to communicate with the app controller
        self.queue_to_controller = mp.Queue()
        self.queue_from_controller = mp.Queue()
        # Colors
        theme_colors = gui_utils.get_theme_colors('dark')
        self.log_color = theme_colors['THEME_TEXT_ACCENT']

        # Find EEG
        self.eeg_worker_name = self.get_eeg_worker_name(
            working_lsl_streams_info)

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

        target_coords = self.app_settings.encoding_settings.get_coords_from_labels(
            self.app_settings.run_settings.train_target, self.app_settings.encoding_settings.matrices)
        n_col = self.app_settings.encoding_settings.matrices[0].n_col
        target_idx = [n_col*target[1] + target[2] for target in target_coords]
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
            spell_target=target_idx
        )

        # Debugging?
        self.is_debugging = False

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
        count = 0
        for lsl_info in working_lsl_streams_info:
            if lsl_info['lsl_type'] == 'EEG':
                count += 1
                # Check if labels are correct
                ch_set = meeg.EEGChannelSet()
                try:
                    ch_set.set_standard_montage(l_cha=lsl_info['l_cha'])
                except meeg.ChannelNotFound as e:
                    raise exceptions.IncorrectLSLConfig(
                    "It seems that channel labels are not present in the LSL "
                    "stream. This application requires standard labels to use "
                    "topographic coordinates and perform artifact rejection. "
                    "Original exception: %s" % str(e)
                )
        if count == 0:
            raise exceptions.IncorrectLSLConfig(
                "No EEG stream detected. This application requires an EEG "
                "stream. Please check the LSL configuration."
            )

    def check_settings_config(self, app_settings):
        """Check settings config.
        By default, this function check if unity path exits"""

        def is_shifted_version(seq1, seq2):
            if len(seq1) != len(seq2):
                return False
            for j in range(len(seq1)):
                if np.all(np.array(seq1) == np.roll(seq2, -j)):
                    return True
            return False

        if app_settings.run_settings.mode == ONLINE_MODE:
            # Check if we are on online and no model is specified
            if app_settings.run_settings.cvep_model_path == '':
                raise exceptions.IncorrectSettingsConfig(
                    "Cannot run ONLINE mode if c-VEP model is missing")
            # Check if the model has been trained with the same sequence
            curr_seq = tuple(app_settings.encoding_settings.matrices[0].item_list[
                0].sequence)
            with open(app_settings.run_settings.cvep_model_path, 'rb') as h:
                cvep_model = pickle.load(h)
            if isinstance(cvep_model, cvep.CVEPModelCircularShifting):
                trained_seq = list(cvep_model.methods['clf_method']['instance'].
                                   fitted['sequences'].keys())[0]
                if curr_seq != trained_seq and not \
                        is_shifted_version(curr_seq, trained_seq):
                    raise exceptions.IncorrectSettingsConfig(
                        "It seems that the model (%s) has been trained using a "
                        "different sequence! Please, train a new model using "
                        "the desired sequence" %
                        app_settings.run_settings.cvep_model_path)
            elif isinstance(cvep_model, (cvep.CMDModelBWRLDA,
                                        cvep.CMDModelBWREEGInception,
                                        cvep.CVEPModelBWRRiemannianLDA)):
                pass
            else:
                raise exceptions.IncorrectSettingsConfig('Unknown model type!')

    def get_eeg_worker_name(self, working_lsl_streams_info):
        for lsl_info in working_lsl_streams_info:
            if lsl_info['medusa_type'] == 'EEG':
                self.send_to_log('Linked to EEG stream: %s' % lsl_info['lsl_name'])
                return lsl_info['lsl_name']
        raise Exception('[cvep_speller] Cannot find any EEG stream on LSL!')

    def get_lsl_worker(self):
        """Returns the LSL worker"""
        return self.lsl_workers[self.eeg_worker_name]

    # ---------------------------- LOG ----------------------------
    def send_to_log(self, msg):
        """ Styles a message to be sent to the main MEDUSA log."""
        self.medusa_interface.log(
            msg, {'color': self.log_color, 'font-style': 'italic'})

    # ---------------------------- MANAGER THREAD ----------------------------
    def manager_thread_worker(self):
        TAG = '[apps/cvep_speller/App/manager_thread_worker] '

        # Function to close everything
        def close_everything():
            # Notify Unity that it must stop
            self.app_controller.stop()  # Send the stop signal to unity
            print(TAG, 'Close signal emitted to Unity.')

            # Wait until the Unity server notify us that the app is closed
            while self.app_controller.unity_state.value != UNITY_FINISHED:
                pass
            print(TAG, 'Unity application closed!')

            # Exit the loop
            self.stop = True

        # Wait until MEDUSA is ready
        print(TAG, "Waiting MEDUSA to be ready...")
        while self.run_state.value != mds_constants.RUN_STATE_READY:
            time.sleep(0.1)

        # Wait until the app_controller is initialized
        while self.app_controller is None: time.sleep(0.1)

        # Set up the TCP server and wait for the Unity client
        self.app_controller.start_server()
        self.send_to_log(f'[{self.app_name}] TCP server listening!')

        # Wait until UNITY is UP and send the parameters
        while self.app_controller.unity_state.value == UNITY_DOWN:
            time.sleep(0.1)
        self.app_controller.send_parameters()

        # Wait until UNITY is ready
        while self.app_controller.unity_state.value == UNITY_UP:
            time.sleep(0.1)
        self.send_to_log(f'[{self.app_name}] Unity is ready to start')

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
                fps= self.cvep_data.fps_resolution
                seq_len = len(self.app_settings.encoding_settings.matrices[0].item_list[0].sequence)
                onsets = self.cvep_data.onsets
                times_, signal_, fs, channels, equip = self.get_eeg_data()
                # We need to wait until the signal from the last onset is
                # enough to extract the full epoch
                if isinstance(self.cvep_model, cvep.CVEPModelCircularShifting) \
                        and not self.cvep_model.check_predict_feasibility_signal(
                       times_, onsets, fs):
                    print('[cvep_speller] Epoch length is not enough, '
                              'waiting for more samples...')
                elif isinstance(self.cvep_model, (cvep.CVEPModelBWRRiemannianLDA,
                                                  cvep.CMDModelBWREEGInception,
                                                  cvep.CMDModelBWRLDA)) \
                        and not self.cvep_model.check_predict_feasibility_signal(
                        times_, onsets, fps, seq_len, fs):
                    print('[cvep_speller] Epoch length is not enough, '
                              'waiting for more samples...')
                else:
                    self.process_required = False
                    decoding = self.process_trial()
                    self.app_controller.notify_selection(
                            selection_coords=decoding['coords'],
                            selection_label=decoding['cmd_label']
                    )
        print(TAG, 'Terminated')

    def process_event(self, dict_event):
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
        if self.is_debugging:
            # When debugging
            while self.app_controller:
                time.sleep(1)
        else:
            try:
                # Start application (blocking method)
                self.app_controller.start_application()
            except Exception as ex:
                self.handle_exception(ex)
                self.medusa_interface.error(ex)
        # 5 - Close (only if close app has not been called yet)
        if self.app_controller.server_state.value != SERVER_DOWN:
            self.app_controller.close()
        while self.app_controller.server_state.value == SERVER_UP:
            time.sleep(0.1)
        # 6 - Change app state to powering off
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_POWERING_OFF)
        # 7 - Save recording
        self.stop_working_threads()
        if self.get_lsl_worker().data.shape[0] > 0:
            file_path = self.get_file_path_from_rec_info()
            rec_streams_info = self.get_rec_streams_info()
            if file_path is None:
                qt_app = QApplication([])
                self.save_file_dialog = resources.SaveFileDialog(
                    rec_info=self.rec_info,
                    rec_streams_info=rec_streams_info,
                    app_ext=self.app_info['extension'],
                    allowed_formats=self.allowed_formats)
                self.save_file_dialog.accepted.connect(self.on_save_rec_accepted)
                self.save_file_dialog.rejected.connect(self.on_save_rec_rejected)
                qt_app.exec()
            else:
                # Save file automatically
                self.save_recording(file_path, rec_streams_info)
        else:
            print(self.TAG, 'Cannot save because we have no data!!')
        # 8 - Change app state to power off
        self.medusa_interface.app_state_changed(
            mds_constants.APP_STATE_OFF)

    def close_app(self, force=False):
        """ Closes the app controller and working threads. The force parameter
                is not required in Unity apps
                """
        # Trigger the close event in the AppController
        if self.app_controller.server_state.value != SERVER_DOWN:
            self.app_controller.close()
        self.stop_working_threads()

    # ---------------------------- SAVE DATA ----------------------------
    @exceptions.error_handler(scope='app')
    def on_save_rec_accepted(self):
        file_path, self.rec_info = self.save_file_dialog.get_rec_info()
        rec_streams_info = self.save_file_dialog.get_rec_streams_info()
        self.save_recording(file_path, rec_streams_info)

    @exceptions.error_handler(scope='app')
    def on_save_rec_rejected(self):
        pass

    @exceptions.error_handler(scope='app')
    def save_recording(self, file_path, rec_streams_info):
        # Recording
        rec = components.Recording(
            subject_id=self.rec_info.pop('subject_id'),
            recording_id=self.rec_info.pop('rec_id'),
            date=time.strftime("%d-%m-%Y %H:%M", time.localtime()),
            **self.rec_info)
        # Experiment data
        rec.add_experiment_data(self.cvep_data)
        # Biosignal data
        for lsl_stream in self.lsl_streams_info:
            if not rec_streams_info[lsl_stream.medusa_uid]['enabled']:
                continue
            # Get stream data class
            lsl_worker = self.lsl_workers[lsl_stream.medusa_uid]
            stream_data = lsl_worker.get_data_class()
            # Save stream
            att_key = rec_streams_info[lsl_stream.medusa_uid]['att-name']
            rec.add_biosignal(stream_data, att_key)
        # Save recording
        rec.save(file_path)
        # Print a message
        self.medusa_interface.log('Recording saved successfully')

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
        decoding = dict()

        # Get current eeg data
        times_, signal_, fs, channels, equip = self.get_eeg_data()
        eeg = meeg.EEG(times_, signal_, fs, channels, equipement=equip)

        # Process the last trial
        if isinstance(self.cvep_model, cvep.CVEPModelCircularShifting):
            prediction = self.cvep_model.predict(times=times_, signal=signal_,
                                               trial_idx=self.cvep_data.trial_idx[-1],
                                               exp_data=self.cvep_data,
                                               sig_data=eeg)
            label = prediction['spell_result']
            coords = self.app_settings.encoding_settings.get_coords_from_labels(
            label, self.app_settings.encoding_settings.matrices)
            decoding = {
                'coords': coords[0],
                'cmd_label': label,
            }
        elif isinstance(self.cvep_model, (cvep.CMDModelBWRLDA,
                                          cvep.CMDModelBWREEGInception,
                                          cvep.CVEPModelBWRRiemannianLDA)):
            # Get sequence of first command
            seq_len = len(self.app_settings.encoding_settings.matrices[0].item_list[0].sequence)
            # Get trial idx to send only the last command
            trial_idx = self.cvep_data.trial_idx.astype(int)
            last_trial_idx = trial_idx == trial_idx[-1]
            # Get last trial info
            x_info = dict()
            x_info['fps'] = self.cvep_data.fps_resolution
            x_info['code_len'] = seq_len
            x_info['commands_info'] = [self.cvep_data.commands_info]
            x_info['run_idx'] = \
                np.zeros_like(trial_idx).astype(int)[last_trial_idx]
            x_info['trial_idx'] = \
                trial_idx[last_trial_idx]
            x_info['cycle_idx'] = \
                self.cvep_data.cycle_idx.astype(int)[last_trial_idx]
            x_info['cycle_onsets'] = \
                self.cvep_data.onsets[last_trial_idx]
            # Process the last trial
            cmds, __, __ = self.cvep_model.predict(times=times_,
                                                  signal=signal_,
                                                  fs=fs,
                                                  channel_set=channels,
                                                  x_info=x_info)
            cmd = cmds[-1][-1][-1]
            cmd_item = self.cvep_data.commands_info[cmd[0]][cmd[1]]
            coords = [cmd[0], cmd_item['row'], cmd_item['col']]
            label = cmd_item['label']
            decoding = {
                'coords': coords,
                'cmd_label': label,
            }
        return decoding

    def get_conf(self, mode):
        # TODO: nested matrices (units) are not implemented yet
        cvep_conf = []
        cvep_comms = []

        prev_idx = 0
        for m in self.app_settings.encoding_settings.matrices:
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
