# External modules
import asyncio
import subprocess
import multiprocessing as mp
# Medusa modules
import constants
from .app_constants import *
from tcp.async_tcp_server import TCPServer


class AppController(TCPServer):
    """ Class that handles the communication between MEDUSA and Unity.

    The AppController must execute the Unity app and control the communication
    flow between MEDUSA and Unity by using a separate thread called
    ``AppControllerWorker``, which is initialized and started in the constructor
    of this class. The asynchronous communication is handled by a TCPServer
    instance, which is also initialized in the constructor. For this reason,
    this class must inherit from ``TCPServerReadInterface``, making mandatory
    the overriding of the following methods:
        - `on_data_received(messageReceived)` to receive messages.
        - `on_server_up()` to being notified that the server is ready.

    Attributes
    ----------
    app_settings : Settings
        Settings of this application (defined in `settings.py`).
    run_state : multiprocessing.Value
        State that controls the flow of MEDUSA.
    queue_to_controller : queue.Queue
        Queue used to receive messages in `AppControllerWorker`.
    queue_from_controller : queue.Queue
        Queue used to send messages from `AppControllerWorker`.
    tcp_server : TCPServer
        Asynchronous TCP server to receive and send parameters between MEDUSA
        and Unity.
    server_state : multiprocessing.Value
        State that controls the status of the TCP server app according to
        `app_constants.py`.
    unity_state : multiprocessing.Value
        State that controls the status of the Unity app according to
        `app_constants.py`.
    working_thread : AppControllerWorker
        Thread that controls the communication flow between MEDUSA and Unity.
    """

    def __init__(self, callback, app_settings, run_state):
        """ Constructor method for the ``AppController```class.

        Parameters
        ----------
        app_settings : Settings
        run_state : multiprocessing.Value
        queue_to_controller : queue.Queue
        queue_from_controller : queue.Queue
        """
        # Parameters
        self.TAG = '[apps.cvep_speller.AppController]'
        self.callback = callback
        self.app_settings = app_settings
        self.run_state = run_state

        # Pass the IP and port to the TCPServer
        super().__init__(ip=self.app_settings.connection_settings.ip,
                         port=self.app_settings.connection_settings.port)

        # States
        self.server_state = mp.Value('i', SERVER_DOWN)
        self.unity_state = mp.Value('i', UNITY_DOWN)

    def closeEvent(self, event):
        self.close()
        event.accept()

    def close(self):
        super().stop()
        self.server_state.value = SERVER_DOWN

    def start_application(self):
        """ Starts the Unity application that will act as a TCP client. """
        try:
            subprocess.call([self.app_settings.connection_settings.path_to_exe,
                             self.app_settings.connection_settings.ip,
                             str(self.app_settings.connection_settings.port)])
        except Exception as ex:
            raise ex

    def start_server(self):
        """ Starts the TCP server in MEDUSA. """
        try:
            super().start()
        except Exception as ex:
            raise ex

    # --------------- SEND MESSAGES TO UNITY --------------- #
    def send_parameters(self):
        print(self.TAG, "Setting parameters...")
        msg = dict()
        msg["event_type"] = "setParameters"
        msg["mode"] = self.app_settings.run_settings.mode
        msg["trainCycles"] = self.app_settings.run_settings.train_cycles
        msg["trainTrials"] = self.app_settings.run_settings.train_trials
        msg["testCycles"] = self.app_settings.run_settings.test_cycles
        msg["tPrevText"] = self.app_settings.timings.t_prev_text
        msg["tPrevIddle"] = self.app_settings.timings.t_prev_iddle
        msg["tFinishText"] = self.app_settings.timings.t_finish_text
        msg["fpsResolution"] = self.app_settings.run_settings.fps_resolution
        msg["photodiodeEnabled"] =  \
            self.app_settings.run_settings.enable_photodiode
        msg["matrices"] = self.app_settings.get_dict_matrices()
        msg["color_background"] = self.app_settings.colors.color_background
        msg["color_target_box"] = self.app_settings.colors.color_target_box
        msg["color_highlight_result_box"] =  \
            self.app_settings.colors.color_highlight_result_box
        msg["color_result_info_box"] =  \
            self.app_settings.colors.color_result_info_box
        msg["color_result_info_label"] =  \
            self.app_settings.colors.color_result_info_label
        msg["color_result_info_text"] = \
            self.app_settings.colors.color_result_info_text
        msg["color_fps_good"] = self.app_settings.colors.color_fps_good
        msg["color_fps_bad"] = self.app_settings.colors.color_fps_bad
        msg["color_box_0"] = self.app_settings.colors.color_box_0
        msg["color_box_1"] = self.app_settings.colors.color_box_1
        msg["color_text_0"] = self.app_settings.colors.color_text_0
        msg["color_text_1"] = self.app_settings.colors.color_text_1

        self.send_command(msg)

    def play(self):
        print(self.TAG, "Play!")
        msg = dict()
        msg["event_type"] = "play"
        self.send_command(msg)

    def pause(self):
        print(self.TAG, "Pause!")
        msg = dict()
        msg["event_type"] = "pause"
        self.send_command(msg)

    def resume(self):
        print(self.TAG, "Resume!")
        msg = dict()
        msg["event_type"] = "resume"
        self.send_command(msg)

    def stop(self):
        print(self.TAG, "Stop!")
        msg = dict()
        msg["event_type"] = "stop"
        self.send_command(msg)

    def notify_selection(self, selection_coords, selection_label):
        print(self.TAG, "Notifying selection: " + selection_label)
        msg = dict()
        msg["event_type"] = "selection"
        msg["selection_coords"] = selection_coords
        self.send_command(msg)

    def notify_model_trained(self):
        print(self.TAG, "Notifying that model is already fitted...")
        msg = dict()
        msg["event_type"] = "model_trained"
        self.send_command(msg)

    # --------------------- ABSTRACT METHODS -------------------- #
    def on_server_up(self):
        self.server_state.value = SERVER_UP
        print(self.TAG, "Server is UP!")

    def send_command(self, command_dict, client_addresses=None):
        """ Stores a dict command in the TCP server's buffer to send it in the
        next loop iteration.

        Parameters
        ----------
        command_dict : dict
            Dictionary that includes the command to be sent.
        client_addresses : list of (string, int)
            List of client's addresses who must receive the command, use None
            for sending the message to all connected clients.
        """
        super().send_command(client_addresses=client_addresses,
                             msg=command_dict)

    def on_data_received(self, client_address, received_message):
        """ Callback when TCP server receives a message from Unity.

        Parameters
        ----------
        client_address : (string, int)
            IP and port of the client that sent the message
        received_message : string
            JSON encoded string of the message received from the client,
            which will be decoded as a dictionary afterward.
        """
        client_address, msg = super().on_data_received(
            client_address, received_message)
        # Decoding
        if msg["event_type"] == "waiting":
            # Unity is UP and waiting for the parameters
            self.unity_state.value = UNITY_UP
            print(self.TAG, "Unity app is opened.")
        elif msg["event_type"] == "ready":
            # Unity is READY to start
            self.unity_state.value = UNITY_READY
            self.run_state.value = constants.RUN_STATE_READY
            print(self.TAG, "Unity app is ready.")
        elif msg["event_type"] == "close":
            # Unity has closed the client
            self.unity_state.value = UNITY_DOWN
            print(self.TAG, "Unity closed the client")
        elif msg["event_type"] == "finish":
            # Unity has finished the stimulation and standby until STOP button
            # is pressed (manager is still recording)
            self.unity_state.value = UNITY_FINISHED
        elif msg["event_type"] == "train" or msg["event_type"] == "test":
            # Onset information. E.g.: msg = {"event_type":"train","target":"C",
            # "cycle":0,"onset":5393}
            self.callback.process_event(msg)
        elif msg["event_type"] == "resize":
            # Raster latencies information
            self.callback.process_event(msg)
        elif msg["event_type"] == "trainModelPlease":
            # Unity is requesting MEDUSA to train the model with the
            # received trials
            self.callback.process_event(msg)
        elif msg["event_type"] == "processPlease":
            # Unity is requesting MEDUSA to process the previous trial
            self.callback.process_event(msg)
        elif msg["event_type"] == "random":
            # todo: borrar porque es random stim
            self.callback.process_event(msg)
        else:
            print(self.TAG, "Unknown message in 'on_data_received': " +
                  msg["event_type"])
