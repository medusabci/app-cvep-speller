from PySide6.QtUiTools import loadUiType
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QSizePolicy, QApplication, QColorDialog
from gui import gui_utils
from . import settings
import os
import glob
import json
from functools import partial
from medusa.bci import cvep_spellers
from medusa import components
import pickle
from gui.qt_widgets.notifications import NotificationStack
from gui.qt_widgets.dialogs import error_dialog, warning_dialog
from medusa.bci.cvep_spellers import LFSR, LFSR_PRIMITIVE_POLYNOMIALS
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
# from matplotlib.backends.backend_qt import FigureCanvasQT
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from .utils_win_monitor_rates import get_monitor_rates

# Load the .ui files
ui_main_file = loadUiType(os.path.dirname(__file__) + "/config.ui")[0]
ui_target_file = loadUiType(os.path.dirname(__file__) +
                                "/config_target.ui")[0]
ui_encoding_file = loadUiType(os.path.dirname(__file__) +
                                "/config_encoding.ui")[0]


class Config(QtWidgets.QDialog, ui_main_file):
    """ This class provides graphical configuration for the app """

    close_signal = Signal(object)

    def __init__(self, sett, medusa_interface,
                 working_lsl_streams_info, theme_colors=None):
        """
        Config constructor.

        Parameters
        ----------
        sett: settings.Settings
            Instance of class Settings defined in settings.py in the app
            directory
        """
        QtWidgets.QDialog.__init__(self)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setupUi(self)
        self.TAG = '[apps/cvep_speller/config] '

        # Load default settings
        self.settings = sett.Settings() if sett is None else sett

        # Initialize the gui application
        theme_colors = gui_utils.get_theme_colors('dark')
        self.stl = gui_utils.set_css_and_theme(self, theme_colors)
        self.setWindowIcon(QtGui.QIcon('gui/images/medusa_task_icon.png'))
        self.setWindowTitle('c-VEP Speller Configuration')
        self.changes_made = False
        self.notifications = NotificationStack(parent=self, timer_ms=500)

        # Adjust parameters depending on default option
        if self.comboBox_mode.currentText() == 'Online':
            self.train_test_box.setCurrentIndex(1)
            self.lineEdit_session.setText('Test')
            self.label_cvep_model.setVisible(True)
            self.label_test_cycles.setVisible(True)
            self.spinBox_testcycles.setVisible(True)
            self.lineEdit_cvepmodel.setVisible(True)
            self.btn_browse_cvepmodel.setVisible(True)
            self.label_train_trials.setVisible(False)
            self.label_train_cycles.setVisible(False)
            self.spinBox_traincycles.setVisible(False)
            self.spinBox_traintrials.setVisible(False)
        else:
            self.train_test_box.setCurrentIndex(0)
            self.lineEdit_session.setText('Train')
            self.label_cvep_model.setVisible(False)
            self.label_test_cycles.setVisible(False)
            self.spinBox_testcycles.setVisible(False)
            self.lineEdit_cvepmodel.setVisible(False)
            self.btn_browse_cvepmodel.setVisible(False)
            self.label_train_trials.setVisible(True)
            self.label_train_cycles.setVisible(True)
            self.spinBox_traincycles.setVisible(True)
            self.spinBox_traintrials.setVisible(True)

        # Connect signals
        self.btn_reset.clicked.connect(self.reset)
        self.btn_save.clicked.connect(self.save)
        self.btn_load.clicked.connect(self.load)
        self.btn_done.clicked.connect(self.done)
        self.btn_train_model.clicked.connect(self.train_model)
        self.btn_browse_cvepmodel.clicked.connect(self.browse_model)
        self.btn_update_matrix.clicked.connect(self.update_test_matrix)
        self.comboBox_seqlength.currentTextChanged.connect(
            self.on_seqlen_changed)
        self.spinBox_fpsresolution.valueChanged.connect(
            self.on_fpsresolution_changed)
        self.comboBox_mode.currentIndexChanged.connect(self.on_mode_changed)

        # Color buttons
        self.btn_color_box0.clicked.connect(self.open_color_dialog(
            self.btn_color_box0))
        self.btn_color_box1.clicked.connect(
            self.open_color_dialog(self.btn_color_box1))
        self.btn_color_text0.clicked.connect(self.open_color_dialog(
            self.btn_color_text0))
        self.btn_color_text1.clicked.connect(
            self.open_color_dialog(self.btn_color_text1))
        self.btn_color_target_box.clicked.connect(self.open_color_dialog(
            self.btn_color_target_box))
        self.btn_color_highlight_result_box.clicked.connect(
            self.open_color_dialog(self.btn_color_highlight_result_box))
        self.btn_color_fps_good.clicked.connect(self.open_color_dialog(
            self.btn_color_fps_good))
        self.btn_color_fps_bad.clicked.connect(self.open_color_dialog(
            self.btn_color_fps_bad))
        self.btn_color_result_info_box.clicked.connect(
            self.open_color_dialog(self.btn_color_result_info_box))
        self.btn_color_result_info_label.clicked.connect(
            self.open_color_dialog(self.btn_color_result_info_label))
        self.btn_color_result_info_text.clicked.connect(
            self.open_color_dialog(self.btn_color_result_info_text))

        # Background buttons
        if self.comboBox_scenario_name.currentText() == "Solid Color":
            self.lineEdit_scenario.setVisible(False)
            self.btn_browse_scenario.setVisible(False)
        elif self.comboBox_scenario_name.currentText() == "Real Scenario":
            self.label_color_background.setVisible(False)
            self.btn_color_background.setVisible(False)

        self.comboBox_scenario_name.currentIndexChanged.connect(self.on_background_changed)
        self.btn_browse_scenario.clicked.connect(self.browse_scenario)
        self.btn_color_background.clicked.connect(self.open_color_dialog(
            self.btn_color_background))

        # Set settings to GUI
        self.set_settings_to_gui()
        self.on_seqlen_changed()
        self.notifications.new_notification('Default settings loaded')

        # Train model items
        self.tableWidget_bpf.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget_bpf.customContextMenuRequested.connect(
            self.on_custom_table_menu
        )

        # Application ready
        self.setModal(True)
        self.show()

    # --------------------- Settings updating --------------------
    def on_fpsresolution_changed(self):
        self.textEdit_monitor_rates.clear()
        monitors = get_monitor_rates()
        if len(monitors) == 0:
            text = "No connected monitor is detected. The app cannot " \
                   "guarantee a real updating using %s Hz" % \
                   self.spinBox_fpsresolution.value()
            self.textEdit_monitor_rates.append(text)
        else:
            rates = list()
            text = "Connected monitors:\n"
            for monitor in monitors:
                name, rate = monitor
                text += " * %s - max. %i Hz\n" % (name, rate)
                rates.append(rate)
            self.textEdit_monitor_rates.append(text)

            if len(rates) > 1 and not np.all(np.array(rates == rates[0])):
                self.textEdit_monitor_rates.append(
                    "<span style='color: yellow; font-weight: \"bold\"'>"
                    "\n[Warning]: the monitors have different refresh "
                    "rates! The fps_resolution may vary. An exact "
                    "target FPS can only be guaranteed if all monitors "
                    "have the same refresh rate.</span>\n"
                )

            for rate in rates:
                if rate < self.spinBox_fpsresolution.value():
                    self.textEdit_monitor_rates.append(
                        "<span style='color: red; font-weight: \"bold\"'>"
                        "\n[Error]: at least one monitor will not be "
                        "able to reach the desired Target FPS! The "
                        "paradigm will not work.</span>\n"
                    )
        self.update_table_cutoffs()

    def on_mode_changed(self):
        if self.comboBox_mode.currentText() == 'Online':
            self.train_test_box.setCurrentIndex(1)
            self.lineEdit_session.setText('Test')
            self.label_cvep_model.setVisible(True)
            self.label_test_cycles.setVisible(True)
            self.spinBox_testcycles.setVisible(True)
            self.lineEdit_cvepmodel.setVisible(True)
            self.btn_browse_cvepmodel.setVisible(True)
            self.label_train_trials.setVisible(False)
            self.label_train_cycles.setVisible(False)
            self.spinBox_traincycles.setVisible(False)
            self.spinBox_traintrials.setVisible(False)
        else:
            self.train_test_box.setCurrentIndex(0)
            self.lineEdit_session.setText('Train')
            self.label_cvep_model.setVisible(False)
            self.label_test_cycles.setVisible(False)
            self.spinBox_testcycles.setVisible(False)
            self.lineEdit_cvepmodel.setVisible(False)
            self.btn_browse_cvepmodel.setVisible(False)
            self.label_train_trials.setVisible(True)
            self.label_train_cycles.setVisible(True)
            self.spinBox_traincycles.setVisible(True)
            self.spinBox_traintrials.setVisible(True)

    def on_seqlen_changed(self):
        mseqlen = int(self.comboBox_seqlength.currentText())

        # Compute parameters
        if mseqlen == 31:
            order = 5
            seed = [1 for i in range(order)]
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][5]
        elif mseqlen == 63:
            order = 6
            seed = [1, 1, 1, 1, 1, 0]
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][6]
        elif mseqlen == 127:
            order = 7
            seed = [1 for i in range(order)]
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][7]
        elif mseqlen == 255:
            order = 8
            seed = [1 for i in range(order)]
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][8]
        else:
            raise ValueError('[cvep_speller/settings] Sequence length of %i '
                             'not supported (use 31, 63, 127 or 255)!' %
                             mseqlen)
        tau = round(mseqlen / (int(self.spinBox_nrow.value()) *
                               int(self.spinBox_nrow.value())))
        cycle_dur = mseqlen / float(self.spinBox_fpsresolution.value())

        # Update things
        self.lineEdit_poly.setText(str(poly_))
        self.lineEdit_base.setText(str(2))
        self.lineEdit_order.setText(str(order))
        self.lineEdit_seed.setText(str(seed))
        self.lineEdit_tau.setText(str(tau))
        self.lineEdit_cycleduration.setText(str(cycle_dur))

    def on_background_changed(self):
        if self.comboBox_scenario_name.currentText() == "Solid Color":
            self.btn_color_background.setVisible(True)
            self.label_color_background.setVisible(True)
            self.lineEdit_scenario.setVisible(False)
            self.btn_browse_scenario.setVisible(False)
            gui_utils.modify_property(self.btn_color_background,
                                      'background-color',
                                      self.settings.background.color_background[:7])

        elif self.comboBox_scenario_name.currentText() == "Real Scenario":
            self.btn_color_background.setVisible(False)
            self.label_color_background.setVisible(False)
            self.lineEdit_scenario.setVisible(True)
            self.btn_browse_scenario.setVisible(True)

    def set_settings_to_gui(self):
        # Run settings
        self.lineEdit_user.setText(self.settings.run_settings.user)
        self.lineEdit_session.setText(self.settings.run_settings.session)
        self.spinBox_run.setValue(self.settings.run_settings.run)
        self.comboBox_mode.setCurrentText(self.settings.run_settings.mode)
        self.spinBox_traincycles.setValue(
            self.settings.run_settings.train_cycles)
        self.spinBox_traintrials.setValue(
            self.settings.run_settings.train_trials)
        self.spinBox_testcycles.setValue(self.settings.run_settings.test_cycles)

        self.lineEdit_cvepmodel.setText(
            self.settings.run_settings.cvep_model_path)
        self.spinBox_fpsresolution.setValue(
            self.settings.run_settings.fps_resolution)
        self.checkBox_photodiode.setChecked(
            self.settings.run_settings.enable_photodiode)

        # Timings
        self.doubleSpinBox_t_prev_text.setValue(
            self.settings.timings.t_prev_text)
        self.doubleSpinBox_t_prev_iddle.setValue(
            self.settings.timings.t_prev_iddle)
        self.doubleSpinBox_t_finish_text.setValue(
            self.settings.timings.t_finish_text)

        # Colors
        gui_utils.modify_property(self.btn_color_box0, 'background-color',
                                  self.settings.colors.color_box_0[:7])
        self.spinBox_op_box_0.setValue(self.settings.colors.color_op_box_0)
        gui_utils.modify_property(self.btn_color_box1, 'background-color',
                                  self.settings.colors.color_box_1[:7])
        self.spinBox_op_box_1.setValue(self.settings.colors.color_op_box_1)
        gui_utils.modify_property(self.btn_color_text0, 'background-color',
                                  self.settings.colors.color_text_0[:7])
        self.spinBox_op_text_0.setValue(self.settings.colors.color_op_text_0)
        gui_utils.modify_property(self.btn_color_text1, 'background-color',
                                  self.settings.colors.color_text_1[:7])
        self.spinBox_op_text_1.setValue(self.settings.colors.color_op_text_1)
        gui_utils.modify_property(self.btn_color_target_box, 'background-color',
                                  self.settings.colors.color_target_box[:7])
        gui_utils.modify_property(self.btn_color_highlight_result_box,
                                  'background-color',
                                  self.settings.colors.color_highlight_result_box[:7])
        gui_utils.modify_property(self.btn_color_fps_good,
                            'background-color',
                                  self.settings.colors.color_fps_good[:7])
        gui_utils.modify_property(self.btn_color_fps_bad,
                                  'background-color',
                                  self.settings.colors.color_fps_bad[:7])
        gui_utils.modify_property(self.btn_color_result_info_box,
                                  'background-color',
                                  self.settings.colors.color_result_info_box[:7])
        gui_utils.modify_property(self.btn_color_result_info_label,
                                  'background-color',
                                  self.settings.colors.color_result_info_label[:7])
        gui_utils.modify_property(self.btn_color_result_info_text,
                                  'background-color',
                                  self.settings.colors.color_result_info_text[:7])

        # Background
        self.comboBox_scenario_name.setCurrentText(self.settings.background.scenario_name)
        gui_utils.modify_property(self.btn_color_background,
                                  'background-color',
                                  self.settings.background.color_background[:7])
        self.lineEdit_scenario.setText(
            self.settings.background.scenario_path)


        # Useful PyQt policies
        policy_max_pre = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        policy_max_max = QSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
        policy_fix_fix = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Test matrices
        # Create the required number of tabs
        n_extra = len(self.settings.matrices['test']) - \
                  self.widget_nested_test.count()
        for t in range(1, n_extra + 1):
            mtx_widget_ = QtWidgets.QWidget(self.widget_nested_test)
            mtx_idx_ = self.nested_box.count() - 1
            self.widget_nested_test.insertTab(mtx_idx_, mtx_widget_,
                                              'Matrix #' + str(mtx_idx_))
            self.widget_nested_test.setCurrentIndex(0)
        # Create each test matrix
        for m in range(len(self.settings.matrices['test'])):
            # Set the current index and create the general layout
            curr_mtx = self.settings.matrices['test'][m]
            self.widget_nested_test.setCurrentIndex(m)
            global_layout = QtWidgets.QVBoxLayout()
            # Create the result text frame
            result_frame = QtWidgets.QFrame()
            result_text = QtWidgets.QLabel('B C I')
            result_text.setObjectName('label_result_text')
            result_text.setAlignment(QtCore.Qt.AlignLeft)
            result_title = QtWidgets.QLabel('RESULT ')
            result_title.setObjectName('label_result_title')
            result_title.setAlignment(QtCore.Qt.AlignLeft)
            result_title.setSizePolicy(policy_max_pre)
            fps_monitor = QtWidgets.QLabel(
                'FPS @%iHz' % self.settings.run_settings.fps_resolution)
            fps_monitor.setObjectName('label_fps')
            fps_monitor.setAlignment(QtCore.Qt.AlignRight)
            result_layout = QtWidgets.QHBoxLayout()
            result_layout.addWidget(result_title)
            result_layout.addWidget(result_text)
            result_layout.addWidget(fps_monitor)
            result_frame.setLayout(result_layout)
            global_layout.addWidget(result_frame)
            # Customize the result text
            gui_utils.modify_properties(
                result_title, {
                    "font-style": "italic",
                    "color": self.settings.colors.color_result_info_text[:7]
                })
            gui_utils.modify_property(
                result_frame, "background-color",
                self.settings.colors.color_result_info_box[:7])
            gui_utils.modify_property(
                fps_monitor, "color",
                self.settings.colors.color_fps_good[:7])
            # Create a new layout for the commands
            new_layout = QtWidgets.QGridLayout()
            new_layout.setContentsMargins(0, 0, 0, 0)
            new_layout.setSpacing(10)
            new_layout.setContentsMargins(10, 10, 10, 10)
            # Add buttons as commands
            for r in range(curr_mtx.n_row):
                for c in range(curr_mtx.n_col):
                    key_ = curr_mtx.matrix_list[r][c].sequence[0]
                    temp_button = QtWidgets.QToolButton()
                    temp_button.setObjectName('btn_command')
                    temp_button.setText(curr_mtx.matrix_list[r][c].text)
                    temp_button.clicked.connect(self.btn_command_on_click(r, c))
                    temp_button.setMinimumSize(60, 60)
                    temp_button.setSizePolicy(policy_max_max)
                    box_color_ = self.settings.colors.color_box_0 if \
                        key_ == 0 else self.settings.colors.color_box_1
                    text_color_ = self.settings.colors.color_text_0 if \
                        key_ == 0 else self.settings.colors.color_text_1
                    gui_utils.modify_properties(
                        temp_button, {
                            "background-color": box_color_,
                            "font-family": 'sans-serif, Helvetica, Arial',
                            'font-size': '30px',
                            'color': text_color_,
                            'border': 'transparent'
                        })
                    new_layout.addWidget(temp_button, r, c)
            global_layout.addLayout(new_layout)
            global_layout.setSpacing(0)
            global_layout.setContentsMargins(0, 0, 0, 0)
            # Update the tab
            new_tab = QtWidgets.QFrame()
            new_tab.setLayout(global_layout)
            gui_utils.modify_property(new_tab, 'background-color',
                                      self.settings.background.color_background[:7])
            self.update_tab(self.widget_nested_test, m, new_tab)
        self.spinBox_nrow.setValue(self.settings.matrices['test'][0].n_row)
        self.spinBox_ncol.setValue(self.settings.matrices['test'][0].n_col)

        # Train matrices
        # Create the required number of tabs
        n_extra = len(self.settings.matrices['train']) - \
                  self.widget_nested_train.count()
        for t in range(1, n_extra + 1):
            mtx_widget_ = QtWidgets.QWidget(self.widget_nested_train)
            mtx_idx_ = self.nested_box.count() - 1
            self.widget_nested_train.insertTab(mtx_idx_, mtx_widget_,
                                               'Matrix #' + str(mtx_idx_))
            self.widget_nested_train.setCurrentIndex(0)
        # Create each training matrix
        for m in range(len(self.settings.matrices['train'])):
            # Set the current index and create the general layout
            curr_mtx = self.settings.matrices['train'][m]
            self.widget_nested_train.setCurrentIndex(m)
            global_layout = QtWidgets.QVBoxLayout()
            # Create the result text frame
            result_frame = QtWidgets.QFrame()
            result_text = QtWidgets.QLabel('B C I')
            result_text.setObjectName('label_result_text')
            result_text.setAlignment(QtCore.Qt.AlignLeft)
            result_title = QtWidgets.QLabel('RESULT ')
            result_title.setObjectName('label_result_title')
            result_title.setAlignment(QtCore.Qt.AlignLeft)
            result_title.setSizePolicy(policy_max_pre)
            fps_monitor = QtWidgets.QLabel(
                'FPS @%iHz' % self.settings.run_settings.fps_resolution)
            fps_monitor.setObjectName('label_fps')
            fps_monitor.setAlignment(QtCore.Qt.AlignRight)
            result_layout = QtWidgets.QHBoxLayout()
            result_layout.addWidget(result_title)
            result_layout.addWidget(result_text)
            result_layout.addWidget(fps_monitor)
            result_frame.setLayout(result_layout)
            global_layout.addWidget(result_frame)
            # Customize the result text
            gui_utils.modify_properties(
                result_title, {
                    "font-style": "italic",
                    "color": self.settings.colors.color_result_info_text[:7]
                })
            gui_utils.modify_property(
                result_frame, "background-color",
                self.settings.colors.color_result_info_box[:7])
            gui_utils.modify_property(
                fps_monitor, "color",
                self.settings.colors.color_fps_good[:7])
            # Create a new layout for the commands
            new_layout = QtWidgets.QGridLayout()
            new_layout.setContentsMargins(0, 0, 0, 0)
            new_layout.setSpacing(10)
            new_layout.setContentsMargins(10, 10, 10, 10)
            # Add buttons as commands
            for r in range(curr_mtx.n_row):
                for c in range(curr_mtx.n_col):
                    key_ = curr_mtx.matrix_list[r][c].sequence[0]
                    temp_button = QtWidgets.QToolButton()
                    temp_button.setObjectName('btn_command')
                    temp_button.setSizePolicy(policy_max_max)
                    temp_button.setText(curr_mtx.matrix_list[r][c].text)
                    temp_button.setMinimumSize(60, 60)
                    temp_button.clicked.connect(
                        self.btn_command_on_click(r, c))
                    box_color_ = self.settings.colors.color_box_0 if \
                        key_ == 0 else self.settings.colors.color_box_1
                    text_color_ = self.settings.colors.color_text_0 if \
                        key_ == 0 else self.settings.colors.color_text_1
                    gui_utils.modify_properties(
                        temp_button, {
                            "background-color": box_color_,
                            "font-family": 'sans-serif, Helvetica, Arial',
                            'font-size': '30px',
                            'color': text_color_,
                            'border': 'transparent'
                        })
                    new_layout.addWidget(temp_button, r, c)
            global_layout.addLayout(new_layout)
            global_layout.setSpacing(0)
            global_layout.setContentsMargins(0, 0, 0, 0)

            # Update the tab
            new_tab = QtWidgets.QFrame()
            new_tab.setLayout(global_layout)
            gui_utils.modify_property(new_tab, 'background-color',
                                      self.settings.background.color_background[:7])
            self.update_tab(self.widget_nested_train, m, new_tab)

        # Filter cutoffs according to fps_resolution
        self.update_table_cutoffs()

        # Sequence length in test
        seqlen = len(self.settings.matrices['test'][0].item_list[0].sequence)
        index = self.comboBox_seqlength.findText(str(seqlen), Qt.MatchFixedString)
        self.comboBox_seqlength.setCurrentIndex(index)

    def btn_command_on_click(self, row, col):
        def set_config():
            # This function is required in order to accept passing arguments
            # (function factory)
            current_index = self.widget_nested_test.currentIndex()
            target_dialog = TargetConfigDialog(
                self.settings.matrices['test'][current_index].
                    matrix_list[row][col], current_index)
            if target_dialog.exec_():
                # Get the returned values
                self.settings.matrices['test'][
                    current_index].matrix_list[row][col].set_text(
                    target_dialog.input_target_text.text())
                self.settings.matrices['test'][
                    current_index].matrix_list[row][col].set_label(
                    target_dialog.input_target_label.text())
                seq = eval(target_dialog.input_target_sequence.text())
                self.settings.matrices['test'][
                    current_index].matrix_list[row][col].set_sequence(
                    seq)

                # Update the GUI
                self.set_settings_to_gui()

        return set_config

    def update_tab(self, tabwidget, index, new_content):
        """ Updates the specified tab content.

            :param index: int
                Tab index.
            :param new_content:
                Content to insert.
        """
        tabwidget.insertTab(index, new_content, 'New')
        tabwidget.removeTab(index + 1)
        self.update_matrix_names(tabwidget)

    @staticmethod
    def update_matrix_names(tabwidget):
        """ Updates the matrix names in case that some middle matrix
        has been deleted. """
        for tab_idx in range(tabwidget.count()):
            tabwidget.setTabText(tab_idx, 'Matrix #' + str(tab_idx + 1))

    def get_settings_from_gui(self):
        # Run settings
        self.settings.run_settings.user = self.lineEdit_user.text()
        self.settings.run_settings.session = self.lineEdit_session.text()
        self.settings.run_settings.run = self.spinBox_run.value()
        self.settings.run_settings.mode = self.comboBox_mode.currentText()
        self.settings.run_settings.train_cycles = \
            self.spinBox_traincycles.value()
        self.settings.run_settings.train_trials = \
            self.spinBox_traintrials.value()
        self.settings.run_settings.test_cycles = self.spinBox_testcycles.value()
        self.settings.run_settings.cvep_model_path = self.lineEdit_cvepmodel.text()
        self.settings.run_settings.fps_resolution = self.spinBox_fpsresolution.value()
        self.settings.run_settings.enable_photodiode = self.checkBox_photodiode.isChecked()

        # Timings
        self.settings.timings.t_prev_text = self.doubleSpinBox_t_prev_text.value()
        self.settings.timings.t_prev_iddle = self.doubleSpinBox_t_prev_iddle.value()
        self.settings.timings.t_finish_text = self.doubleSpinBox_t_finish_text.value()

        # Colors
        self.settings.colors.color_box_0 = gui_utils.get_property(
            self.btn_color_box0, 'background-color')
        self.settings.colors.color_op_box_0 = self.spinBox_op_box_0.value()
        self.settings.colors.color_box_1 = gui_utils.get_property(
            self.btn_color_box1, 'background-color')
        self.settings.colors.color_op_box_1 = self.spinBox_op_box_1.value()
        self.settings.colors.color_text_0 = gui_utils.get_property(
            self.btn_color_text0, 'background-color')
        self.settings.colors.color_op_text_0 = self.spinBox_op_text_0.value()
        self.settings.colors.color_text_1 = gui_utils.get_property(
            self.btn_color_text1, 'background-color')
        self.settings.colors.color_op_text_1 = self.spinBox_op_text_1.value()
        self.settings.colors.color_target_box = gui_utils.get_property(
            self.btn_color_target_box, 'background-color')
        self.settings.colors.color_highlight_result_box = gui_utils.get_property(
            self.btn_color_highlight_result_box, 'background-color')
        self.settings.colors.color_fps_good = gui_utils.get_property(
            self.btn_color_fps_good, 'background-color')
        self.settings.colors.color_fps_bad = gui_utils.get_property(
            self.btn_color_fps_bad, 'background-color')
        self.settings.colors.color_result_info_box = gui_utils.get_property(
            self.btn_color_result_info_box, 'background-color')
        self.settings.colors.color_result_info_label = gui_utils.get_property(
            self.btn_color_result_info_label, 'background-color')
        self.settings.colors.color_result_info_text = gui_utils.get_property(
            self.btn_color_result_info_text, 'background-color')

        # Background
        self.settings.background.scenario_name = self.comboBox_scenario_name.currentText()
        self.settings.background.color_background = gui_utils.get_property(
            self.btn_color_background, 'background-color')
        self.settings.background.scenario_path = self.lineEdit_scenario.text()

    def update_gui(self):
        self.get_settings_from_gui()
        self.set_settings_to_gui()

    def update_table_cutoffs(self):
        for i in range(self.tableWidget_bpf.rowCount()):
            self.tableWidget_bpf.setItem(i, 1, QtWidgets.QTableWidgetItem(
                str(int(self.settings.run_settings.fps_resolution / 2))))

    def on_custom_table_menu(self, pos):
        # Get action
        menu = QtWidgets.QMenu()
        delete_row_action = menu.addAction("Delete row")
        add_row_action = menu.addAction("Add row")
        action = menu.exec_(self.tableWidget_bpf.viewport().mapToGlobal(pos))

        # Delete row action
        if action == delete_row_action:
            it = self.tableWidget_bpf.itemAt(pos)
            if it is None or self.tableWidget_bpf.rowCount() == 1:
                return
            r = it.row()
            item_range = QtWidgets.QTableWidgetSelectionRange(
                r, 0, r, self.tableWidget_bpf.columnCount() - 1)
            self.tableWidget_bpf.setRangeSelected(item_range, True)
            self.tableWidget_bpf.removeRow(r)
            self.tableWidget_bpf.setVerticalHeaderLabels(
                [str(x) for x in range(1, self.tableWidget_bpf.rowCount() + 1)])

        # Add row action (and populate, otherwise won't work)
        if action == add_row_action:
            r = self.tableWidget_bpf.rowCount()
            self.tableWidget_bpf.insertRow(r)
            self.tableWidget_bpf.setItem(r, 0, QtWidgets.QTableWidgetItem('0'))
            self.tableWidget_bpf.setItem(r, 1, QtWidgets.QTableWidgetItem(
                str(int(self.settings.run_settings.fps_resolution / 2))))
            self.tableWidget_bpf.setItem(r, 2, QtWidgets.QTableWidgetItem('7'))
            self.tableWidget_bpf.setItem(r, 3, QtWidgets.QTableWidgetItem(
                'bandpass'))
            self.tableWidget_bpf.setVerticalHeaderLabels(
                [str(x) for x in range(1, self.tableWidget_bpf.rowCount() + 1)])

    # --------------------- Buttons ------------------------
    def reset(self):
        # Set default settings
        self.settings = settings.Settings()
        self.set_settings_to_gui()
        self.notifications.new_notification('Loaded default settings')

    def save(self):
        config_path = os.getcwd() + "/../config/"
        fdialog = QtWidgets.QFileDialog()
        fname = fdialog.getSaveFileName(
            fdialog, 'Save settings', config_path, 'JSON (*.json)')
        if fname[0]:
            self.get_settings_from_gui()
            self.settings.save(path=fname[0])
            self.notifications.new_notification('Settings saved as %s' %
                                                fname[0].split('/')[-1])

    def load(self):
        """ Opens a dialog to load a configuration file. """
        config_path = os.getcwd() + "/../config/"
        fdialog = QtWidgets.QFileDialog()
        fname = fdialog.getOpenFileName(
            fdialog, 'Load settings', config_path, 'JSON (*.json)')
        if fname[0]:
            loaded_settings = self.settings.load(fname[0])
            self.settings = loaded_settings
            self.set_settings_to_gui()
            self.notifications.new_notification('Loaded settings: %s' %
                                                fname[0].split('/')[-1])

    def done(self):
        """ Shows a confirmation dialog if non-saved changes has been made. """
        self.close()

    def train_model(self):
        def check_train_feasible():
            error_msg = ''
            if self.tableWidget_bpf.rowCount() == 0:
                error_msg += 'Cannot train if we do not have, at least, ' \
                             'one filter!\n'
            for i in range(self.tableWidget_bpf.rowCount()):
                cut1 = float(self.tableWidget_bpf.item(i, 0).text())
                cut2 = float(self.tableWidget_bpf.item(i, 1).text())
                if cut1 == cut2:
                    error_msg += 'Filter %i cannot have the same value for ' \
                                 'cutoff1 and cutoff2 (%.2f Hz)!\n' % (i, cut1)
            if error_msg != '':
                error_dialog(error_msg, 'Error!')
                return False
            else:
                return True

        # Is it feasible?
        if not check_train_feasible():
            return

        # Get data to be trained
        filt = "c-VEP Files (*.cvep.bson)"
        files = QtWidgets.QFileDialog.getOpenFileNames(
            caption="Select training files",
            dir=os.getcwd() + "/../data/",
            filter=filt)
        if files[0]:
            self.notifications.new_notification('Training model...')
            # Get files
            dataset = None
            for idx, file in enumerate(files[0]):
                rec = components.Recording.load(file)
                if idx == 0:
                    dataset = cvep_spellers.CVEPSpellerDataset(
                        channel_set=rec.eeg.channel_set,
                        fs=rec.eeg.fs
                    )
                dataset.add_recordings(rec)
            # Get configuration
            bpf = []
            max_cut2 = 0.0
            for i in range(self.tableWidget_bpf.rowCount()):
                cut1 = float(self.tableWidget_bpf.item(i, 0).text())
                cut2 = float(self.tableWidget_bpf.item(i, 1).text())
                order = int(self.tableWidget_bpf.item(i, 2).text())
                type = self.tableWidget_bpf.item(i, 3).text()
                bpf.append([order, (cut1, cut2)])
                if cut2 > max_cut2:
                    max_cut2 = cut2
            notch = [7, (self.doubleSpinBox_notch.value() - 1,
                         self.doubleSpinBox_notch.value() + 1)]

            # Check if notch is required
            if max_cut2 < notch[1][0]:
                notch = None

            # Train the model
            art_rej = None
            if self.checkBox_calibration_art_rej.isChecked():
                art_rej = 3.0
            model = cvep_spellers.CVEPModelCircularShifting(
                bpf=bpf,
                notch=notch,
                art_rej=art_rej,
                correct_raster_latencies=False
            )
            try:
                fitted_info = model.fit_dataset(dataset)
            except Exception as e:
                error_dialog(str(e), "Cannot train model!")
                return
            if art_rej is not None:
                print(self.TAG, 'Model trained\n  > Discarded %i/%i epochs!' %
                      (fitted_info['no_discarded_epochs'],
                       fitted_info['no_total_epochs']))
            else:
                print(self.TAG, 'Model trained')

            # Disable art_rej for online mode
            model.get_inst("clf_method").art_rej = None

            # Save model
            model_pkl = model.to_pickleable_obj()
            fdialog = QtWidgets.QFileDialog()
            fname = fdialog.getSaveFileName(
                fdialog, 'Save c-VEP Model',
                os.path.join(os.getcwd(), "../models/"),
                'c-VEP Model (*.cvep.mdl)')
            if fname[0]:
                with open(fname[0], 'wb') as handle:
                    pickle.dump(model_pkl, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                self.notifications.new_notification('Model saved as %s' %
                                                    fname[0].split('/')[-1])
                self.lineEdit_cvepmodel.setText(fname[0])

    def browse_model(self):
        filt = "c-VEP Model (*.cvep.mdl)"
        directory = os.getcwd() + "/../models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Created directory %s!' % directory)
        filepath = QtWidgets.QFileDialog.getOpenFileName(caption="c-VEP Model",
                                                         dir=directory,
                                                         filter=filt)
        self.lineEdit_cvepmodel.setText(filepath[0])

    def browse_scenario(self):
        filt = "Image (*.jpg *.jpeg *.png)"
        directory = os.path.dirname(__file__) + "/background/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Created directory %s!' % directory)
        filepath = QtWidgets.QFileDialog.getOpenFileName(caption="Scenario",
                                                         dir=directory,
                                                         filter=filt)
        self.lineEdit_scenario.setText(filepath[0])

    def update_test_matrix(self):
        # Get the parameters
        n_row = int(self.spinBox_nrow.value())
        n_col = int(self.spinBox_ncol.value())
        mseqlen = int(self.comboBox_seqlength.currentText())

        # Check if everything is correct
        tau = mseqlen / (n_row * n_col)
        if tau < 1:
            error_msg = 'Cannot encode all the commands (%i) with that ' \
                        'sequence length (%i)! Decrease the number of ' \
                        'commands or increase the sequence length to have ' \
                        'enough room to get a positive delay.' % \
                        (n_col * n_row, mseqlen)
            error_dialog(error_msg, 'Oops!')
            return
        if tau < 2:
            warn_msg = 'With that number of commands (%i) and that sequence ' \
                       'length (%i), the delay between shifted-version ' \
                       'sequences will be %.2f. Consider to decrease the ' \
                       'number of commands or increase the sequence length ' \
                       'to space more the shifted-sequences and favor the ' \
                       'performance' \
                        % (n_col * n_row, mseqlen, tau)
            warning_dialog(warn_msg, 'Be careful!')
        self.lineEdit_tau.setText("%.2f" % tau)

        # Compute the matrices
        self.get_settings_from_gui()
        train_matrices, test_matrices, lags_info = \
            self.settings.standard_single_sequence_matrices(
            n_row=n_row, n_col=n_col, mseqlen=mseqlen)
        self.settings.matrices = {'train': train_matrices, 'test':
            test_matrices}

        # Update the gui
        self.set_settings_to_gui()

        # Show the encoding
        order = int(self.lineEdit_order.text())
        seed = self.lineEdit_seed.text()
        monitor_rate = float(self.spinBox_fpsresolution.value())
        current_index = self.widget_nested_test.currentIndex()
        visualize_dialog = VisualizeEncodingDialog(
            n_row=n_row, n_col=n_col, base=2, order=order,
            monitor_rate=monitor_rate, item_list=self.settings.matrices[
                'test'][current_index].item_list, lags_info=lags_info)
        visualize_dialog.exec_()

    # --------------------- Colors ------------------------
    def open_color_dialog(self, handle):
        """ Opens a color dialog and sets the selected color in the desired button.

        :param handle: QToolButton
            Button handle.
        """

        def set_color():
            # This function is required in order to accept passing arguments (function factory)
            color = QColorDialog.getColor()  # Open the color dialog and get the QColor instance
            if not color.isValid():
                # For example, if the user closes the dialog
                print("Color is not valid (%s)." % color)
            else:
                handle.setStyleSheet('background-color: ' + color.name() + ';')
                self.update_gui()

        return set_color

    # --------------------- Close events ------------------------
    @staticmethod
    def close_dialog():
        """ Shows a confirmation dialog that asks the user if he/she wants to
        close the configuration window.

        Returns
        -------
        output value: QtWidgets.QMessageBox.No or QtWidgets.QMessageBox.Yes
            If the user do not want to close the window, and
            QtWidgets.QMessageBox.Yes otherwise.
        """
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowIcon(QtGui.QIcon(os.path.join(
            os.path.dirname(__file__), '../../gui/images/medusa_task_icon.png')))
        msg.setText("Do you want to leave this window?")
        msg.setInformativeText("Non-saved changes will be discarded.")
        msg.setWindowTitle("c-VEP Speller Configuration")
        msg.setStandardButtons(
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        return msg.exec_()

    def closeEvent(self, event):
        """ Overrides the closeEvent in order to show the confirmation dialog.
        """
        if self.changes_made:
            retval = self.close_dialog()
            if retval == QtWidgets.QMessageBox.Yes:
                self.close_signal.emit(None)
                event.accept()
            else:
                event.ignore()
        else:
            self.get_settings_from_gui()
            self.close_signal.emit(self.settings)
            event.accept()


class TargetConfigDialog(QtWidgets.QDialog, ui_target_file):

    def __init__(self, target, current_matrix_idx):
        """Class that represents a Target Configuration dialog.
        Parameteres
        -----------
        target: CVEPTarget
            Instance of CVEPTarget that stores the settings of the target to
            customize.
        current_matrix_idx : int
            Index of the currently selected test matrix.
        """
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)  # Attach the .ui
        self.TAG = '[apps/cvep_speller/config_target] '

        # Initialize the dialog
        theme_colors = gui_utils.get_theme_colors('dark')
        self.stl = gui_utils.set_css_and_theme(self, theme_colors)
        self.setWindowIcon(QtGui.QIcon('gui/images/medusa_task_icon.png'))
        self.setWindowTitle('c-VEP target customization')

        # Set tooltips
        self.label_target_text.setToolTip(
            'Text that will be displayed in the target cell')
        self.label_target_label.setToolTip(
            'Label that identifies the target cell')
        self.label_target_sequence.setToolTip(
            'Encoding sequence for this target')

        # Set the current parameters
        self.input_target_matrix.setText(str(current_matrix_idx))
        self.input_target_row.setText(str(target.row))
        self.input_target_column.setText(str(target.col))
        self.input_target_text.setText(target.text)
        self.input_target_label.setText(target.label)
        self.input_target_sequence.setText(str(target.sequence))


class VisualizeEncodingDialog(QtWidgets.QDialog, ui_encoding_file):
    def __init__(self, n_row, n_col, base, order, monitor_rate, item_list,
                 lags_info):
        QtWidgets.QDialog.__init__(self)
        self.setupUi(self)  # Attach the .ui
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.TAG = '[apps/cvep_speller/config_encoding] '

        # Initialize the dialog
        theme_colors = gui_utils.get_theme_colors('dark')
        self.stl = gui_utils.set_css_and_theme(self, theme_colors)
        self.setWindowIcon(QtGui.QIcon('gui/images/medusa_task_icon.png'))
        self.setWindowTitle('c-VEP target customization')

        # Initialize the canvas
        self.fig_autocorr = Figure(figsize=(60, 30), dpi=150, )
        self.canvas_autocorr = FigureCanvas(figure=self.fig_autocorr)
        self.layout_autocorr.addWidget(self.canvas_autocorr)
        self.axes_autocorr = self.fig_autocorr.add_subplot(111)
        self.fig_encoding = Figure(figsize=(60, 30), dpi=150, )
        self.canvas_encoding = FigureCanvas(figure=self.fig_encoding)
        self.layout_encoding.addWidget(self.canvas_encoding)
        self.axes_encoding = self.fig_encoding.add_subplot(111)

        # Autocorrelation plot
        lags = lags_info['lags']
        SMALL_SIZE = 4
        MEDIUM_SIZE = 6
        plt.rcParams.update({'font.size': 4})
        poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order'][order]
        seq = LFSR(poly_, base=base, center=True).sequence
        rxx_, tr_ = self.autocorr_circular(seq)
        rxx_ = rxx_ / np.max(np.abs(rxx_))
        with plt.style.context('dark_background'):
            # Autocorrelation
            tr_s = np.array(tr_) / monitor_rate
            big_lagged_seqs_ = np.zeros((n_row * n_col, len(seq)))
            seq = np.array(seq)
            self.axes_autocorr.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            self.axes_autocorr.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            self.axes_autocorr.plot(tr_s, rxx_, linewidth=1)
            yoff = -min(np.abs(rxx_))
            for i, lag in enumerate(lags):
                big_lagged_seqs_[i, :] = np.array(np.roll(seq, lag)).reshape(-1, 1).T
                self.axes_autocorr.plot([lag/monitor_rate, lag/monitor_rate],
                         [yoff - 0.05, yoff + 0.05], color='#ff1e55',
                                        linewidth=0.5)
                self.axes_autocorr.text(lag / monitor_rate, yoff - 0.1,
                                        item_list[i].text,
                                        horizontalalignment='center',
                                        color='#ff1e55')
            self.axes_autocorr.set_xlim((tr_s[0], tr_s[-1]))
            self.axes_autocorr.set_ylim((min(rxx_) - 0.2, max(rxx_) + 0.2))
            self.axes_autocorr.set_xlabel('Time shifts (s)', fontsize=MEDIUM_SIZE)
            self.axes_autocorr.set_ylabel('Norm. $R_{xx}$', fontsize=MEDIUM_SIZE)
            self.axes_autocorr.set_title('M-sequence autocorrelation',
                                         fontsize=MEDIUM_SIZE)
            self.axes_autocorr.tick_params(axis='x', labelsize=SMALL_SIZE)
            self.axes_autocorr.tick_params(axis='y', labelsize=SMALL_SIZE)
        pos = self.axes_autocorr.get_position()
        pos.x0 = 0.2
        pos.y0 = 0.15
        self.axes_autocorr.set_position(pos)
        self.fig_autocorr.patch.set_alpha(0.5)
        self.canvas_autocorr.draw()

        # Encoding plot
        with plt.style.context('dark_background'):
            commands = list()
            for c in item_list:
                commands.append(c.text)
            self.axes_encoding.imshow(big_lagged_seqs_, aspect='auto',
                                      cmap='gray_r')
            self.axes_encoding.set_yticks(np.arange(len(commands)))
            self.axes_encoding.set_yticklabels(commands)
            self.axes_encoding.set_title('Command encoding', fontsize=MEDIUM_SIZE)
            self.axes_encoding.set_xlabel('Sequence (samples)', fontsize=MEDIUM_SIZE)
            self.axes_encoding.set_ylabel('Commands', fontsize=MEDIUM_SIZE)
            self.axes_encoding.set_yticks([i for i in range(len(commands))])
            self.axes_encoding.tick_params(axis='x', labelsize=SMALL_SIZE)
            self.axes_encoding.tick_params(axis='y', labelsize=SMALL_SIZE)
        pos = self.axes_encoding.get_position()
        pos.x0 = 0.2
        pos.y0 = 0.15
        self.axes_encoding.set_position(pos)
        self.fig_encoding.patch.set_alpha(0.5)
        self.canvas_encoding.draw()

        # Correlation values table
        self.table_values.setRowCount(1)
        self.table_values.setColumnCount(n_row * n_col)
        half_rxx = rxx_[int(len(rxx_) / 2):]
        values = list()
        for i, lag in enumerate(lags):
            values.append(half_rxx[lag])
            item = QtWidgets.QTableWidgetItem("{:.2f}".format(half_rxx[lag]))
            self.table_values.setItem(0, i, item)
        values = np.array(values)
        min_p = - min(np.abs(rxx_))
        self.bad_cmds = list()
        for i in range(n_row * n_col):
            if i == 0:
                self.table_values.item(0, i).setBackground(Qt.darkBlue)
                continue
            if min_p != values[i]:
                self.bad_cmds.append(commands[i])
                self.table_values.item(0, i).setBackground(Qt.darkRed)
            else:
                self.table_values.item(0, i).setBackground(Qt.darkGreen)
        self.table_values.setHorizontalHeaderLabels(commands)
        self.table_values.setVerticalHeaderLabels(['p'])
        self.table_values.resizeColumnsToContents()

        # Mean tau
        self.edit_tau.setText("{:.2f}".format(lags_info['tau']))

        # Advise
        if np.all(min_p == values[1:]):
            self.label_values.setText("✔️  Encoding is correct!")
            self.label_values.setStyleSheet("color: limegreen;")
        else:
            self.label_values.setText("❌️  Careful, I could not optimize the "
                                      "lags enough for this configuration"
                                      ", so the correlation for "
                                      "command(s) %s is not minimum!" %
                                      ','.join(self.bad_cmds))
            self.label_values.setStyleSheet("color: orangered;")

    @staticmethod
    def autocorr_circular(x):
        """ With circular shifts (periodic correlation) """
        N = len(x)
        rxx = []
        t = []
        for i in range(-(N - 1), N):
            rxx.append(np.sum(x * np.roll(x, i)))
            t.append(i)
        rxx = np.array(rxx)
        return rxx, t

