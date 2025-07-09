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

        # Connect general unbuttons
        self.btn_reset.clicked.connect(self.reset)
        self.btn_save.clicked.connect(self.save)
        self.btn_load.clicked.connect(self.load)
        self.btn_done.clicked.connect(self.done)

        # Stimuli buttons (to switch layers)
        self.btn_plain.clicked.connect(self.on_stimuli_changed)
        self.btn_grating.clicked.connect(self.on_stimuli_changed)
        self.btn_checkerboard.clicked.connect(self.on_stimuli_changed)

        for l_idx in range(1, self.params_stacked_widget.count()+1):
            # Run settings buttons
            getattr(self, f"comboBox_mode_{l_idx}").currentIndexChanged.connect(
                self.on_mode_changed)
            getattr(self, f"btn_browse_cvepmodel_{l_idx}").clicked.connect(
                self.browse_model)
            getattr(self, f"spinBox_fpsresolution_{l_idx}").valueChanged.connect(
                self.on_fpsresolution_changed)
            getattr(self, f"checkBox_show_point_{l_idx}").stateChanged.connect(self.on_midpoint_changed)
            # Color buttons
            color_buttons = ["btn_color_target_box", "btn_color_highlight_result_box",
                "btn_color_fps_good", "btn_color_fps_bad",
                "btn_color_result_info_box", "btn_color_result_info_label",
                "btn_color_result_info_text", "btn_color_point",
                "btn_color_background"
            ]
            for btn_name in color_buttons:
                btn = getattr(self, f"{btn_name}_{l_idx}")
                btn.clicked.connect(self.open_color_dialog(btn))
            # Background buttons
            getattr(self, f"comboBox_scenario_name_{l_idx}").currentIndexChanged.connect(
                self.on_background_changed)
            getattr(self, f"btn_browse_scenario_{l_idx}").clicked.connect(
                self.browse_scenario)
            # Encoding and matrix buttons
            getattr(self, f"comboBox_base_{l_idx}").currentTextChanged.connect(
                self.on_base_changed)
            getattr(self, f"comboBox_order_{l_idx}").currentTextChanged.connect(
                self.update_encoding_info)
            getattr(self, f"btn_update_matrix_{l_idx}").clicked.connect(
                self.update_matrix)
            # Train model buttons
            getattr(self, f"btn_train_model_{l_idx}").clicked.connect(
                self.train_model)
            # Train model buttons
            table = getattr(self, f"tableWidget_bpf_{l_idx}")
            table.setContextMenuPolicy(Qt.CustomContextMenu)
            table.customContextMenuRequested.connect(
                self.on_custom_table_menu
            )
            self.curr_layer_idx = l_idx

            # Set settings to GUI
            self.set_settings_to_gui()
            self.on_mode_changed()
            self.on_background_changed()
            self.on_base_changed()
            getattr(self, f"comboBox_order_{l_idx}").setCurrentIndex(4) # default 2^6

        self.on_stimuli_changed()
        self.notifications.new_notification('Default settings loaded')

        # Application ready
        self.setModal(True)
        self.show()

    # --------------------- Settings updating --------------------
    def on_stimuli_changed(self):
        if self.btn_plain.isChecked():
            self.params_stacked_widget.setCurrentIndex(0)
        elif self.btn_grating.isChecked():
            self.params_stacked_widget.setCurrentIndex(1)
        elif self.btn_checkerboard.isChecked():
            self.params_stacked_widget.setCurrentIndex(2)
        self.curr_layer_idx = self.params_stacked_widget.currentIndex() + 1

    def on_mode_changed(self):
        l_idx = self.curr_layer_idx
        combo = getattr(self, f"comboBox_mode_{l_idx}")
        online = combo.currentText() == 'Online'

        show_if_online = [
            f"label_cvep_model_{l_idx}", f"label_test_cycles_{l_idx}",
            f"spinBox_testcycles_{l_idx}", f"lineEdit_cvepmodel_{l_idx}",
            f"btn_browse_cvepmodel_{l_idx}"
        ]
        show_if_train = [
            f"label_train_targets_{l_idx}", f"label_train_cycles_{l_idx}",
            f"spinBox_traincycles_{l_idx}", f"lineEdit_train_targets_{l_idx}"
        ]

        getattr(self, f"lineEdit_session_{l_idx}").setText("Test" if online else "Train")

        for w in show_if_online:
            getattr(self, w).setVisible(online)
        for w in show_if_train:
            getattr(self, w).setVisible(not online)

    def on_fpsresolution_changed(self):
        l_idx = self.curr_layer_idx
        text_edit = getattr(self, f"textEdit_monitor_rates_{l_idx}")
        spin_box = getattr(self, f"spinBox_fpsresolution_{l_idx}")
        self.update_table_cutoffs()
        self.update_encoding_info()

        text_edit.clear()
        monitors = get_monitor_rates()

        if len(monitors) == 0:
            text = (
                f"No connected monitor is detected. The app cannot "
                f"guarantee a real updating using {spin_box.value()} Hz"
            )
            text_edit.append(text)
        else:
            rates = []
            text = "Connected monitors:\n"
            for name, rate in monitors:
                text += f" * {name} - max. {rate} Hz\n"
                rates.append(rate)
            text_edit.append(text)

            if len(rates) > 1 and not np.all(np.array(rates) == rates[0]):
                text_edit.append(
                    "<span style='color: yellow; font-weight: bold;'>"
                    "\n[Warning]: the monitors have different refresh "
                    "rates! The fps_resolution may vary. An exact "
                    "target FPS can only be guaranteed if all monitors "
                    "have the same refresh rate.</span>\n"
                )

            for rate in rates:
                if rate < spin_box.value():
                    text_edit.append(
                        "<span style='color: red; font-weight: bold;'>"
                        "\n[Error]: at least one monitor will not be "
                        "able to reach the desired Target FPS! The "
                        "paradigm will not work.</span>\n"
                    )

    def on_midpoint_changed(self):
        l_idx = self.curr_layer_idx
        if getattr(self, f"checkBox_show_point_{l_idx}").isChecked():
            getattr(self, f"spinBox_point_size_{l_idx}").setEnabled(True)
        else:
            getattr(self, f"spinBox_point_size_{l_idx}").setEnabled(False)

    def on_background_changed(self):
        l_idx = self.curr_layer_idx
        combo = getattr(self, f"comboBox_scenario_name_{l_idx}")
        scenario = combo.currentText()

        btn_color_bg = getattr(self, f"btn_color_background_{l_idx}")
        label_color_bg = getattr(self, f"label_color_background_{l_idx}")
        lineedit_scenario = getattr(self, f"lineEdit_scenario_{l_idx}")
        btn_browse_scenario = getattr(self, f"btn_browse_scenario_{l_idx}")

        if scenario == "Solid Color":
            btn_color_bg.setVisible(True)
            label_color_bg.setVisible(True)
            lineedit_scenario.setVisible(False)
            btn_browse_scenario.setVisible(False)

            gui_utils.modify_property(
                btn_color_bg, 'background-color',
                self.settings.background.color_background[:7]
            )

        elif scenario == "Real Scenario":
            btn_color_bg.setVisible(False)
            label_color_bg.setVisible(False)
            lineedit_scenario.setVisible(True)
            btn_browse_scenario.setVisible(True)

    def on_base_changed(self):
        l_idx = self.curr_layer_idx
        base = int(getattr(self, f"comboBox_base_{l_idx}").currentText())
        orders = list(LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order'].keys())
        orders = [str(x) for x in orders]
        getattr(self, f"comboBox_order_{l_idx}").setEnabled(True)
        getattr(self, f"comboBox_order_{l_idx}").clear()
        getattr(self, f"comboBox_order_{l_idx}").addItems(orders)
        getattr(self, f"comboBox_order_{l_idx}").setCurrentIndex(0)
        self.update_encoding_info()

    def update_encoding_info(self):
        l_idx = self.curr_layer_idx
        base = int(getattr(self, f"comboBox_base_{l_idx}").currentText())
        if getattr(self, f"comboBox_order_{l_idx}").currentIndex() == -1:
            return
        order = int(getattr(self, f"comboBox_order_{l_idx}").currentText())
        seqlen = base ** order - 1
        duration = seqlen /getattr(self, f"spinBox_fpsresolution_{l_idx}").value()
        getattr(self, f"lineEdit_seqlength_{l_idx}").setText(str(seqlen))
        getattr(self, f"lineEdit_cycleduration_{l_idx}").setText(str(duration))

    def set_settings_to_gui(self):
        l_idx = self.curr_layer_idx
        # Run settings
        getattr(self, f"lineEdit_user_{l_idx}").setText(self.settings.run_settings.user)
        getattr(self, f"lineEdit_session_{l_idx}").setText(self.settings.run_settings.session)
        getattr(self, f"spinBox_run_{l_idx}").setValue(self.settings.run_settings.run)
        getattr(self, f"comboBox_mode_{l_idx}").setCurrentText(self.settings.run_settings.mode)
        getattr(self, f"spinBox_traincycles_{l_idx}").setValue(self.settings.run_settings.train_cycles)
        getattr(self, f"lineEdit_train_targets_{l_idx}").setText(self.settings.run_settings.train_target)
        getattr(self, f"spinBox_testcycles_{l_idx}").setValue(self.settings.run_settings.test_cycles)
        getattr(self, f"lineEdit_cvepmodel_{l_idx}").setText(self.settings.run_settings.cvep_model_path)
        getattr(self, f"spinBox_fpsresolution_{l_idx}").setValue(self.settings.run_settings.fps_resolution)
        getattr(self, f"checkBox_photodiode_{l_idx}").setChecked(self.settings.run_settings.enable_photodiode)
        getattr(self, f"checkBox_show_point_{l_idx}").setChecked(self.settings.run_settings.show_point)
        getattr(self, f"spinBox_point_size_{l_idx}").setValue(self.settings.run_settings.point_size)
        # Timings
        getattr(self, f"doubleSpinBox_t_prev_text_{l_idx}").setValue(self.settings.timings.t_prev_text)
        getattr(self, f"doubleSpinBox_t_prev_iddle_{l_idx}").setValue(self.settings.timings.t_prev_iddle)
        getattr(self, f"doubleSpinBox_t_finish_text_{l_idx}").setValue(self.settings.timings.t_finish_text)
        # Colors
        gui_utils.modify_property(getattr(self, f"btn_color_point_{l_idx}"), 'background-color',
                                  self.settings.colors.color_point[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_target_box_{l_idx}"), 'background-color',
                                  self.settings.colors.color_target_box[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_highlight_result_box_{l_idx}"), 'background-color',
                                  self.settings.colors.color_highlight_result_box[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_fps_good_{l_idx}"), 'background-color',
                                  self.settings.colors.color_fps_good[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_fps_bad_{l_idx}"), 'background-color',
                                  self.settings.colors.color_fps_bad[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_result_info_box_{l_idx}"), 'background-color',
                                  self.settings.colors.color_result_info_box[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_result_info_label_{l_idx}"), 'background-color',
                                  self.settings.colors.color_result_info_label[:7])
        gui_utils.modify_property(getattr(self, f"btn_color_result_info_text_{l_idx}"), 'background-color',
                                  self.settings.colors.color_result_info_text[:7])
        # Variable colors
        getattr(self, f"tableWidget_color_sequences_{l_idx}").clear()
        getattr(self, f"tableWidget_color_sequences_{l_idx}").setColumnCount(5)
        getattr(self, f"tableWidget_color_sequences_{l_idx}").setRowCount(
            len(self.settings.colors.color_box_dict.keys()))
        getattr(self, f"tableWidget_color_sequences_{l_idx}").setHorizontalHeaderLabels(
            ['Event', 'Box color', 'Box alpha (%)', 'Text color', 'Text alpha (%)'])
        header = getattr(self, f"tableWidget_color_sequences_{l_idx}").horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        idx = 0
        for event in self.settings.colors.color_box_dict.keys():
            color_box_ = self.settings.colors.color_box_dict[event]
            opacity_box_ = self.settings.colors.opacity_box_dict[event]
            color_text_ = self.settings.colors.color_text_dict[event]
            opacity_text_ = self.settings.colors.opacity_text_dict[event]
            btn_box = QtWidgets.QPushButton('')
            spinBox_alpha_box = QtWidgets.QSpinBox()
            spinBox_alpha_box.setRange(0, 100)
            btn_text = QtWidgets.QPushButton('')
            spinBox_alpha_text = QtWidgets.QSpinBox()
            spinBox_alpha_text.setRange(0, 100)
            gui_utils.modify_property(btn_box, 'background-color', color_box_)
            spinBox_alpha_box.setValue(opacity_box_)
            gui_utils.modify_property(btn_text, 'background-color', color_text_)
            spinBox_alpha_text.setValue(opacity_text_)
            btn_box.clicked.connect(self.open_color_dialog(btn_box))
            btn_text.clicked.connect(self.open_color_dialog(btn_text))
            getattr(self, f"tableWidget_color_sequences_{l_idx}").setItem(idx, 0,
                                                     QtWidgets.QTableWidgetItem(
                                                         str(event)))
            getattr(self, f"tableWidget_color_sequences_{l_idx}").setCellWidget(idx, 1, btn_box)
            getattr(self, f"tableWidget_color_sequences_{l_idx}").setCellWidget(idx, 2, spinBox_alpha_box)
            getattr(self, f"tableWidget_color_sequences_{l_idx}").setCellWidget(idx, 3, btn_text)
            getattr(self, f"tableWidget_color_sequences_{l_idx}").setCellWidget(idx, 4, spinBox_alpha_text)
            idx += 1

        # Background
        getattr(self, f"comboBox_scenario_name_{l_idx}").setCurrentText(self.settings.background.scenario_name)
        gui_utils.modify_property(getattr(self, f"btn_color_background_{l_idx}"), 'background-color',
                                  self.settings.background.color_background[:7])
        getattr(self, f"lineEdit_scenario_{l_idx}").setText(self.settings.background.scenario_path)

        # Useful PyQt policies
        policy_max_pre = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        policy_max_max = QSizePolicy(QSizePolicy.Expanding,
                                     QSizePolicy.Expanding)
        policy_fix_fix = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Matrices
        # Create the required number of tabs
        n_extra = len(self.settings.matrices) - \
                  self.widget_nested_matrices.count()
        for t in range(1, n_extra + 1):
            mtx_widget_ = QtWidgets.QWidget(self.widget_nested_matrices)
            mtx_idx_ = self.widget_nested_matrices.count() + 1
            self.widget_nested_matrices.addTab(mtx_widget_,
                                              'Matrix #' + str(mtx_idx_))
        # Create each matrix
        for m in range(len(self.settings.matrices)):
            # Set the current index and create the general layout
            curr_mtx = self.settings.matrices[m]
            self.widget_nested_matrices.setCurrentIndex(m)
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
                    box_color_ = self.settings.colors.color_box_dict[str(key_)]
                    text_color_ = self.settings.colors.color_text_dict[str(key_)]
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
            self.update_tab(self.widget_nested_matrices, m, new_tab)

        getattr(self, f"spinBox_nrow_{l_idx}").setValue(self.settings.matrices[0].n_row)
        getattr(self, f"spinBox_ncol_{l_idx}").setValue(self.settings.matrices[0].n_col)

        # Filter cutoffs according to fps_resolution
        self.update_table_cutoffs()

    def btn_command_on_click(self, row, col):
        def set_config():
            # This function is required in order to accept passing arguments
            # (function factory)
            current_index = self.widget_nested_matrices.currentIndex()
            target_dialog = TargetConfigDialog(
                self.settings.matrices[current_index].
                    matrix_list[row][col], current_index)
            if target_dialog.exec_():
                # Get the returned values
                self.settings.matrices[
                    current_index].matrix_list[row][col].set_text(
                    target_dialog.input_target_text.text())
                self.settings.matrices[
                    current_index].matrix_list[row][col].set_label(
                    target_dialog.input_target_label.text())
                seq = eval(target_dialog.input_target_sequence.text())
                self.settings.matrices[
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
        self.widget_nested_matrices.setCurrentIndex(0)

    @staticmethod
    def update_matrix_names(tabwidget):
        """ Updates the matrix names in case that some middle matrix
        has been deleted. """
        for tab_idx in range(tabwidget.count()):
            tabwidget.setTabText(tab_idx, 'Matrix #' + str(tab_idx + 1))

    def get_settings_from_gui(self):
        l_idx = self.curr_layer_idx

        # Run settings
        self.settings.run_settings.user = (
            getattr(self, f"lineEdit_user_{l_idx}").text())
        self.settings.run_settings.session = (
            getattr(self, f"lineEdit_session_{l_idx}").text())
        self.settings.run_settings.run = (
            getattr(self, f"spinBox_run_{l_idx}").value())
        self.settings.run_settings.mode = (
            getattr(self, f"comboBox_mode_{l_idx}").currentText())
        self.settings.run_settings.train_cycles = (
            getattr(self, f"spinBox_traincycles_{l_idx}").value())
        self.settings.run_settings.train_target = (
            getattr(self, f"lineEdit_train_targets_{l_idx}").text())
        self.settings.run_settings.test_cycles = (
            getattr(self, f"spinBox_testcycles_{l_idx}").value())
        self.settings.run_settings.cvep_model_path = (
            getattr(self, f"lineEdit_cvepmodel_{l_idx}").text())
        self.settings.run_settings.fps_resolution = (
            getattr(self, f"spinBox_fpsresolution_{l_idx}").value())
        self.settings.run_settings.enable_photodiode = (
            getattr(self, f"checkBox_photodiode_{l_idx}").isChecked())
        self.settings.run_settings.show_point = (
            getattr(self, f"checkBox_show_point_{l_idx}").isChecked())
        self.settings.run_settings.point_size = (
            getattr(self, f"spinBox_point_size_{l_idx}").value())

        # Timings
        self.settings.timings.t_prev_text = (
            getattr(self, f"doubleSpinBox_t_prev_text_{l_idx}").value())
        self.settings.timings.t_prev_iddle = (
            getattr(self, f"doubleSpinBox_t_prev_iddle_{l_idx}").value())
        self.settings.timings.t_finish_text = (
            getattr(self, f"doubleSpinBox_t_finish_text_{l_idx}").value())

        # Colors
        self.settings.colors.color_point = (
            gui_utils.get_property(getattr(self, f"btn_color_point_{l_idx}"), 'background-color'))
        self.settings.colors.color_target_box = (
            gui_utils.get_property(getattr(self, f"btn_color_target_box_{l_idx}"),'background-color'))
        self.settings.colors.color_highlight_result_box = gui_utils.get_property(
            getattr(self, f"btn_color_highlight_result_box_{l_idx}"), 'background-color')
        self.settings.colors.color_fps_good = (
            gui_utils.get_property(getattr(self, f"btn_color_fps_good_{l_idx}"), 'background-color'))
        self.settings.colors.color_fps_bad = (
            gui_utils.get_property(getattr(self, f"btn_color_fps_bad_{l_idx}"),'background-color'))
        self.settings.colors.color_result_info_box = (
            gui_utils.get_property(getattr(self, f"btn_color_result_info_box_{l_idx}"), 'background-color'))
        self.settings.colors.color_result_info_label = (
            gui_utils.get_property(getattr(self, f"btn_color_result_info_label_{l_idx}"), 'background-color'))
        self.settings.colors.color_result_info_text = (
            gui_utils.get_property(getattr(self, f"btn_color_result_info_text_{l_idx}"), 'background-color'))
        # Variable colors
        self.settings.colors.color_box_dict = dict()
        self.settings.colors.opacity_box_dict = dict()
        self.settings.colors.color_text_dict = dict()
        self.settings.colors.opacity_text_dict = dict()
        no_colors = getattr(self, f"tableWidget_color_sequences_{l_idx}").rowCount()
        for i in range(no_colors):
            table_item_event = getattr(self, f"tableWidget_color_sequences_{l_idx}").item(i, 0)
            btn_box_color = getattr(self, f"tableWidget_color_sequences_{l_idx}").cellWidget(i, 1)
            spinBox_box_alpha = getattr(self, f"tableWidget_color_sequences_{l_idx}").cellWidget(i, 2)
            btn_text_color = getattr(self, f"tableWidget_color_sequences_{l_idx}").cellWidget(i, 3)
            spinBox_text_alpha = getattr(self, f"tableWidget_color_sequences_{l_idx}").cellWidget(i, 4)
            key = int(table_item_event.text())
            box_color = gui_utils.get_property(btn_box_color, 'background-color')
            box_opacity = spinBox_box_alpha.value()
            text_color = gui_utils.get_property(btn_text_color, 'background-color')
            text_opacity = spinBox_text_alpha.value()
            self.settings.colors.color_box_dict[str(key)] = box_color
            self.settings.colors.opacity_box_dict[str(key)] = box_opacity
            self.settings.colors.color_text_dict[str(key)] = text_color
            self.settings.colors.opacity_text_dict[str(key)] = text_opacity

        # Background
        self.settings.background.scenario_name = (
            getattr(self, f"comboBox_scenario_name_{l_idx}").currentText())
        self.settings.background.color_background = (
            gui_utils.get_property(getattr(self, f"btn_color_background_{l_idx}"), 'background-color'))
        self.settings.background.scenario_path = (
            getattr(self, f"lineEdit_scenario_{l_idx}").text())

    def update_gui(self):
        self.get_settings_from_gui()
        self.set_settings_to_gui()

    def update_table_cutoffs(self):
        table = getattr(self, f"tableWidget_bpf_{self.curr_layer_idx}")
        fps_half = getattr(self, f"spinBox_fpsresolution_{self.curr_layer_idx}").value()/2
        for i in range(table.rowCount()):
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(fps_half)))

    def on_custom_table_menu(self, pos):
        # Get action
        table = getattr(self, f"tableWidget_bpf_{self.curr_layer_idx}")
        menu = QtWidgets.QMenu()
        delete_row_action = menu.addAction("Delete row")
        add_row_action = menu.addAction("Add row")
        action = menu.exec_(table.viewport().mapToGlobal(pos))

        # Delete row action
        if action == delete_row_action:
            it = table.itemAt(pos)
            if it is None or table.rowCount() == 1:
                return
            r = it.row()
            item_range = QtWidgets.QTableWidgetSelectionRange(
                r, 0, r, table.columnCount() - 1)
            table.setRangeSelected(item_range, True)
            table.removeRow(r)
            table.setVerticalHeaderLabels(
                [str(x) for x in range(1, table.rowCount() + 1)])
        # Add row action (and populate, otherwise won't work)
        if action == add_row_action:
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QtWidgets.QTableWidgetItem('0'))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(
                str(int(self.settings.run_settings.fps_resolution / 2))))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem('7'))
            table.setItem(r, 3, QtWidgets.QTableWidgetItem('bandpass'))
            table.setVerticalHeaderLabels(
                [str(x) for x in range(1, table.rowCount() + 1)])

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
        l_idx = self.curr_layer_idx
        def check_train_feasible():
            error_msg = ''
            if getattr(self, f"tableWidget_bpf_{l_idx}").rowCount() == 0:
                error_msg += 'Cannot train if we do not have, at least, ' \
                             'one filter!\n'
            for i in range(getattr(self, f"tableWidget_bpf_{l_idx}").rowCount()):
                cut1 = float(getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 0).text())
                cut2 = float(getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 1).text())
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
            for i in range(getattr(self, f"tableWidget_bpf_{l_idx}").rowCount()):
                cut1 = float(getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 0).text())
                cut2 = float(getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 1).text())
                order = int(getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 2).text())
                type = getattr(self, f"tableWidget_bpf_{l_idx}").item(i, 3).text()
                bpf.append([order, (cut1, cut2)])
                if cut2 > max_cut2:
                    max_cut2 = cut2
            notch = [7, (getattr(self, f"doubleSpinBox_notch_{l_idx}").value() - 1,
                         getattr(self, f"doubleSpinBox_notch_{l_idx}").value() + 1)]

            # Check if notch is required
            if max_cut2 < notch[1][0]:
                notch = None

            # Train the model
            art_rej = None
            if getattr(self, f"checkBox_calibration_art_rej_{l_idx}").isChecked():
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
                getattr(self, f"lineEdit_cvepmodel_{l_idx}").setText(fname[0])

    def browse_model(self):
        filt = "c-VEP Model (*.cvep.mdl)"
        directory = os.getcwd() + "/../models/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Created directory %s!' % directory)
        filepath = QtWidgets.QFileDialog.getOpenFileName(caption="c-VEP Model",
                                                         dir=directory,
                                                         filter=filt)
        getattr(self, f"lineEdit_cvepmodel_{self.curr_layer_idx}").setText(filepath[0])

    def browse_scenario(self):
        filt = "Image (*.jpg *.jpeg *.png)"
        directory = os.path.dirname(__file__) + "/background/"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('Created directory %s!' % directory)
        filepath = QtWidgets.QFileDialog.getOpenFileName(caption="Scenario",
                                                         dir=directory,
                                                         filter=filt)
        getattr(self, f"lineEdit_scenario_{self.curr_layer_idx}").setText(filepath[0])

    def update_matrix(self):
        l_idx = self.curr_layer_idx
        # Get the parameters
        n_row = int(getattr(self, f"spinBox_nrow_{l_idx}").value())
        n_col = int(getattr(self, f"spinBox_ncol_{l_idx}").value())
        base = int(getattr(self, f"comboBox_base_{l_idx}").currentText())
        order = int(getattr(self, f"comboBox_order_{l_idx}").currentText())
        mseqlen = int(getattr(self, f"lineEdit_seqlength_{l_idx}").text())

        # Compute the matrices
        self.get_settings_from_gui()
        self.settings.matrices = self.settings.build_with_pary_sequences(
            n_row=n_row, n_col=n_col, base=base, order=order)
        unique = self.settings.get_unique_sequence_values()
        c_box, op_box, c_text, op_text = settings.Colors.generate_grey_color_dicts(unique)
        self.settings.colors = settings.Colors(color_box_dict=c_box, opacity_box_dict=op_box,
                             color_text_dict=c_text, opacity_text_dict=op_text)

        # Check if everything is correct
        lags_info =  self.settings.matrices[0].info_lags
        if lags_info['tau'] < 1:
            error_msg = 'Cannot encode all the commands (%i) with that ' \
                        'sequence length (%i)! Decrease the number of ' \
                        'commands or increase the sequence length to have ' \
                        'enough room to get a positive delay.' % \
                        (n_col * n_row, mseqlen)
            error_dialog(error_msg, 'Oops!')
            return
        if round(lags_info['tau']) == 1:
            warn_msg = 'With that number of commands (%i) and that sequence ' \
                       'length (%i), the delay between shifted-version ' \
                       'sequences will be approx. 1. Consider to decrease the' \
                       ' number of commands or increase the sequence length ' \
                       'to space more the shifted-sequences and favor the ' \
                       'performance.' \
                       % (n_col * n_row, mseqlen)
            warning_dialog(warn_msg, 'Be careful!')

        # Update the gui
        self.set_settings_to_gui()

        # Show the encoding
        monitor_rate = float(getattr(self, f"spinBox_fpsresolution_{l_idx}").value())
        current_index = self.widget_nested_matrices.currentIndex()
        visualize_dialog = VisualizeEncodingDialog(
            n_row=n_row, n_col=n_col, base=base, order=order,
            monitor_rate=monitor_rate, item_list=self.settings.matrices[current_index].item_list,
            lags_info=lags_info)
        if visualize_dialog.exec_():
            if any(item in lags_info['bad_lags'] for item in lags_info['lags']):
                warn_msg = "Careful, I could not optimize the lags enough for" \
                           " this configuration, so the correlation for" \
                           " command(s) %s is not minimum!" % ','.join(
                    visualize_dialog.bad_cmds)
                warning_dialog(warn_msg, 'Be careful!')

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

