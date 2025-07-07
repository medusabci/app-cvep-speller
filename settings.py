from medusa.components import SerializableComponent
from medusa.bci.cvep_spellers import LFSR, LFSR_PRIMITIVE_POLYNOMIALS
from .app_constants import *
import numpy as np
import os
import math
import json


class Settings(SerializableComponent):

    def __init__(self, connection_settings=None, run_settings=None,
                 timings=None, colors=None, background=None, matrices=None):
        self.connection_settings = connection_settings if \
            connection_settings is not None else ConnectionSettings()
        self.run_settings = run_settings if \
            run_settings is not None else RunSettings()
        self.timings = timings if timings is not None else Timings()
        self.colors = colors if colors is not None else Colors()
        self.background = background if background is not None else Background()

        self.matrices = matrices
        if matrices is None:
            self.matrices = self.standard_single_sequence_matrices()

    def to_serializable_obj(self):
        matrices = []
        for matrix in self.matrices:
            matrices.append(matrix.serialize())

        sett_dict = {'connection_settings': self.connection_settings.__dict__,
                     'run_settings': self.run_settings.__dict__,
                     'timings': self.timings.__dict__,
                     'matrices': matrices,
                     'colors': self.colors.__dict__,
                     'background': self.background.__dict__
                     }
        return sett_dict

    @staticmethod
    def from_serializable_obj(settings_dict):
        # Easy dicts
        conn_sett = ConnectionSettings(**settings_dict['connection_settings'])
        run_sett = RunSettings(**settings_dict['run_settings'])
        timings = Timings(**settings_dict['timings'])
        colors = Colors(**settings_dict['colors'])
        background = Background(**settings_dict['background'])
        # Matrices
        item_list = list()
        m = settings_dict['matrices']
        for i in m['item_list']:
            target = CVEPTarget(text=i['text'],
                                label=i['label'],
                                sequence=i['sequence'],
                                lag=i['lag'])
            item_list.append(target)
        matrix = CVEPMatrix(n_row=m['n_row'], n_col=m['n_col'], info_lags=m['info_lags'])
        matrix.item_list = item_list
        matrix.organize_matrix()
        matrices.append(matrix)
        return Settings(connection_settings=conn_sett,
                        run_settings=run_sett,
                        timings=timings,
                        colors=colors,
                        background=background,
                        matrices=matrices)

    @staticmethod
    def get_coords_from_labels(labels, matrices):
        coords = []
        for label in labels:
            label_coord = None
            for idx, matrix in enumerate(matrices):
                target = matrix.get_target_from_label(label)
                if len(target) > 1:
                    print('WARNING in get_codes_from_labels: more than one '
                          'command for label %s, taking the first one' % label)
                if len(target) > 0:
                    label_coord = [idx, target[0].row, target[0].col]
                    break
            if label_coord is None:
                raise ValueError("Label %s not found" % label)
            coords.append(label_coord)
        return coords

    @staticmethod
    def standard_single_sequence_matrices(n_row=4, n_col=4, mseqlen=63):
        """ Computes a predefined standard c-VEP matrix that modulates commands
        using a single-sequence via circular shifting.

        Parameter
        -----------
        n_row: int
            Number of rows.
        n_col: int
            Number of columns.
        mseqlen: int
            Length of the binary m-sequence (supported 31, 63, 127, 255)

        Returns
        --------
        matrix : CVEPMatrix
            Structured matrix object.
        """
        # Number of commands supported
        no_commands = n_row * n_col
        tau = mseqlen / no_commands
        if tau < 1:
            raise ValueError('[cvep_speller/settings] Sequence length is not '
                             'enough to encode all commands. Please, reduce '
                             'the number of commands (%i) or increment the '
                             'sequence length (supported lengths: 31, 63, '
                             '127 or 255)' % no_commands)

        # Init
        comms = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/*-+.,' \
                '_abcdefghijklmnopqrstuvwxyz'
        comms *= 20
        comms_ = comms[:no_commands]
        lags = np.linspace(0, mseqlen, no_commands + 1)[:-1].astype(int).tolist()

        # M-sequence generation
        if mseqlen == 31:
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][5]
            m_seq = LFSR(poly_, base=2)
        elif mseqlen == 63:
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][6]
            m_seq = LFSR(poly_, base=2, seed=[1, 1, 1, 1, 1, 0])
        elif mseqlen == 127:
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][7]
            m_seq = LFSR(poly_, base=2)
        elif mseqlen == 255:
            poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][2]['order'][8]
            m_seq = LFSR(poly_, base=2)
        else:
            raise ValueError('[cvep_speller/settings] Sequence length of %i '
                             'not supported (use 31, 63, 127 or 255)!' %
                             mseqlen)
        seq = m_seq.sequence

        info_lags = {
            'tau': tau,
            'lags': lags
        }
        # Set up the matrix
        matrix = CVEPMatrix(n_row, n_col, info_lags=info_lags)
        for idx, c in enumerate(comms_):
            seq_ = circular_shift(sequence=seq, lag=lags[idx])
            target = CVEPTarget(text=c, label=c, sequence=seq_)
            matrix.append(target)
        matrix.organize_matrix()

        matrices = [matrix]
        return matrices


class ConnectionSettings:

    def __init__(self, ip="127.0.0.1", port=50000):
        self.ip = ip
        self.port = port

class RunSettings:
    def __init__(self, user="S0X", session="Train", run=1,
                 mode=TRAIN_MODE,
                 enable_photodiode=True,
                 train_cycles=10, train_target='AAAAA',
                 test_cycles=10,
                 cvep_model_path='',
                 fps_resolution=60):
        self.user = user
        self.session = session
        self.run = run
        self.mode = mode
        self.enable_photodiode = enable_photodiode
        self.train_cycles = train_cycles
        self.train_target = train_target
        self.test_cycles = test_cycles
        self.cvep_model_path = cvep_model_path
        self.fps_resolution = fps_resolution

class Timings:

    def __init__(self, t_prev_text=1.0, t_prev_iddle=1.0, t_finish_text=1.0):
        self.t_prev_text = t_prev_text
        self.t_prev_iddle = t_prev_iddle
        self.t_finish_text = t_finish_text


class Colors:

    def __init__(self, color_target_box='#ff195bff',
                 color_highlight_result_box='#03fc5aff',
                 color_result_info_box='#8c8c8cff',
                 color_result_info_label='#b7b7b7ff',
                 color_result_info_text='#f4f657ff',
                 color_fps_good='#5ee57dff',
                 color_fps_bad='#b43228ff',
                 color_box_0='#000000',
                 color_op_box_0=100,
                 color_box_1='#ffffff',
                 color_op_box_1=100,
                 color_text_0='#ffffff',
                 color_op_text_0=100,
                 color_text_1='#000000',
                 color_op_text_1=100):
        self.color_target_box = color_target_box
        self.color_highlight_result_box = color_highlight_result_box
        self.color_result_info_box = color_result_info_box
        self.color_result_info_label = color_result_info_label
        self.color_result_info_text = color_result_info_text
        self.color_fps_good = color_fps_good
        self.color_fps_bad = color_fps_bad
        self.color_box_0 = color_box_0
        self.color_op_box_0 = color_op_box_0
        self.color_box_1 = color_box_1
        self.color_op_box_1 = color_op_box_1
        self.color_text_0 = color_text_0
        self.color_op_text_0 = color_op_text_0
        self.color_text_1 = color_text_1
        self.color_op_text_1 = color_op_text_1

    @staticmethod
    def concat_dict(dict_):
        concat = []
        for key in dict_:
            concat.append([str(key), str(dict_[key])])
        return concat

class Background:
    def __init__(self, scenario_name='Solid Color', color_background='#a3bec7', scenario_path=os.path.dirname(__file__) + "/background/Scenario-Example.jpg"):
        self.scenario_name = scenario_name
        self.color_background = color_background
        self.scenario_path = scenario_path

class CVEPMatrix:

    def __init__(self, n_row=-1, n_col=-1, info_lags=None):
        """ Class that represents a c-VEP matrix.

        Attribute `item_list` encompasses a vector of commands, while
        `matrix_list` contains these same commands arranged in a matrix-like
        fashion to be accessed using `row` and `col` as indexes. Therefore,
        it is mandatory to call `organize_matrix()` sometime.

        Parameters
        ----------
        n_row: int
            (Optional, default:-1) Number of rows of the output matrix.
            If -1, `n_row` will be taken from the class.
        n_col: int
            (Optional, default:-1) Number of columns of the output matrix.
            If -1, `n_col` will be taken from the class.
        info_lags : dict()
            (Optional, default: None) Sometimes is useful to store additional
            info such as tau or lag positions for circular shifted c-VEPs.
        """
        self.n_row = n_row
        self.n_col = n_col
        self.info_lags = info_lags

        self.item_list = []  # Vector of targets
        self.matrix_list = []  # Matrix of targets.
        # Each target can be accessed matrix.matrix_list[row][col]

    def remove(self, index):
        """ Removes a CVEPTarget element from the list of targets."""
        self.item_list.pop(index)

    def append(self, new_item):
        """ Appends a new CVEPTarget item to the list of targets. """
        if type(new_item) != CVEPTarget:
            raise ValueError('Cannot append, object type is not CVEPTarget.')
        self.item_list.append(new_item)

    def organize_matrix(self, n_row=-1, n_col=-1):
        """ Arranges the target list into a matrix.

        Parameters
        ----------
        n_row: int
            (Optional, default:-1) Number of rows of the output matrix.
            If -1, `n_row` will be taken from the class.
        n_col: int
            (Optional, default:-1) Number of columns of the output matrix.
            If -1, `n_col` will be taken from the class.

        Returns
        -------
        matrix_list: 2D list of CVEPTarget items
            Matrix-shaped target list.
        """
        # If n_row or n_col are set to -1, take the current values
        if n_row == -1:
            n_row = self.n_row
        if n_col == -1:
            n_col = self.n_col
        # Check if the list of targets can be reorganized in n_row and n_col
        if len(self.item_list) != (n_col * n_row):
            raise ValueError('Cannot organize elements in matrix. Number of '
                             'elements (%d) does not match the product of'
                             ' %d rows and %d columns.' % (len(self.item_list),
                                                           n_row, n_col))

        # Re-organize matrix
        r_count = c_count = 0
        self.matrix_list = [[None for y in range(n_col)] for x in range(n_row)]
        for element in self.item_list:
            element.set_row(r_count)
            element.set_col(c_count)
            self.matrix_list[r_count][c_count] = element
            c_count += 1
            if c_count >= n_col:
                c_count = 0
                r_count += 1
        return self.matrix_list

    def get_target_from_label(self, label):
        """ This method returns a list of items for the given label. """
        target = []
        for row in self.matrix_list:
            for item in row:
                if label == item.label:
                    target.append(item)
        return target

    def get_row_col_from_idx(self, idx):
        """ This function returns the [row, col] for a item_list index. """
        row = math.floor(idx / self.n_col)
        col = idx % self.n_col
        return [row, col]

    def serialize(self):
        items = []
        for i in self.item_list:
            items.append(i.to_dict())
        return {"n_row": self.n_row,
                "n_col": self.n_col,
                "info_lags": self.info_lags,
                "item_list": items}


class CVEPTarget:

    def __init__(self, row=-1, col=-1, text='', label='', sequence=None):
        """ Class that represents a target cell of the c-VEP speller matrix.

        Parameters
        ----------
        row: int
            Row of the target. It can be specified later using `set_row` method.
        col: int
            Column of the target. It can be specified later using `set_col`
             method.
        text: basestring
            Text that will be displayed in the target cell.
        label: basestring
            Label that identifies the target cell.
        sequence : list
            Sequence that modulates this target item. The modulation will
            always start with the first index, i.e., sequence[0]
        """

        # Useful parameters
        self.row = row
        self.col = col
        self.text = text
        self.label = label
        self.sequence = sequence

    def set_row(self, row):
        self.row = row

    def set_col(self, col):
        self.col = col

    def set_text(self, text):
        self.text = text

    def set_label(self, label):
        self.label = label

    def set_sequence(self, sequence):
        self.sequence = sequence

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.__dict__)


def circular_shift(sequence, lag):
    """ Shifts circularly a sequence list the desired lag to the left.

    Parameters
    ----------
    sequence : list
        Input sequence to shift.
    lag : int
        Lag to be shifted.

    Returns
    ---------
    shifted_sequence: list
        Shifted sequence. E.g., sequence = [0, 1, 2, 3, 4] and lag = 1 will
        return [1, 2, 3, 4, 0]
    """
    return np.roll(sequence, -lag).tolist()
