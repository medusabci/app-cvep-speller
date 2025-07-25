from medusa.components import SerializableComponent
from medusa.bci.cvep_spellers import LFSR, LFSR_PRIMITIVE_POLYNOMIALS, Burst
from gui import gui_utils as gu
from .app_constants import *
import numpy as np
import os
import base64
import io
from PIL import Image
import math
import json


class Settings(SerializableComponent):

    def __init__(self, connection_settings=None, run_settings=None,
                 timings=None, colors=None, background=None,
                 encoding_settings=None, stimulus=None):
        self.connection_settings = connection_settings if \
            connection_settings is not None else ConnectionSettings()
        self.run_settings = run_settings if \
            run_settings is not None else RunSettings()
        self.timings = timings if timings is not None else Timings()
        self.colors = colors if colors is not None else Colors()
        self.background = background if background is not None else Background()
        self.encoding_settings = encoding_settings or EncodingSettings()
        self.stimulus = stimulus
        if stimulus is None:
            unique = self.encoding_settings.get_unique_sequence_values()
            c_box, op_box, c_text, op_text = Stimulus.generate_grey_color_dicts(unique)
            self.stimulus = Stimulus(stimulus_box_dict=c_box, opacity_box_dict=op_box,
                                 color_text_dict=c_text, opacity_text_dict=op_text)

    def to_serializable_obj(self):
        matrices = [m.serialize() for m in self.encoding_settings.matrices]

        sett_dict = {'connection_settings': self.connection_settings.__dict__,
                     'run_settings': self.run_settings.__dict__,
                     'timings': self.timings.__dict__,
                     'colors': self.colors.__dict__,
                     'background': self.background.__dict__,
                     'encoding_settings': {
                         'seq_type': self.encoding_settings.seq_type,
                         'matrices': matrices}
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
        matrices = []
        item_list = list()
        m = settings_dict['enconding_settings']['matrices']
        for i in m['item_list']:
            target = CVEPTarget(text=i['text'],
                                label=i['label'],
                                sequence=i['sequence'])
            item_list.append(target)
        matrix = CVEPMatrix(n_row=m['n_row'], n_col=m['n_col'], info_seq=m['info_seq'])
        matrix.item_list = item_list
        matrix.organize_matrix()
        matrices.append(matrix)
        encoding_settings = EncodingSettings(
            seq_type=settings_dict['encoding_settings']['seq_type'],
            matrices=matrices
        )
        return Settings(connection_settings=conn_sett,
                        run_settings=run_sett,
                        timings=timings,
                        colors=colors,
                        background=background,
                        encoding_settings=encoding_settings)

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
                 fps_resolution=60,
                 show_point=True,
                 point_size=8):
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
        self.show_point = show_point
        self.point_size = point_size

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
                 color_point='#800000'):
        self.color_target_box = color_target_box
        self.color_highlight_result_box = color_highlight_result_box
        self.color_result_info_box = color_result_info_box
        self.color_result_info_label = color_result_info_label
        self.color_result_info_text = color_result_info_text
        self.color_fps_good = color_fps_good
        self.color_fps_bad = color_fps_bad
        self.color_point = color_point

class Background:
    def __init__(self, scenario_name='Solid Color', color_background='#a3bec7', scenario_path=os.path.dirname(__file__) + "/background/Scenario-Example.jpg"):
        self.scenario_name = scenario_name
        self.color_background = color_background
        self.scenario_path = scenario_path

class EncodingSettings:
    def __init__(self, seq_type='M-sequence', matrices=None):
        self.seq_type = seq_type # 'M-sequence' or 'Burst sequence'
        self.matrices = matrices
        if matrices is None:
            if self.seq_type == 'M-sequence':
                self.matrices = self.build_with_pary_sequences()
            elif self.seq_type == 'Burst sequence':
                self.matrices = self.build_with_burst_sequences()

    @staticmethod
    def build_with_pary_sequences(n_row=4, n_col=4, base=2, order=6):
        """ Computes a predefined standard c-VEP matrix that modulates commands
        using a single-sequence via circular shifting.

        Parameter
        -----------
        n_row: int
            Number of rows.
        n_col: int
            Number of columns.
        base: int
            Base of the sequence (must be prime).
        order: int
            Order of the sequence (length will be L = base^order - 1).

        Returns
        --------
        matrix : CVEPMatrix
            Structured matrix object.
        """
        # Number of commands supported
        no_commands = n_row * n_col
        mseqlen = base ** order - 1
        tau = mseqlen / no_commands

        # Init
        comms = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/*-+.,' \
                '_abcdefghijklmnopqrstuvwxyz'
        comms *= 20
        comms_ = comms[:no_commands]

        # M-sequence generation
        if base not in LFSR_PRIMITIVE_POLYNOMIALS['base']:
            raise ValueError('[cvep_speller/settings] Base %i not supported!'
                             % base)
        if order not in LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order']:
            raise ValueError('[cvep_speller/settings] Cannot find any '
                             'primitive polynomial of order %i for base %i'
                             % (order, base))
        poly_ = LFSR_PRIMITIVE_POLYNOMIALS['base'][base]['order'][order]
        seq = LFSR(poly_, base=base).sequence
        centered_seq = LFSR(poly_, base=base, center=True).sequence

        # Optimize correlations
        rxx_, tr_ = EncodingSettings.autocorr_circular(centered_seq)
        rxx_ = rxx_ / np.max(np.abs(rxx_))
        half_rxx = rxx_[int(len(rxx_) / 2):]
        min_p = - min(np.abs(np.unique(half_rxx)))
        bad_indexes = np.where(half_rxx != min_p)[0][1:]
        lags = np.linspace(0, len(seq), no_commands + 1)[:-1].astype(int)
        if any(item in bad_indexes for item in lags):
            # Need to optimize: there are non-minimal correlations
            tau = (len(seq) - len(bad_indexes)) / no_commands
            temp_lags = np.linspace(0, len(seq) - len(bad_indexes),
                                    no_commands + 1).astype(int)[:-1]

            good_idxs = np.arange(0, len(seq))
            good_idxs = np.delete(good_idxs, bad_indexes)
            lags = good_idxs[temp_lags]
        info_lags = {
            'tau': tau,
            'lags': lags.tolist(),
            'bad_lags': bad_indexes.tolist(),
        }
        if tau <= 1:
            print('[cvep_speller/settings] Sequence length is not enough to '
                  'encode all commands. Please, reduce the number of commands '
                  '(%i) or increment the sequence length (supported lengths: '
                  '31, 63, 127 or 255)' % no_commands)
        if any(item in bad_indexes for item in lags):
            print('[cvep_speller/settings] Could not find any optimal '
                  'distributions of lags for this configuration!')

        # Set up the matrix
        matrix = CVEPMatrix(n_row, n_col, info_seq=info_lags)
        for idx, c in enumerate(comms_):
            seq_ = circular_shift(sequence=seq, lag=lags[idx])
            target = CVEPTarget(text=c, label=c, sequence=seq_)
            matrix.append(target)
        matrix.organize_matrix()

        matrices = [matrix]
        return matrices

    @staticmethod
    def build_with_burst_sequences(n_row=4, n_col=4, burstseqlen=132, n_burst=4, f_burst=1):
        """ Computes a predefined standard c-VEP matrix that modulates commands
                using multiple burst sequences.

                Parameter
                -----------
                n_row: int
                    Number of rows.
                n_col: int
                    Number of columns.
                seq_len: int
                    Sequence length in bits.
                n_burst: int
                    Number of bursts per sequence.
                f_burst: int
                    Duration of each burst in frames.

                Returns
                --------
                matrix : CVEPMatrix
                    Structured matrix object.
                """
        # Number of commands supported
        no_commands = n_row * n_col
        # Init
        comms = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/*-+.,' \
                '_abcdefghijklmnopqrstuvwxyz'
        comms *= 20
        comms_ = comms[:no_commands]

        # Burst sequenece generation
        seqs, info_burst = Burst.gen_codes(n=no_commands, length=burstseqlen, n_bursts=n_burst, f_burst=f_burst)

        # Set up the matrix
        matrix = CVEPMatrix(n_row, n_col, info_seq=info_burst)
        for idx, c in enumerate(comms_):
            seq_ = seqs[idx]
            target = CVEPTarget(text=c, label=c, sequence=seq_)
            matrix.append(target)
        matrix.organize_matrix()

        matrices = [matrix]
        return matrices

    def get_unique_sequence_values(self):
        unique_values = []
        for m in self.matrices:
            for i in m.item_list:
                unique_values += list(np.unique(i.sequence))
        return list(np.unique(unique_values))

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

class Stimulus:
    def __init__(self, stimulus_type='Plain',
                 stimulus_box_dict=None, opacity_box_dict=None,
                 color_text_dict=None, opacity_text_dict=None):

        self.stimulus_type = stimulus_type # 'Plain', 'Grating', 'Checkerboard' or 'Customize'
        self.stimulus_box_dict = stimulus_box_dict
        self.opacity_box_dict = opacity_box_dict
        self.color_text_dict = color_text_dict
        self.opacity_text_dict = opacity_text_dict

    @staticmethod
    def generate_grey_color_dicts(unique_sequence_values):
        # Init color
        init_grey = (255, 255, 255)
        end_grey = (0, 0, 0)
        init_hex = gu.rgb_to_hex(init_grey)
        end_hex = gu.rgb_to_hex(end_grey)
        init_grey = gu.rgb_to_hsv(init_grey)
        end_grey = gu.rgb_to_hsv(end_grey)
        # HSV gradient
        h = np.linspace(init_grey[0], end_grey[0], len(unique_sequence_values))
        s = np.linspace(init_grey[1], end_grey[1], len(unique_sequence_values))
        v = np.linspace(init_grey[2], end_grey[2], len(unique_sequence_values))
        # Genetare the HEX dictionary
        color_text_dict = dict()
        stimulus_box_dict = dict()
        opacity_box_dict = dict()
        opacity_text_dict = dict()
        for i in range(len(unique_sequence_values)):
            rgb1 = np.round(gu.hsv_to_rgb((h[i], s[i], v[i]))).astype(int)
            hex1 = gu.rgb_to_hex(tuple(rgb1))
            blob = Stimulus.generate_image_blob_from_color(hex1)
            stimulus_box_dict[str(unique_sequence_values[i])] = blob
            color_text_dict[str(unique_sequence_values[i])] = end_hex if v[i] > 50 else init_hex
            opacity_box_dict[str(unique_sequence_values[i])] = 100
            opacity_text_dict[str(unique_sequence_values[i])] = 100
        return stimulus_box_dict, opacity_box_dict, color_text_dict, opacity_text_dict

    @staticmethod
    def generate_grating_dicts():
        color_text_dict = dict()
        stimulus_box_dict = dict()
        opacity_box_dict = dict()
        opacity_text_dict = dict()
        blob_gray = Stimulus.generate_image_blob_from_color('#808080')
        blob_gabor = Stimulus.generate_image_blob_from_file(os.path.dirname(__file__) + "/stimulus/gabor.png")
        stimulus_box_dict[str(0)] = blob_gray
        opacity_box_dict[str(0)] = 100
        color_text_dict[str(0)] = '#000000'
        opacity_text_dict[str(0)] = 100
        stimulus_box_dict[str(1)] = blob_gabor
        opacity_box_dict[str(1)] = 100
        color_text_dict[str(1)] = '#000000'
        opacity_text_dict[str(1)] = 100
        return stimulus_box_dict, opacity_box_dict, color_text_dict, opacity_text_dict

    @staticmethod
    def generate_empty_dicts(unique_sequence_values):
        color_text_dict = dict()
        stimulus_box_dict = dict()
        opacity_box_dict = dict()
        opacity_text_dict = dict()
        for i in range(len(unique_sequence_values)):
            stimulus_box_dict[str(unique_sequence_values[i])] = ''
            color_text_dict[str(unique_sequence_values[i])] = ''
            opacity_box_dict[str(unique_sequence_values[i])] = 100
            opacity_text_dict[str(unique_sequence_values[i])] = 100
        return stimulus_box_dict, opacity_box_dict, color_text_dict, opacity_text_dict

    @staticmethod
    def concat_dict(dict_):
        concat = []
        for key in dict_:
            concat.append([str(key), str(dict_[key])])
        return concat

    @staticmethod
    def generate_image_blob_from_color(color_hex: str, size=(150, 150)) -> str:
        img = Image.new("RGB", size, color_hex)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    @staticmethod
    def generate_image_blob_from_file(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

class CVEPMatrix:

    def __init__(self, n_row=-1, n_col=-1, info_seq=None):
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
        info_seq : dict()
            (Optional, default: None) Sometimes is useful to store additional
            info from the sequences such as tau or lag positions for circular
            shifted c-VEPs or burst onsets, distances, etc. for burst based
            c-VEPs.
        """
        self.n_row = n_row
        self.n_col = n_col
        self.info_seq = info_seq

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
                "info_seq": self.info_seq,
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
