import sys
import numpy as np
from sklearn.metrics import confusion_matrix

class Metrics(dict):
    def __init__(self, *args, scale=1, order=['mIoU', 'OA', 'mACC'], **kwargs):
        super(Metrics, self).__init__(*args, **kwargs)
        self.scale = scale
        self.order = [order] if isinstance(order, str) else list(order)  # the importance rank of metrics - main key = order[0]
    # def __missing__(self, key):
    #     return None

    # Comparison
    # ------------------------------------------------------------------------------------------------------------------

    def _is_valid(self, other, raise_invalid=True):
        if self.order[0] not in other:
            if raise_invalid:
                raise ValueError(f'missing main key - {self.order[0]}, in order {self.order}')
            return False
        return True

    def __eq__(self, other):  # care only the main key
        self._is_valid(self)
        self._is_valid(other)
        return self[self.order[0]] == other[self.order[0]]

    def __gt__(self, other):
        self._is_valid(self)
        self._is_valid(other)
        for k in self.order:
            if k not in self:  # skip if not available
                continue
            if k not in other or self[k] > other[k]:  # True if more completed
                return True
            elif self[k] < other[k]:
                return False

        # all equal (at least for main key)
        return False

    # Pretty print
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def scalar_str(self):
        scalar_m = [k for k in ['mIoU', 'OA', 'mACC'] if k in self and self[k]]
        s = ''.join([f'{k}={self[k]/self.scale*100:<6.2f}' for k in scalar_m])
        return s
    @property
    def list_str(self):
        list_m = [k for k in ['IoUs'] if k in self and self[k] is not None]
        s = []
        for k in list_m:
            m = self.list_to_line(k)
            s += [m]
        s = ' | '.join(s)
        return s
    @property
    def final_str(self):
        s = str(self)
        s = ['-' * len(s), s, '-' * len(s)]
        if 'ACCs' in self:
            s = ['ACCs = ' + self.list_to_line('ACCs')] + s
        return '\n'.join(s)
 
    def print(self, full=False, conf=True):
        s = self.full() if full else self.final_str
        if conf and 'conf' in self:
            conf = self['conf']
            assert np.issubdtype(conf.dtype, np.integer)
            with np.printoptions(linewidth=sys.maxsize, precision=3):
                print(self['conf'])
        print(s)

    def full(self, get_list=False):
        # separate line print each group of metrics
        scalar_m = [k for k in ['OA', 'mACC', 'mIoU'] if k in self and self[k]]
        name_d = {'IoUs': 'mIoU', 'ACCs':'mACC'}  # list_m -> scalar_m

        str_d = {k: f'{k}={self[k]/self.scale*100 if self[k] < 1 else self[k]:<6.2f}' for k in scalar_m}  # scalar_m -> str
        for k_list, k_scalar in name_d.items():
            str_d[k_scalar] += ' | ' + self.list_to_line(k_list)

        max_len = max(len(v) for v in str_d.values())
        s = ['-' * max_len, *[v for v in str_d.values()], '-' * max_len]
        s = s if get_list else '\n'.join(s)
        return s

    def __repr__(self):
        return ' | '.join([k for k in [self.scalar_str, self.list_str] if k])

    def list_to_line(self, k):
        l = k if isinstance(k, list) else self[k] if k in self else None
        m = ' '.join([f'{i/self.scale*100:<5.2f}' if i < 1 else f'{i:<5.2f}' for i in l]) if l is not None else ''
        return m
