from scipy.signal import tf2ss as tf2ss1


class ss:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D


def tf2ss(num, den):
    ABCD = tf2ss1(num, den)
    return ss(*[float(arg[0][0]) for arg in ABCD])
