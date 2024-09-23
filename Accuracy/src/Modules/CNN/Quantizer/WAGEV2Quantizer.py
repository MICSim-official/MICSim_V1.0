from .WAGEQuantizer import WAGEQuantizer

class WAGEV2Quantizer(WAGEQuantizer):

    def weight_init(self, weight, bits_W=None,factor=2.0, mode="fan_in"):
        scale = 1.0
        return scale                    