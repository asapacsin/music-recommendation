import numpy as np
import librosa
import torch
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype('float32')


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype('int16')

def main():
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel= 'HTSAT-base')
    print(model)

if __name__ == "__main__":
    main()