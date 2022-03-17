"""
Online analysis - Stimulus delivery
"""

import serial
import time
import datetime
from OnlineAnalysis import Config


def get_white_noise_function():

    audio = serial.Serial(Config.comport)
    is_open = audio.is_open
    if is_open:
        audio.close()
        audio.open()

    def white_noise(duration=4):
        nonlocal audio
        on = (b"1\n")
        off = (b"0\n")
        print('Starting the audio on: {}'.format(datetime.datetime.now()))
        audio.write(on)
        time.sleep(duration)
        audio.write(off)
        print('Stopping the audio on: {}'.format(datetime.datetime.now()))

    return white_noise
