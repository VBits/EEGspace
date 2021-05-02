import serial
import time
import datetime


audio = serial.Serial("COM4")


def white_noise(duration=4):
    on = (b"1\n")
    off = (b"0\n")
    print('Starting the audio on: {}'.format(datetime.datetime.now()))
    audio.write(on)
    time.sleep(duration)
    audio.write(off)
    print('Stopping the audio on: {}'.format(datetime.datetime.now()))


