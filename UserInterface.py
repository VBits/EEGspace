# from PyQt5 import uic, QtWidgets
# from PyQt5.QtWidgets import *
import PySimpleGUI as sg  # Part 1 - The import

#todo move this to a shared location
state_colors = {
    "REM": "#443e99",
    "SWS": "#3e8399",
    "LMwake": "#3e9944",
    "HMwake": "#f2ef30",
}

def create_user_interface(input_queue, output_queue, channels):
    layout = []
    for channel_number in channels:
        text_element = sg.Text("Reading data for mouse " + str(channel_number),
                               text_color='black',
                               background_color='white',
                               key=str(channel_number),
                               border_width=5)
        layout.append([text_element])

    window = sg.Window('Current Mouse Statuses', layout)

    while True:
        event, values = window.read(timeout=10)
        if event == sg.WIN_CLOSED:
            break
        next_status = None
        if not input_queue.empty():
            next_status = input_queue.get()
        if next_status is not None:
            channel_number = next_status[0]
            class_name = next_status[3]
            color = "white" if class_name is None or class_name not in state_colors else state_colors[class_name]
            details = "Mouse " + str(channel_number) + " class: " + class_name
            text_element = window.FindElement(str(channel_number))
            text_element.Update(background_color=color)
            text_element.Update(details)

    output_queue.put("Quit")
    window.close()



