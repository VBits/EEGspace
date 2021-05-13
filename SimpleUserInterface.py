import PySimpleGUI as sg

import Config


def create_user_interface(input_queue, output_queue, channels):
    layout = []
    for channel_number in channels:
        text_element = sg.Text("Reading buffer for mouse " + str(channel_number),
                               text_color='black',
                               background_color='white',
                               key=str(channel_number),
                               border_width=5,
                               pad=(100, 5),
                               justification="center")
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
            channel_number = next_status.mouse_number
            class_name = next_status.standardized_class_name
            color = "white" if class_name is None or class_name not in Config.state_colors else Config.state_colors[class_name]
            details = "Mouse " + str(channel_number) + " class: " + class_name
            text_element = window.FindElement(str(channel_number))
            text_element.Update(background_color=color)
            text_element.Update(details)

    output_queue.put("Quit")
    window.close()



