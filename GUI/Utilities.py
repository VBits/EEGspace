#prevent circular references
def set_input_default(input, setting):
    text = input.text()
    if text is None or text == "":
        input.setText(str(setting))