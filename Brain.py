import time


class Brain:

    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.output_callback_function = None

    def activate_input(self, index):
        time.sleep(1)
        self.output_callback_function(1)

    def register_output_callback(self, output_callback_function):
        self.output_callback_function = output_callback_function

    def test_output(self):
        self.output_callback_function()
