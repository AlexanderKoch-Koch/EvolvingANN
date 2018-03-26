def float_to_binary_list(float_value, precision, len_list):
    result_list = []
    for i in range(int(len_list / 2)):
        if float_value < -i * precision:
            result_list.append(1)
        else:
            result_list.append(0)

        if float_value > i * precision:
            result_list.append(1)
        else:
            result_list.append(0)
    return result_list


def preprocess_inputs(observation):
    inputs = []
    inputs += float_to_binary_list(observation[0], 0.05, 10)
    inputs += float_to_binary_list(observation[1], 0.05, 4)
    inputs += float_to_binary_list(observation[2], 0.05, 8)
    inputs += float_to_binary_list(observation[3], 0.05, 4)
    return inputs
