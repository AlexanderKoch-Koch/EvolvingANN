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
    

def binary_list_to_float(binary_list, min_float_value, max_float_value):
    """
    converts a list of binary digits (0 and 1) to a float value in the range(min_float_value, max_float_value)
    :param binary_list: list of binary digits (0 and 1)
    :param min_float_value: smallest float value this function will return
    :param max_float_value: largest float value this function will return
    """
    sum_value = float(sum(binary_list))
    length = len(binary_list)
    difference = float(max_float_value - min_float_value)
    scaling_factor = difference/length
    return sum_value * scaling_factor + min_float_value