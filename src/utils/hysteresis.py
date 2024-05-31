def hysteresis_valve_pos(vcol, delta):
    """
    Simulate hysteresis in valve position
    """
    vhyst = vcol.copy(deep=True)
    if delta == 0:
        return vhyst

    for i in range(1, len(vcol)):
        if vcol[i] - vhyst[i-1] > delta:
            vhyst[i] = vcol[i] - delta
        elif vcol[i] - vhyst[i-1] < -delta:
            vhyst[i] = vcol[i] + delta
        else:
            vhyst[i] = vhyst[i-1]
            
    return vhyst

def append_hysteresis(data, delta_percent):
    """
    Append hysteresis-filtered valve positions to data
    """
    # append hysteresis with different d-values
    for i in range(4):
        data[f'vh{i}'] = hysteresis_valve_pos(data[f'v{i}'], delta_percent/100)

    return data