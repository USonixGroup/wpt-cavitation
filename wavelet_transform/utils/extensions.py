def sci_form(number, dp=3):
    "returns a string of the scientific form of a number"
    e_power = 0
    if number == 0:
        return f"{number}"
    while number >= 10:
        number /= 10
        e_power += 1
    while number < 1:
        number *= 10
        e_power -= 1
    return f"{round(number,dp)}e{e_power}"


def to_list(x, *, convert=None):
    "handling keyword arguments"
    if x is None:
        return None
    x = [x] if not isinstance(x, list) else x
    if convert == "float":
        x = [float(i) if isinstance(i, int) else i for i in x]
    elif convert == "int":
        x = [int(i) if isinstance(i, float) else i for i in x]
    return x
