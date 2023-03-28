from numpy import hanning

def sci_form(number,dp = 3):
    'returns a string of the scientific form of a number'
    e_power = 0
    if number == 0:
        return f'{number}'
    while number >= 10:
        number /= 10
        e_power += 1
    while number < 1:
        number *= 10
        e_power -= 1
    return f'{round(number,dp)}e{e_power}'