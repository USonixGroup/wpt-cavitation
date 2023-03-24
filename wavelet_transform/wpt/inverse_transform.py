import pywt
def idwt(coeffs,wavelet):
    return pywt.waverec(coeffs,wavelet)

def idwpt(coeffs, levels, wavelet):
    input = coeffs
    for _ in range(levels):
        output = []
        for i in range(int(len(input)/2)):
            approx = input[2*i]
            detail = input[2*i+1]
            if (approx is None) and (detail is None): # if neither node is used, move on
                output.append(None)
                continue
            upcoef = pywt.idwt(approx,detail,wavelet,mode='symmetric')
            output.append(upcoef)
        input = output.copy()
    return output[0]
