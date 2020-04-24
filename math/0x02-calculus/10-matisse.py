def poly_derivative(poly):
    if poly == []:
        return None
    result = []
    for i in range(1, len(poly)):
        result.append(i * poly[i])
    if result == [0] * len(result):
        return [0]
    return result
