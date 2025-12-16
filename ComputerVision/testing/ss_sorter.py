def sort_ss(strings):
    """
    Ordena una lista de pares de tiros siteswap en orden descendente.
    Args:
        strings (list of str): List of ss strings to sort.

    Returns:
        list of str: Sorted list of ss strings.
    """
    return sorted(strings, reverse=True)

"""podria ser util sacar el primer elemento de esta lista para empezar a ordenar el siteswap y matchearlo con la secuencia detectada trimeada
si es que no hay errores entre medio (porque se podria  sacar un maximo de ahi que este malo y haga que no se matchee la secuencia)"""
print("1:",sort_ss(["(6x,2x)", "(2x,2x)", "(6x,2x)", "(2x,2x)"]))
print("2:",sort_ss(["(2x,2x)", "(6x,2x)", "(2x,2x)", "(6x,2x)"]))

#def funcion que usa sort_ss para seleccionar donde empieza un string y generar el siteswap y acortarlo al periodo
#secuencia mas larga que calce toda 

#funcion que use el sistema para outputear la secuencia segun los encontrado de las duraciones relativas, 
# definir bien como obtengo las proporciones diferenciando un 6,6 2,2 de un 4,4

#le paso un lower case en el contrato
def siteswap_char_to_int(ch):
    if ch.isdigit():
        return int(ch)
    elif ch.isalpha():
        return 10 + ord(ch) - ord('a')
    else:
        raise ValueError(f"Invalid character: {ch}")


def verificar_ss(ss,b):
    """
    Verificar que la secuencia de tiros cumpla la propiedad de promedio entero comprobando si coincide con el número de pelotas esperado.
    No por tener promedio de números entero quiere decir que sea siteswap pero todos los siteswap tienen promedio entero.
    Args:
        ss (str): El siteswap a verificar.
        b (int): Número de pelotas del siteswap

    Returns:
        bool: True si el par es válido, False en caso contrario.
    Ejemplo:
        verificar_ss("(6x,2x)",3) -> True
    """
    import re
    trim = ss.strip("()")
    print("strip:", trim)
    limpio = trim.replace("x", "")
    print("replace:", limpio)
    resultado = re.split(r'[,)(]+', limpio)
    print("split:", resultado)

    values = [siteswap_char_to_int(ch) for ch in resultado]
    print("values:", values)
    avg=sum(values)/len(values)
    print("avg:", avg)
    balls=b
    print("#balls expected:", balls)

    return balls==avg
print(verificar_ss("(6x,2x)(2x,2x)",3))