from math import log10, exp

def exponencial(mean_rate: float, rand):
    """ Função que retorna um valor aleatório de uma distribuição exponencial 
    
        Parâmetros:
            -mean_rate (float): Taxa média da distribuição exponencial.
            -rand (Random): Objeto Random do Python.

        Retorno:
            -float: Valor aleatório da distribuição exponencial.
    """

    random = rand.random()
    return (-1 / mean_rate) * (log10(random)/log10(exp(1)))