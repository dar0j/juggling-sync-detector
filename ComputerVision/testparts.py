def minimal_period(pairs):
    seq=['(' + p[0]+','+p[1]+')' for p in pairs]
    n=len(seq)
    for k in range(1,n+1):
        if all(seq[i]==seq[i%k] for i in range(n)):
            return seq[:k]
    return seq

#una vez encontrado verificar orden alfabetico segun las rotaciones de las secuencias
#devolver la rotacion alfabeticamente menor
#ejemplo: de (6x,6x)(2x,6)(6x,6x)(6,2x) devolver (6x,6x)(6,2x)(6x,6x)(2x,6)
def rotate_sequence(seq, k):
    return seq[k:] + seq[:k]

#devolver la rotacion alfabeticamente mayor
def maximal_rotation(seq):
    n = len(seq)
    rotations = [rotate_sequence(seq, k) for k in range(n)]
    return max(rotations)

if __name__ == "__main__":
    # Example usage
    test_pairs = [('6','6'), ('6','6'), ('0','6x'), ('6','6'), ('6','6'), ('6x','0'), ('6','6'), ('6','6')]
    #ejemplo que no termina cuando termina el ciclo/periodo, tiene periodo de mas de 1 par y se repite más de una vez
    #test_pairs = [('6x','6x'), ('2','6x'), ('6x','6x'), ('6x','2'), ('6x','6x'), ('2','6x'), ('6x','6x'), ('6x','2'), ('6x','6x')]
    m = minimal_period(test_pairs)
    p = 2*len(m) # la secuencia detectada puede estar en cualquier rotacion asi que tengo que probar todas no es como que solo pueda rotar en n/2
    print("period:", p)
    print(''.join(maximal_rotation(m)))


#script para procesar de las carpetas de holdput si coincide o no la secuencia con la detectada
#el nombre esta en la 2da posicion despues de split por guion bajo

