import numpy as np
import AlgebraicLinOp as ALO

# Define Pauli matrices
I = np.matrix([[1,0],
               [0,1]])
X = np.matrix([[0,1],
               [1,0]])
Y = np.matrix([[0 ,-1j],
               [1j,  0]])
Z = np.matrix([[1, 0],
               [0,-1]])


###########
#
# Compare small operator diagonalization
# between AlgebraicLinOp and numpy
#
###########

# Define MatArray for the following 3-spin hamiltonian
#
#       <3>   <3>   <3>
# for  X   + Y   + Z   + (ZxIxI) + 2(IxZxI) + 3(IxIxZ)
MatArray = [ 3*[X], 3*[Y], 3*[Z], [Z,I,I], [I,2*Z,I], [I,I,3*Z] ]

# Make a 3-spin LinearOperator Hamiltonian
H_LO = ALO.LinearOperator( MatArray )

# Make a 3-spin matrix Hamiltonian:
H_mat =   np.kron( np.kron(X, X ), X)\
      +   np.kron( np.kron(Y, Y ), Y)\
      +   np.kron( np.kron(Z, Z ), Z)\
      +   np.kron( np.kron(Z, I ), I)\
      + 2*np.kron( np.kron(I, Z ), I)\
      + 3*np.kron( np.kron(I, I ), Z)

# printed informal representation of the Hamiltonian
Hamilt3Text = """
   <3>   <3>   <3>
  X   + Y   + Z   + (ZxIxI) + 2(IxZxI) + 3(IxIxZ)
"""
print("Diagonalizing the 3-spin Hamiltonian")
print(Hamilt3Text)

# Diagonalize a 3-spin LinearOperator Hamiltonian
# and get 2**3=8 eigenvalues and eigenvectors
E_LO, u_LO = ALO.eigh( H_LO, 2**3 )

# Diagonalize a 3-spin matrix Hamiltonian
E_mat, u_mat = np.linalg.eigh(H_mat)

# Display com results next to each other:
print("Eigenvalues")
print("  AlgebraicLinOp LinearOperator diagonalization:")
print(E_LO)
print("  numpy matrix diagonalization:")
print(E_mat)

print("Eigenvectors (Real Part):")
print("  AlgebraicLinOp LinearOperator diagonalization:")
print(np.real(u_LO).round(2))
print("  numpy matrix diagonalization:")
print(np.real(u_mat).round(2))

print("Eigenvectors (Imaginary Part):")
print("  AlgebraicLinOp LinearOperator diagonalization:")
print(np.imag(u_LO).round(2))
print("  numpy matrix diagonalization:")
print(np.imag(u_mat).round(2))

Comment = """
Note that eigenvectors corresponding to non-degenerate
eigenvalues should be equal up to a phase factor.
Eigenvectors corresponding to degerate eigenvvalues 
may enter as various superpositions for the two 
diagonalization methods.


"""
print(Comment)





##############
#
# Now Switch to a Large hamiltonian
#
##############

# Number of spins (1 or 2 digit)
N_large = 17

# 2-character string representing the number of spins
NS = " "+str(N_large) if N_large<10 else str(N_large)
# Now let us find eigenvectors of a large Hamiltonian
# for N_large spins
H_large_text = "  <"+NS+">    <"+NS+">    <"+NS+"> \n"\
             + " X    "+ " + Y     +"+ " Z         \n"
print("Now let us diagonalize a large "+NS+"-spin Hamiltonian ")
print(H_large_text)
print( "The Hilbert space is 2^"+NS+" = "+str(2**N_large) )

# Define MatArray for this Hamiltonian
MatArray = [ N_large*[X], N_large*[Y], N_large*[Z] ]

H_large_LO = ALO.LinearOperator(MatArray)

# calculat 6 lowest eigenvalues
E_large, u_large = ALO.eigh( H_large_LO, 6 )

print("Eigenvalues")
print(E_large.round(8))
print("Eigenvectors")
print(u_large.round(1))




