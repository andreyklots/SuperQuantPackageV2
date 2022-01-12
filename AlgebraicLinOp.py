import numpy as np
import scipy as sp
from scipy.sparse import linalg as spLA
from scipy import sparse as spsp
import time as tm
"""
class LinearOperator:
    Property:
        MatArray: Describes the structure of a linear operator.
                  It is a list of Terms. Each Term is also a
                  list of Factors. With general format
                  MatArray = [ [Term1_Factor1, Term1_Factor2, ...],
                               [Term2_Factor1, Term2_Factor2, ...],
                               ...
                             ]
                  Thus, a n expression A,B,C,D,E,F is described 
                  by a list [ [A,B], [C,D,E], [F]  ].

    Creation of an object:
        LinearOperator(Description)
            Description can be either
                (1) list: then combination AxB+CxDxE+F of matrices
                          A,B,C,D,E,F is described by a list
                                    [ [A,B], [C,D,E], [F]  ]
                (2) LinearOperator: returns same LinearOperator
                                    not creating a copy -- only pointer
                (3) number, array or matrix: creates a Linear Operator
                                    with MatArray = 
                                          [[numpy.matrix(Description)]]

    Method:
        act(Vector): act on a N-dimensional column Vector or ona NxM
                     -matrix with a LinearOperator. Vector must be a
                     matrix. Function called as
                     LinearOperator.act(Vector).
                     Dimensionality N of a column-vector must be equal
                     to product of dimensionalities of all matrices in
                     each Term in LinearOperator.MatArray. Number M of
                     column-vectors can be any.
                     The way it acts on a vector is as follows:
                     Consider how it would act on a vector that is
                     a direct product of vectors:

                      |a|   |v|   |av|
                      | | x | | = |aw|
                      |b|   |w|   |bv|
                                  |bw|

                     Then linear operator will act on it as

                                  |1  0|  |0 1|          |av|     | aw|
                LinearOperator([[ |0 -1|, |1 0| ]]).act( |aw| ) = | av|
                                                         |bv|     |-bw|
                                                         |bw|     |-bv|
                     This result is equivalent to
                     
                     | aw|   | a|   |w|     |1  0| |a|       |0 1| |v| 
                     | av| = |  | x | | = ( |    |.| | ) x ( |   |.| | )
                     |-bw|   |-b|   |v|     |0 -1| |b|       |1 0| |w| 
                     |-bv|


            LinearOperator.H: returns hermitian-conjugated

    Operators:
        Following operations can be performed on a LinearOperator:
                A + B, A - B, -A and a*A.
                Here A, B are LinearOperator objects and "a" is a 
                scalar. When we add or subtract linear operators,
                their MatArray properties are simply concatenated.
                When we multiply a LinearOperator by a scalar "a"
                from the left, then the first Factor matrix in
                each Term is multiplied by a factor "a".

                One can also multiply two LinearOperator objects
                L1 and L2 as L1*L2 if they have same structure 
                (Same number of Factors in each Term)

    Representation:
        The LinearOperator object is represented as a sum of
        products of [M x N] notations where M and N are
        dimensions of each matrix.


kron(A,B):  makes a Kronecker product of LinearOperator
            objects A and B. This function goes through
            Terms in MatArrays of A and B and concatenates
            Factors in each combination of Terms.
            Example:
            consider direct product of operator
            
            L  = A x B + C   --> [ [A,B], [C] ] 
             1

            and operator

            L  = D + E x F   --> [ [D], [E,F] ] 
             2
            
            As a result of a Kroneker product we will get
            
            L x L   = (A x B x D) + (A x B x E x F) 
             1   2  
                    + (C x D) + (C x E x F)      -->

                    --> [ [A,B,D], [A,B,E,F], [C,D], [C,E,F] ]



eigh(A, u_initial): A -- LinearOperator,
                    u_initial - integer or N x k
                                -matrix
                    finds lowest k eigenvalues of eigenvectors
                    of LinearOperator A.
                    u_initial describes column-vectors that are
                    first approximations of eigenvectors of A.
                      If u_initial is a matrix or array then it
                    must have dimensions of N x k. Here N is
                    the dimensionality of a vector space on
                    which A is designed to act and k is the
                    number of eigenvectors/eigenvalues we want
                    to find. Thus, each column of u_initial
                    is a first aproximation of an eigenvector.
                      If u_initial is an integer of value k,
                    then initial approcimations of eigenvectors
                    are set automatically as a random N x k
                    -array.
"""



class LinearOperator():

    MatArray = []

    def __init__(self, MatArray):
        # MatArray defines terms and direct products
        # of matrices: AB+CDE+F --> [ [A,B], [C,D,E], [F] ]
        if type(MatArray) == list:
            self.MatArray = MatArray
        else:
            if type(MatArray) == LinearOperator:
                self.MatArray = MatArray.MatArray
            else:
                self.MatArray = [[np.matrix(MatArray)]]

    @property
    def H(self): # hermitian conjugate
        NewMatArray = []
        for Term in self.MatArray:
            NewFactor = []
            for Factor in Term:
                NewFactor += [ Factor.T ]
            NewMatArray += [ NewFactor ]
        return LinearOperator( NewMatArray )

    def __ActOnReshapedVector(self, Mat, ReshapedVector, AxesN):
        

        # Calculate
        #
        # Mat     vec
        #    j j     j j ...j ...
        #     0 1     2 3    1
        #                    ^
        #                    |
        #                   AxesN-th
        #
        VecShape = np.shape( ReshapedVector )
        MatIndexes = [0,1]
        ReshapedVectorIndexes = np.arange( 2, len(VecShape)+2 )
        ResultReshapedVectorIndexes = np.arange( 2, len(VecShape)+2 )
        # modify index arrays
        ReshapedVectorIndexes[AxesN] = 1
        ResultReshapedVectorIndexes[AxesN] = 0
        return np.einsum( Mat, MatIndexes,
                          ReshapedVector, 
                               ReshapedVectorIndexes.tolist()+[...],
                          ResultReshapedVectorIndexes.tolist()+[...]
                        ) 


    def __dimensionsOfTerm(self, Term):
            Dims = []
            for Factor in Term:
                Dims += [ np.shape(Factor)[0] ]
            return Dims
 

    def act(self, Vector):
        # here VecDim means size of a colunn-vector 
        # and NVecs os the number of the column-vectors
        
        print(tm.time())

        VecShape = np.shape(Vector)

        # make sure that Vector has dimension of (N,M), not (N,)
        if len(VecShape)==1:
            VecShape += (1,)
            NewVector = np.array( np.matrix(Vector).T )
        else:
            NewVector = np.array( Vector )

        VecDim, NVecs = VecShape

        # This woll be resulting vector
        ResVector = 0.j*NewVector[:,0:NVecs]
        
        
        for Term in self.MatArray:

            ####T0 = tm.time()

            # Reshape Vector for each term if terms have different
            # dimensions of constituting Factor matrices
            Dims = self.__dimensionsOfTerm(Term) + [ NVecs  ]
            ReshapedVector = np.reshape(NewVector, Dims)

            AxesN = 0 # Number of axes which we multiply

            # calculate direct product of matrices in Term acting on
            # reshaped vector

            ####T = tm.time()
            ####print("Step1: "+str(T-T0))
            for Factor in Term:
                # multiply vector of dimension m*n by matrix M:
                # M[m x m] . vec[m*n x k]
                ReshapedVector = self.__ActOnReshapedVector(
                                            Factor, ReshapedVector, AxesN
                                                           )
                AxesN += 1 # Number of axes which we multiply

            ResVector += np.reshape(ReshapedVector, [VecDim,NVecs])

            ####T = tm.time()
            ####print("Step2: "+str(T-T0))
 

        return np.matrix( ResVector )

    def __mul__(self,other):
        NewMatArray = []
        for Term_other in other.MatArray:
            for Term_self in self.MatArray:
                NewFactor = []
                for N in range( len(Term_self) ):
                    NewFactor += [ Term_self[N]*Term_other[N] ]
                NewMatArray += [ NewFactor ]
        return LinearOperator(NewMatArray)

    def __rmul__(self, other):
        if other == 0.:  # if multiply by 0, return empty MatArray
            return LinearOperator([])
        NewMatArray = []
        for Term in self.MatArray:
            NewTerm = Term.copy()
            NewTerm[0] = other * NewTerm[0]
            NewMatArray += [ NewTerm ]
        return LinearOperator(NewMatArray)

    def __add__(self,other):
        return LinearOperator( self.MatArray + other.MatArray  )

    def __sub__(self,other):
        return LinearOperator( self.MatArray + ((-1)*other).MatArray) 

    def __neg__(self):
        return (-1)*self

    def __pos__(self):
        return self

    def __repr__(self):
        ResStr = ""
        for Term in self.MatArray:
            ResStr += "+ "
            for Factor in Term:
                FactorShape = np.shape(Factor)
                ResStr += "["+str(FactorShape[0])+"x"+str(FactorShape[1])+"] "
            ResStr += "\n"
        return ResStr




def kron(A,B):
    
    NewMatArray = []
    
    # if one operand is a number then just multiply
    try:
        if type(A+0.j)==type(1.+1.j):
            return LinearOperator( A*B )
    except: pass
    try:
        if type(B+0.j)==type(1.+1.j):
            return LinearOperator( B*A )
    except:  pass

    # also multiply by scalar if one operand
    # is a 1x1 matrix or array
    try:
        if np.shape(A)==(1,1):
            return LinearOperator( A[0,0]*B )
    except: pass
    try:
        if np.shape(B)==(1,1):
            return LinearOperator( B[0,0]*A )
    except:  pass

    # if not multiplying by a scalar then
    # make sure we are dealing with LinearOperators
    A = LinearOperator(A)
    B = LinearOperator(B)

    for Aterm in A.MatArray:
        for Bterm in B.MatArray:
            NewMatArray += [ Aterm+Bterm ]
    return LinearOperator(NewMatArray)




def eigh(A, u_initial):
    if np.shape(u_initial) == (): 
    # if u_initial is a scalar, then use scipy.sparse.linalg.eigsh
        # Find dimensionality of the Linear Operator.
        VecDim = 1
        for Factor in A.MatArray[0]:
            VecDim *= np.shape(Factor)[0]
        # turn LinearOperator into Scipy Linear Operator
        spLA_A = spLA.LinearOperator( (VecDim,VecDim), matvec=A.act  )
        u0 = np.array(
                       np.random.rand(VecDim, u_initial)
                     )
        E, u = spLA.eigsh( spLA_A , k=u_initial, which='SA')
    else:
    # if u_initial is an array/matrix then it is our 0th approx.
    # and use LOBPCG method
        u0 = np.array( u_initial )
        E, u = spLA.lobpcg( spLA_A , u0, largest=False )
    return E, u
