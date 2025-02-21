import numpy as np
from numpy import outer, trace, dot, vdot, pi, log2, exp, sin, cos, sqrt, sign, diag, linspace, arange, array, inf, zeros, eye, arccos, arcsin, arctan, mean, std, concatenate, kron, sign, ceil, log, unique
from numpy.random import uniform, normal, randint, choice
from scipy.linalg import eig, eigh, norm, expm, sqrtm
from numpy.linalg import svd, norm, inv, pinv
from scipy.stats import sem
from scipy.optimize import minimize
from functools import reduce, partial
from itertools import product
import qutip as qp
from multiprocessing import Pool
from time import time

P0 = np.array([[1., 0.],
               [0., 0.]]) # |0><0|
P1 = np.array([[0., 0.],
               [0., 1.]]) # |1><1|
X = np.array([[0.,1.],
              [1.,0.]]) # X Pauli matrix
Y = np.array([[0.,-1.j],
              [1.j, 0.]]) # Y Pauli matrix
Z = np.array([[1., 0.],
              [0.,-1.]]) # Z Pauli matrix
I = np.array([[1.,0.],
              [0.,1.]]) # 2x2 identity matrix

# some functions # 

def kron_A_N(A, N): # fast kron(A, eye(N))
    m,n = A.shape
    out = zeros((m, N, n, N), dtype=A.dtype)
    r = arange(N)
    out[:, r, :, r] = A
    out.shape = (m*N, n*N)
    return out
    
def kron_N_A(A, N): # fast kron(eye(N), A)
    m,n = A.shape
    out = zeros((N, m, N, n), dtype=A.dtype)
    r = np.arange(N)
    out[r, :, r, :] = A
    out.shape = (m*N, n*N)
    return out

def kron_A_I_diag(A, N): # same, but "diagonal"
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.T.reshape(m*N)
    return out

def kron_I_A_diag(A, N):  # same, but "diagonal"
    m = len(A)
    out = zeros((N, m), dtype=A.dtype)
    out[arange(N)] = A
    out = out.reshape(m*N)
    return out

def trace_distance(A, B):
    sub = A - B
    return trace(sqrtm(dot(sub.conj().T, sub))).real/2

def fidelity(A, B):
    res = reduce(dot, [sqrtm(A), B, sqrtm(A)])
    res = sqrtm(res)
    return trace(res).real**2

def sup_fidelity(A, B):
    """ An upper bound for the usual fidelity. """
    t1 = trace(A@B).real
    t2 = max(0, 1 - trace(A@A).real)
    t3 = max(0, 1 - trace(B@B).real)
    return t1 + sqrt(t2)*sqrt(t3)

def partial_trace(dm, m=None, n=None, subsystem=0):
    """ Simple and fast, but cuts only in halves. """
    if (m is None) or (n is None): # cut in equal halves
        N = log2(len(dm))
        m = int(N / 2)
        n = int(N - m)
        m = 2**m
        n = 2**n
    if subsystem == 0:
        return trace(dm.reshape((m, n, m, n)), axis1=0, axis2=2)
    elif subsystem == 1:
        return trace(dm.reshape((m, n, m, n)), axis1=1, axis2=3)

def concurrence_pure(dm):
    dm_red = partial_trace(dm)
    return sqrt(2*(1 - trace(dm_red@dm_red).real))

def concurrence(dm):
    YY = kron(Y, Y)
    dm_t = YY@dm.conj()@YY
    R = dm_t@dm
    lambdas = [l if l > 0 else 0 for l in np.sort(eig(R)[0].real)]
    c = sqrt(lambdas[3]) - sqrt(lambdas[2]) - sqrt(lambdas[1]) - sqrt(lambdas[0])
    return max(0, c)

def two_subsys_negativity(dm):
    def partial_transpose(A, n, m):
        A_c = array(A)
        Bt = A[:n, m:].copy()
        Ct = A[n:, :m].copy()
        A_c[:n, m:] = Ct
        A_c[n:, :m] = Bt
        return A_c
    dm_ptrans = partial_transpose(dm, int(len(dm)/2), int(len(dm)/2))
    lambda_min = eigh(dm_ptrans)[0][0]
    return 2*abs(min(0, lambda_min))

def prev_to_next_ansatz(pars, n_tot_p, n_meas_p, n_layers_p, n_tot_n, n_meas_n, n_layers_n, subsval=0):
    """Extends the outcome values and the ansatz hea_cry_rzrx to new n_tot and n_layers, filling the angles with zeros."""
    x0 = []
    it = iter(pars)
    for q in range(n_tot_p):
        x0.append(next(it))
        x0.append(next(it))
    for q in range(n_tot_n - n_tot_p):
        x0.append(subsval)
        x0.append(subsval)   
    for l in range(n_layers_p):
        for q in range(n_tot_p - 1):
            x0.append(next(it))
        for q in range(n_tot_p - 1, n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_p):
            x0.append(next(it))
            x0.append(next(it))
        for q in range(n_tot_p, n_tot_n):
            x0.append(subsval)
            x0.append(subsval)
    for l in range(n_layers_n - n_layers_p):
        for q in range(n_tot_n - 1):
            x0.append(subsval)
        for q in range(n_tot_n):
            x0.append(subsval)
            x0.append(subsval) 
    x0 = x0 + list(kron(diag([next(it) for i in range(2**n_meas_p)]), eye(2**(n_meas_n - n_meas_p))).diagonal())
    return x0


def gen_even_ent_data(n, n_inp=2, mixed=True, marks="neg", n_chunks=100, eps=0):
    """ Generates a data set of states with evenly distributed entanglements. """
    
    d = 2**n_inp
    
    if marks == "neg":
        ent_measure_func = two_subsys_negativity
    elif marks == "con":
        ent_measure_func = concurrence 
    
    ent_count_max = int(ceil(n/n_chunks))
    ent_line = linspace(0, 1, n_chunks + 1)[1:]
    ent_counts = [0]*n_chunks
    count = 0
    labels = zeros(n, dtype=float)
    if mixed == True:
        states = zeros([n, d, d], dtype=complex)
    else:
        states = zeros([n, d], dtype=complex)
    while count < n:
        print("%d" %count, end="\r")
        if mixed == True:
            state = (qp.rand_dm(d, distribution='ginibre', rank=randint(1, d + 1))).full() # lame, but works faster for mixed states
            ent = ent_measure_func(state)
        else:
            state = (qp.rand_ket(d)).full().reshape(-1)
            ent = ent_measure_func(outer(state, state.conj().T))
        if ent >= eps:
            ent_diffs = ent_line - ent
            ind = np.abs(ent_diffs).argmin()
            if sign(ent_diffs[ind]) == -1:
                ind += 1
            if ent_counts[ind] < ent_count_max:
                ent_counts[ind] += 1        
                states[count] = state
                labels[count] = ent
                count += 1
            
    return array(states), array(labels)


# quantum state generators # 

def rand_sv(n_qubits):
    """ Generates a random pure state as a vector. """
    d = 2**n_qubits
    sv = uniform(-1, 1, d) + 1j*uniform(-1, 1, d)
    return sv/norm(sv)

def rand_dm(n_qubits):
    """ Generates a random mixed state as a full-rank density matrix. """
    d = 2**n_qubits
    H = uniform(-1, 1, [d, d]) + 1j*uniform(-1, 1, [d, d])
    dm = H@H.conj().T
    dm = dm/trace(dm).real
    return dm


# Fisher informations #

def cfi(channel_func, dm, p, channel_args, povm, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes classical Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 0
    for op in povm:
        prob = trace(dot(dm_n, op)).real
        if prob > 0:
            prob_p = trace(dot(dm_n_p, op)).real
            prob_m = trace(dot(dm_n_m, op)).real
            der = (prob_p - prob_m)/(2*dp)
            fi += der**2/prob
    return fi

def qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n, dm_n_p))) / dp**2
    return fi

def qfi_central(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-2):
    """ Computes quantum Fisher information via "central differences". Only for channels! """
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_m = reduce(kron, [channel_func(dm, p-dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(fidelity(dm_n_m, dm_n_p))) / dp**2/4
    return fi

def sup_qfi(channel_func, dm, p, channel_args, n_copies=1, n_ext=0, dp=1e-5):
    """ Computes an upper bound (?) for quantum Fisher information. Only for channels! """
    dm_n = reduce(kron, [channel_func(dm, p, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    dm_n_p = reduce(kron, [channel_func(dm, p+dp, *channel_args)]*n_copies + [diag([1] + [0]*(2**(n_ext) - 1))])
    fi = 8*(1 - sqrt(sup_fidelity(dm_n, dm_n_p))) / dp**2
    return fi


def sld(channel_func, channel_par, channel_args, dm_ini, n_copies=1, n_ext=0, dp=1e-5, return_fi=False):
    """
        Numerically finds the SLD operator L.
        Optionally returns the classical and quantum Fisher informations.
        Only for channels!
    """
    
    n_inp = int(log2(len(dm_ini)))*n_copies
    n_tot = n_inp + n_ext
    d = 2**n_tot
        
    dm_ext = diag([1] + [0]*(2**(n_ext) - 1))
    dm_n = reduce(kron, [channel_func(dm_ini, channel_par, *channel_args)]*n_copies + [dm_ext])
    dm_n_p = reduce(kron, [channel_func(dm_ini, channel_par+dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_m = reduce(kron, [channel_func(dm_ini, channel_par-dp, *channel_args)]*n_copies + [dm_ext])
    dm_n_der = (dm_n_p - dm_n_m)/(2*dp)
    
    evals, evecs = eigh(dm_n)
    evecs = evecs.T
    
    L = zeros([d, d], dtype=complex)
    for i in range(d):
        for j in range(d):
            denom = evals[i] + evals[j]
            if denom > 1e-5:
                numer = evecs[i].conj().T@dm_n_der@evecs[j]
                oper = outer(evecs[i], evecs[j].conj().T)
                L += 2*numer/denom*oper
    # print("Incompliance with the definition of SLD:", norm( (L@dm_n + dm_n@L)/2 - dm_n_der ))
    
    if return_fi == True:
        evecs_L = eigh(L)[1].T
        projs_L = [outer(vec, vec.conj().T) for vec in evecs_L]
        CFI = cfi(channel_func, dm_ini, channel_par, channel_args, projs_L, n_copies=n_copies, n_ext=n_ext, dp=dp)
        QFI = trace(L@L@dm_n).real
        return L, CFI, QFI
    else:
        return L
        

### channels ###

def hw_channel(dm, p):
    """ Holevo-Werner channel """
    d = len(dm)
    return ((d - p)*eye(d) + (d*p - 1)*dm.T)/(d**2 - 1)
    
def depolarizing_channel(dm, p):
    d = len(dm)
    return (1 - p)*dm + p/d*eye(d)

def generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0, sqrt(1 - g)]])*sqrt(1 - N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt(g*(1 - N))],
                [0,               0]])
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(1 - g), 0],
                [          0, 1]])*sqrt(N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,         0],
                [sqrt(g*N), 0]])
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4


def another_generalized_amplitude_damping_channel(dm, g, N, target_qubit):
    """ Adapted from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.012317 """
    
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    
    K1 = array([[1,           0],
                [0,     sqrt(g)]])*sqrt(N)
    K1 = reduce(kron, [eye(dl), K1, eye(dr)]) # inefficient
    
    K2 = array([[0, sqrt((1 - g))],
                [0,             0]])*sqrt(N)
    K2 = reduce(kron, [eye(dl), K2, eye(dr)])
    
    K3 = array([[sqrt(g), 0],
                [      0, 1]])*sqrt(1 - N)
    K3 = reduce(kron, [eye(dl), K3, eye(dr)])
            
    K4 = array([[0,           0],
                [sqrt(1 - g), 0]])*sqrt(1 - N)
    K4 = reduce(kron, [eye(dl), K4, eye(dr)])
    
    dm1 = reduce(dot, [K1, dm, K1.conj().T])
    dm2 = reduce(dot, [K2, dm, K2.conj().T])
    dm3 = reduce(dot, [K3, dm, K3.conj().T])
    dm4 = reduce(dot, [K4, dm, K4.conj().T])
        
    return dm1 + dm2 + dm3 + dm4

    
def X_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [X]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Y_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Y]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def Z_rotations(dm, p):
    n_qubits = int(log2(len(dm)))
    op = reduce(kron, [Z]*n_qubits)
    U = expm(-1j*p*op)
    return reduce(dot, [U, dm, U.conj().T])

def z_rot(dm, p, target_qubit):
    """
        z-rotation of the specified qubit.
        Mind the division by two!
    """
    n_qubits = int(log2(len(dm)))
    dl = 2**target_qubit
    dr = 2**(n_qubits - target_qubit - 1)
    U = reduce(kron, [eye(dl), expm(-1j*p/2*Z), eye(dr)]) # inefficient
    return U@dm@U.conj().T


def random_channel(dm, p, pars, p_index, pauli_basis=None):
    """
        Random single-parametrized channel.
        Attaches to a given n-qubit state dm a 2n-qubit pure state,
        applies to the joint state a unitary with (4^(3n) - 1) parameters pars, one of which is the parameter p in question with the index p_index.        
    """
    d = len(dm)
    n_inp = int(log2(d))
    n_ext = 2*n_inp
    n_tot = n_inp + n_ext
    pars_conc = concatenate([pars[:p_index], [p], + pars[p_index:]])
    V = su2n(n_tot, pars_conc, pauli_basis=pauli_basis)
    dm_ext = diag([1] + [0]*(2**n_ext - 1))
    dm_n = V@kron(dm, dm_ext)@V.conj().T
    dm_n = trace(dm_n.reshape(2**n_inp, 2**n_ext, 2**n_inp, 2**n_ext), axis1=1, axis2=3) # partial trace with respect to the extension
    return dm_n


### Hamiltonians ###

def ising_ham(n_qubits, h, J=1, bc="closed"):
    d = 2**n_qubits
    Hx = zeros((d, d), dtype=complex)
    for q in range(n_qubits):
        X_op = [I]*q + [X] + [I]*(n_qubits-q-1)
        Hx = Hx + reduce(kron, X_op)
    Hzz = zeros((d, d), dtype=complex)
    for q in range(n_qubits-1):
        Hzz = Hzz + reduce(kron, [I]*q + [Z, Z] + [I]*(n_qubits-q-2))
    if bc == "closed" and n_qubits > 2:
        Hzz = Hzz + reduce(kron, [Z] + [I]*(n_qubits-2) + [Z])
    if n_qubits == 1: # lame
        Hzz = 1*Z
    return -J*(Hzz + h*Hx)


def schwinger_ham(n_qubits, m, w=1, g=1):
    
    d = 2**n_qubits
    sp = (X + 1j*Y)/2
    sm = (X - 1j*Y)/2
    
    term_1 = 1j * zeros((d, d))
    for j in range(n_qubits):
        k = (j + 1) % n_qubits
        crea = [I]*j + [sp] + [I]*(n_qubits - j - 1)
        anni = [I]*k + [sm] + [I]*(n_qubits - k - 1)
        crea = reduce(kron, crea)
        anni = reduce(kron, anni)
        op = crea@anni
        term_1 = term_1 + op + op.conj().T
    term_1 = w * term_1

    term_2 = 1j * zeros((d, d))
    for j in range(n_qubits):
        operator = [I]*j + [Z] + [I]*(n_qubits - 1 - j)
        term_2 = term_2 + (-1)**(j + 1) * reduce(kron, operator)
    term_2 = m / 2 * term_2

    term_3 = 1j * zeros((d, d))
    for j in range(n_qubits):
        L = 1j * zeros((d, d))
        for l in range(j + 1):
            operator = [I]*n_qubits
            operator[l] = Z + (-1)**(l + 1) * I
            L = L - 0.5 * reduce(kron, operator)
        term_3 = term_3 + L@L
    term_3 = g * term_3
        
    return term_1 + term_2 + term_3
    

# measure #    

def measure_z_counts(dm, n_shots):
    probs = dm.diagonal().real
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts

def measure_povm_counts(dm, n_shots, povm):
    probs = [trace(dm@el).real for el in povm]
    d = len(probs)
    measurements = choice(arange(d), size=n_shots, p=probs)
    measurements = unique(measurements, return_counts=True)
    counts = zeros(d)
    counts[measurements[0]] = measurements[1]
    return counts
