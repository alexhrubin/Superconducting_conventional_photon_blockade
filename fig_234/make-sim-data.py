import pandas as pd
import numpy as np
import qutip as qt
from datetime import datetime
import asyncio

N0 = False
N1 = False
N2 = False
N3 = True

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

if N0:
    lamb = 3.640
    fq = 5313.25
    fr = 5230.2

    N = 10
    Nq = 3
    a = qt.destroy(N)
    b = qt.destroy(Nq)
    kappa = 0.1
    gamma = kappa
    aq = 227
    g = 17
    r = 0
    def H_drive(lamb, dfreq, damp):
        H = 0
        H += (fr + lamb - dfreq)*qt.tensor([a.dag()*a, qt.qeye(Nq)])
        H += (fq - lamb - dfreq)*qt.tensor([qt.qeye(N), b.dag()*b])
        H += g*(qt.tensor([a.dag(), b]) + qt.tensor([a, b.dag()]))
        H += -(aq/2)*qt.tensor([qt.qeye(N), b.dag()*b.dag()*b*b])
        H += damp*qt.tensor([(a + a.dag()), qt.qeye(Nq)])
        H += r*damp*qt.tensor([qt.qeye(N), b + b.dag()])
        return H

    powers = np.logspace(-3, 0.45, 201, endpoint=True)
    cops = [np.sqrt(kappa)*qt.tensor([a, qt.qeye(Nq)]), np.sqrt(gamma)*qt.tensor([qt.qeye(N), b])]

    @background
    def find_ss(power):
        H = H_drive(lamb, fr, power)
        rho = qt.steadystate(H, cops)
        rho = rho.ptrace(0)
        x = np.abs(np.diag(rho.full()))
        return x
                                                    
    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers])              
    data = loop.run_until_complete(looper)                                  

    data = np.array(data)

    ncol = []
    pcol = []
    dcol = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ncol.append(j)
            pcol.append(powers[i])
            dcol.append(data[i,j])

    df_dict = {
        'n': ncol,
        'power': pcol,
        'data': dcol
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('N0-blockade-sim-test.csv')

if N1:
    lamb = 2.50
    fq = 5348.5
    fr = 5230.2

    N = 7
    Nq = 3
    Nb = 2
    a = qt.destroy(N)
    b = qt.destroy(Nq)
    c = qt.destroy(Nb)
    kappa = 0.1
    gamma = kappa
    gammab = gamma
    aq = 227
    g = 17
    gb = 14
    r = 0
    def H_drive(lamb, dfreq, damp):
        H = 0
        H += (fr + lamb - dfreq)*qt.tensor([a.dag()*a, qt.qeye(Nq), qt.qeye(Nb)])
        H += (fq - lamb - dfreq)*qt.tensor([qt.qeye(N), b.dag()*b, qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c.dag()*c])
        H += g*(qt.tensor([a.dag(), b, qt.qeye(Nb)]) + qt.tensor([a, b.dag(), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), c]) + qt.tensor([a, qt.qeye(Nq), c.dag()]))
        H += -(aq/2)*qt.tensor([qt.qeye(N), b.dag()*b.dag()*b*b, qt.qeye(Nb)])
        H += damp*qt.tensor([(a + a.dag()), qt.qeye(Nq), qt.qeye(Nb)])
        return H
    
    powers = np.logspace(-3, 0.8, 51, endpoint=True)
    cops = [np.sqrt(kappa)*qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb)]), np.sqrt(gamma)*qt.tensor([qt.qeye(N), b, qt.qeye(Nb)]), np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c])]

    H0 = H_drive(lamb, 0, 0)
    fd = H0.eigenstates()[0][1] - H0.eigenstates()[0][0]
    
    @background
    def find_ss(power):
        H = H_drive(lamb, fd, power)
        rho = qt.steadystate(H, cops)
        x = rho
        return x
    
    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers])              
    data = loop.run_until_complete(looper)  

    psim = (1/np.sqrt(2))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1)]) - qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0)]))
    psip = (1/np.sqrt(2))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1)]) + qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0)]))
    vacs = qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0)])

    vac = np.zeros(len(powers))
    sp = np.zeros(len(powers))
    sm = np.zeros(len(powers))
    for i, p in enumerate(powers):
        rho = data[i].ptrace([0,2])
        vac[i] = np.abs(rho.overlap(vacs))
        sp[i] = np.abs(rho.overlap(psip))
        sm[i] = np.abs(rho.overlap(psim))

    df_dict = {
        'power': powers,
        'vac': vac,
        'sp': sp,
        'sm': sm 
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('N1-blockade-sim-test.csv')

if N2: 
    start_time = datetime.now()
    lamb = 2.50
    fq = 5348.5
    fr = 5230.2

    N = 7
    Nq = 3
    Nb = 2
    a = qt.destroy(N)
    b = qt.destroy(Nq)
    c = qt.destroy(Nb)
    kappa = 0.1
    gamma = kappa
    gammab = gamma
    aq = 227
    g = 17
    gb = 14
    r = 0
    def H_drive(lamb, dfreq, damp):
        H = 0
        H += (fr + lamb - dfreq)*qt.tensor([a.dag()*a, qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb)])
        H += (fq - lamb - dfreq)*qt.tensor([qt.qeye(N), b.dag()*b, qt.qeye(Nb), qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c.dag()*c, qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), c.dag()*c])
        H += g*(qt.tensor([a.dag(), b, qt.qeye(Nb), qt.qeye(Nb)]) + qt.tensor([a, b.dag(), qt.qeye(Nb), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), c, qt.qeye(Nb)]) + qt.tensor([a, qt.qeye(Nq), c.dag(), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), qt.qeye(Nb), c]) + qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb), c.dag()]))
        H += -(aq/2)*qt.tensor([qt.qeye(N), b.dag()*b.dag()*b*b, qt.qeye(Nb), qt.qeye(Nb)])
        H += damp*qt.tensor([(a + a.dag()), qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb)])
        return H
    
    H0 = H_drive(lamb, 0, 0)
    fd = H0.eigenstates()[0][1] - H0.eigenstates()[0][0]
    powers = np.logspace(-3, 0.8, 51, endpoint=True)
    cops = [np.sqrt(kappa)*qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb)]), 
            np.sqrt(gamma)*qt.tensor([qt.qeye(N), b, qt.qeye(Nb), qt.qeye(Nb)]), 
            np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c, qt.qeye(Nb)]), 
            np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), c])]
    
    @background
    def find_ss(power):
        H = H_drive(lamb, fd, power)
        rho = qt.steadystate(H, cops)
        x = rho
        return x
    
    bsize = 26
    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[:bsize]])              
    data1 = loop.run_until_complete(looper)  

    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[bsize:]])              
    data2 = loop.run_until_complete(looper)  
    
    data = data1 + data2

    psim = (1/np.sqrt(2))*((1/np.sqrt(2))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 1)])) - qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0), qt.basis(Nb, 0)]))
    psip = (1/np.sqrt(2))*((1/np.sqrt(2))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 1)])) + qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0), qt.basis(Nb, 0)]))
    vacs = qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 0)])

    vac = np.zeros(len(powers))
    sp = np.zeros(len(powers))
    sm = np.zeros(len(powers))
    for i, p in enumerate(powers):
        rho = data[i].ptrace([0,2,3])
        vac[i] = np.abs(rho.overlap(vacs))
        sp[i] = np.abs(rho.overlap(psip))
        sm[i] = np.abs(rho.overlap(psim))

    df_dict = {
        'power': powers,
        'vac': vac,
        'sp': sp,
        'sm': sm 
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('N2-blockade-sim-test.csv')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if N3: 
    start_time = datetime.now()
    lamb = 2.50
    fq = 5348.5
    fr = 5230.2

    N = 7
    Nq = 3
    Nb = 2
    a = qt.destroy(N)
    b = qt.destroy(Nq)
    c = qt.destroy(Nb)
    kappa = 0.1
    gamma = kappa
    gammab = gamma
    aq = 227
    g = 17
    gb = 14
    r = 0
    def H_drive(lamb, dfreq, damp):
        H = 0
        H += (fr + lamb - dfreq)*qt.tensor([a.dag()*a, qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)])
        H += (fq - lamb - dfreq)*qt.tensor([qt.qeye(N), b.dag()*b, qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c.dag()*c, qt.qeye(Nb), qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), c.dag()*c, qt.qeye(Nb)])
        H += (fr - dfreq)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), c.dag()*c])
        H += g*(qt.tensor([a.dag(), b, qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)]) + qt.tensor([a, b.dag(), qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), c, qt.qeye(Nb), qt.qeye(Nb)]) + qt.tensor([a, qt.qeye(Nq), c.dag(), qt.qeye(Nb), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), qt.qeye(Nb), c, qt.qeye(Nb)]) + qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb), c.dag(), qt.qeye(Nb)]))
        H += gb*(qt.tensor([a.dag(), qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), c]) + qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), c.dag()]))
        H += -(aq/2)*qt.tensor([qt.qeye(N), b.dag()*b.dag()*b*b, qt.qeye(Nb),  qt.qeye(Nb), qt.qeye(Nb)])
        H += damp*qt.tensor([(a + a.dag()), qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)])
        return H
    
    H0 = H_drive(lamb, 0, 0)
    fd = H0.eigenstates()[0][1] - H0.eigenstates()[0][0]
    powers = np.logspace(-3, 0.8, 51, endpoint=True)
    cops = [np.sqrt(kappa)*qt.tensor([a, qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)]), 
            np.sqrt(gamma)*qt.tensor([qt.qeye(N), b, qt.qeye(Nb), qt.qeye(Nb), qt.qeye(Nb)]), 
            np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), c, qt.qeye(Nb), qt.qeye(Nb)]), 
            np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), c, qt.qeye(Nb)]),
            np.sqrt(gammab)*qt.tensor([qt.qeye(N), qt.qeye(Nq), qt.qeye(Nb), qt.qeye(Nb), c])]
    
    @background
    def find_ss(power):
        H = H_drive(lamb, fd, power)
        rho = qt.steadystate(H, cops)
        x = rho
        return x
    
    bsize = 13
    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[:bsize]])              
    data1 = loop.run_until_complete(looper)  

    print('done batch 1')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[bsize:2*bsize]])              
    data2 = loop.run_until_complete(looper)  

    print('done batch 2')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[2*bsize:3*bsize]])              
    data3 = loop.run_until_complete(looper)  

    print('done batch 3')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[3*bsize:]])              
    data4 = loop.run_until_complete(looper)  

    print('done batch 4')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    
    data = data1 + data2 + data3 + data4

    psim = (1/np.sqrt(2))*((1/np.sqrt(3))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1), qt.basis(Nb, 0), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 1), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 0), qt.basis(Nb, 1)])) - qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0), qt.basis(Nb, 0), qt.basis(Nb, 0)]))
    psip = (1/np.sqrt(2))*((1/np.sqrt(3))*(qt.tensor([qt.basis(N, 0), qt.basis(Nb, 1), qt.basis(Nb, 0), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 1), qt.basis(Nb, 0)])+qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 0), qt.basis(Nb, 1)])) + qt.tensor([qt.basis(N, 1), qt.basis(Nb, 0), qt.basis(Nb, 0), qt.basis(Nb, 0)]))
    vacs = qt.tensor([qt.basis(N, 0), qt.basis(Nb, 0), qt.basis(Nb, 0), qt.basis(Nb, 0)])

    vac = np.zeros(len(powers))
    sp = np.zeros(len(powers))
    sm = np.zeros(len(powers))
    for i, p in enumerate(powers):
        rho = data[i].ptrace([0,2,3,4])
        vac[i] = np.abs(rho.overlap(vacs))
        sp[i] = np.abs(rho.overlap(psip))
        sm[i] = np.abs(rho.overlap(psim))

    df_dict = {
        'power': powers,
        'vac': vac,
        'sp': sp,
        'sm': sm 
    }
    df = pd.DataFrame(df_dict)
    df.to_csv('N3-blockade-sim-test-d7.csv')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))