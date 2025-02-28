import numpy as np

from probset import ProbSetup

pn = 4
w = 'H'
ad = 'remote/res/%s_case%d_'%(w, pn)

ers = np.load(ad+'ers.npy')

print(ers)

Hlist = ['$\\frac{1}{12}$', '$\\frac{1}{24}$', '$\\frac{1}{48}$']
llist = [5, 7, 8]
# ers *= 100

for i in range(ers.shape[1]):
    if w == 'H':
        print(Hlist[i], '& %d'%llist[i], end=' & ')
        
    for j in range(ers.shape[0]):
        for k in range(ers.shape[2]):
            if (j == ers.shape[0]-1) & (k == ers.shape[2]-1):
                print("%.2e"%ers[j, i, k], end='\n')
            else:
                print("%.2e"%ers[j, i, k], end=' & ')

    print("\\\\ \hline")


# NX = NY = 192
# ps = ProbSetup(pn, NX, NY, 'res/')

# k, Psi, f = ps.generate_kfI()
# k = np.log10(k)
# ps.plot_average(k, r'$\log_{10}(\kappa_{%d})$'%pn, 'k%d.png'%pn)


