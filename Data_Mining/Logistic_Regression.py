import numpy as np
from scipy import stats

# likelihood function when normal distribution
def norm(x, mu = 0, sigma = 1):
    return 1/np.sqrt(2*np.pi)/sigma * np.exp(-(x-mu)**2/2/sigma**2)

D = [2.61, 3.73, 2.80, 4.29, 3.12]

# np.prod : array 내부의 곱
np.prod([1,2,3,4])

np.prod([norm(x, mu=-15) for x in D])
np.sum([np.log(norm(x, mu=-15)) for x in D])



from scipy.optimize import minimize

# Log likelihood function(generalization)
def logL(x,data,sigma):
    return np.sum([-np.log(norm(d,mu=x,sigma=sigma)) for d in data])

# 최적화 값 찾기, x가 3.31일때
res = minimize(logL, 0 ,(D,1), method = 'Nelder-Mead', options = {'xatol':1e-8, 'disp':True})
print(res)
# 이 값이 3.31이다
print(np.mean(D))
# 결론적으로 res == np.mean(D)