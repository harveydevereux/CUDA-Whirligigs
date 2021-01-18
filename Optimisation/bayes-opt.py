import numpy as np
import subprocess
import scipy.stats as stats
import time
import sys

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import SequentialDomainReductionTransformer

N=199
kde_domain = (0,1)
dx = 0.01
n_runs = 3
geometry = "circle"
γ = 0.5

D200 = np.loadtxt("density-200.csv")
D100 = np.loadtxt("density-100.csv")
D50 = np.loadtxt("density-50.csv")

kde_200 = stats.gaussian_kde(D200)
kde_100 = stats.gaussian_kde(D100)
kde_50 = stats.gaussian_kde(D50)

B_kde = [kde_50,kde_100,kde_200]

def KLDivergenceKDE(kde_p,kde_q,domain=(0,1),dx=0.01):
    X = np.linspace(domain[0],domain[1],int(np.floor((domain[1]-domain[0])/dx)))
    L = 0.0
    for x in X:
        l = (kde_p(x)*np.log2(kde_p(x)/kde_q(x)))*dx
        if np.isnan(l) == False and np.isinf(l) == False:
            L += l

    return L

def MSEKDE(kde_p,kde_q,domain=(0,1),dx=0.01):
    X = np.linspace(domain[0],domain[1],int(np.floor((domain[1]-domain[0])/dx)))
    L = 0.0
    for x in X:
        L += (kde_p(x)-kde_q(x))**2.0
    return L/X.shape[0]

def BeetleModellLossCircle(alpha,tau,mu):
    v = 13.3*2.0
    try:
        L = 0.0
        p = [alpha,tau,mu]
        for i in range(n_runs):
            cmd = "../CUDAABP --initial-packing-fraction 0.5 -N {} -T 10 -a 1 -b 1 -alpha {} -tau {} -dt 0.00111111 -mu {} -mur 1 -v {} -Dr 2.37 --random-seed {} -silent 1".format(N,alpha,tau,mu,v,int(time.time()))
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            T = np.loadtxt("trajectories.txt",delimiter=',')
            h = int(np.ceil(T.shape[0]/2.0))
            D = T[h:-1,3]
            kde_data = stats.gaussian_kde(D)
            L += MSEKDE(kde_data,kde_200,kde_domain,dx)
        print(p,-L[0]/n_runs)
        return -L[0]/n_runs
    except Exception as ex:
        if type(ex) == KeyboardInterrupt:
            sys.exit(0)
        return -1e9

def BeetleModellLossEllipse(alpha,tau):
    try:
        L = 0.0
        p = [alpha,tau]
        for i in range(n_runs):
            cmd = "./CUDAABP --initial-packing-fraction 0.5 -N {} -T 5 -a 1 -b 0.5 -alpha {} -tau {} -dt 0.00111111 -mu 30 -mur 30 -v 32 -Dr 2.37 --random-seed {} -silent 1".format(N,alpha,tau,int(time.time()))
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            T = np.loadtxt("trajectories.txt",delimiter=',')
            h = int(np.ceil(T.shape[0]/2.0))
            D = T[h:-1,3]
            kde_data = stats.gaussian_kde(D)
            L += MSEKDE(kde_data,kde_200,kde_domain,dx)
        print(p,-L[0]/n_runs)
        return -L[0]/n_runs
    except Exception as ex:
        if type(ex) == KeyboardInterrupt:
            sys.exit(0)
        return -1e9

if geometry == "circle":
    loss = BeetleModellLossCircle
    t1 = time.time()
    loss(-0.5,1.0,20.)
    t = time.time()-t1

    pbounds = {'alpha': (-2.0, 0.0), 'tau': (0., 10.), 'mu': (20.0,40.0)}

elif geometry == "ellipse":
    loss = BeetleModellLossEllipse
    t1 = time.time()
    loss(-0.5,1.0)
    t = time.time()-t1

    pbounds = {'alpha': (-2.0, 0.0), 'tau': (0., 10.), 'v': (15.,60.)}


max_capital = int(np.floor(0.99*4*60*60 / t))
print("Loss runtime {}, max_capital (runs) {}, proportion of random points {}".format(t,max_capital,γ))

# optimizer = BayesianOptimization(
#     f=BeetleModellLoss,
#     pbounds=pbounds,
#     verbose=1, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )

bounds_transformer = SequentialDomainReductionTransformer()
optimizer = BayesianOptimization(
    f=loss,
    pbounds=pbounds,
    verbose=1,
    random_state=1,
    bounds_transformer=bounds_transformer
)

try:
    print("New optimiser knows",len(optimizer.space), " points")
    print("Trying to load logs at ./seed-logs.json")
    load_logs(optimizer, logs=["./seed-logs.json"]);
    print("Logs loaded: optimiser knows",len(optimizer.space), " points")
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
except:
    print("No logs found, skipping")

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

optimizer.maximize(
    init_points=int(np.floor(max_capital*γ)),
    n_iter=int(np.floor(max_capital*(1-γ))),
)

print(optimizer.max)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
