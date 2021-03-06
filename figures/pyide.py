
import numpy as np
import matplotlib.pyplot as plt
    
# class for handling growth functions
#plt.rcParams.update({"text.usetex": True})
class GrowthFunction:
    
    def __init__(self, g):
        self.g = g
    
    def show(self, title, xlims, ylims):
        
        g = self.g
        U = np.arange(*xlims, 0.01)
        
        fig,axs = plt.subplots()
        
        #plt.rcParams.update({"text.usetex": True})

        axs.plot(U, g(U), label="$u_{t+1} = g(u_t)$")
        axs.plot(U,U, color="black", linestyle=":",
        label="$u_{t+1} = u_t$")
        
        axs.set_xlabel("Population at time $t$")
        axs.set_ylabel("Population at time $t+1$")
        axs.legend()
        axs.set_title(title)
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        
        plt.show()
    
    def plot_cobweb_diagram(self, x0):
        pass

class RickerGrowthFunction(GrowthFunction):
    
    def __init__(self, r, K=1):
        self.set_parameter(r, K)
    
    def set_parameter(self, r, K=1):
        self.g = lambda u :  u * np.exp(r*(1-u/K))
        
    def show_multiple(self, title, xlims, ylims, rrange):
        
        U = np.arange(*xlims, 0.01)
        
        fig,axs = plt.subplots()
        
        #plt.rcParams.update({"text.usetex": True})
        
        for r in rrange:
            self.set_parameter(r)
            g = self.g
            axs.plot(U, g(U), label="$r="+str(r)+"$")
            
        axs.plot(U,U, color="black", linestyle=":")
        
        axs.set_xlabel("Population at time $t$")
        axs.set_ylabel("Population at time $t+1$")
        axs.legend()
        axs.set_title(title)
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        
        plt.show()

class LogisticGrowthFunction(GrowthFunction):
    
    def __init__(self, r):
        self.set_parameter(r)
    
    def set_parameter(self, r):
        self.g = lambda u :  u + r * u * (1-u)
        
        
class DisperalKernel:
    
    def __init__(self, k):
        self.k = k
    
    def show(self, title, xlims, ylims, step_size):
        
        k = self.k
        X = np.arange(*xlims, step_size)
        
        fig,axs = plt.subplots()
        
        #plt.rcParams.update({"text.usetex": True})

        axs.plot(X, k(X), label="$y = k(x)$")
        
        axs.set_xlabel("Spatial location")
        axs.set_ylabel("Probability density")
        axs.legend()
        axs.set_title(title)
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)
        
        plt.show()

class GaussianDisperalKernel(DisperalKernel):
    
    def __init__(self, mu=0, sigma=1):
        from scipy.stats import norm
        self.k = lambda x : norm.pdf(x, loc=mu, scale=np.sqrt(sigma))

class LaplaceDisperalKernel(DisperalKernel):
    
    def __init__(self, shape=1):
        self.k = lambda x : 1/(2*shape) * np.exp(-np.abs(x)/shape)


class IDEModel_old:

    def __init__(self, growthFunction, disperalKernel, xmin, xmax, dx):
        self.g = growthFunction.g
        self.k = disperalKernel.k
        self.xmin = xmin
        self.xmax = xmax
        self.dx = dx
        N = int((xmax-xmin)/dx/2)
        
        self.domain = np.linspace(xmin,xmax,2*N+1)
        
        domain_large = np.linspace(2*xmin,2*xmax,4*N+1)
        
        k = self.k
        pdf_array_large = k(domain_large)
        self.pdf_array = pdf_array_large[N:(3*N+1)]
        
        cdf_array_left = [0]
        for i in range(2*N):
            cdf_array_left.append(pdf_array_large[i]+cdf_array_left[i])
        
        cdf_array_right = [0]
        for i in range(2*N):
            cdf_array_right.append(pdf_array_large[4*N-i]+cdf_array_right[i])
        cdf_array_right = cdf_array_right[::-1]
        
        self.cdf_array_left = dx*np.array(cdf_array_left)
        self.cdf_array_right = dx*np.array(cdf_array_right)
            
        return
    
    def set_initial_density(self,u0):
        X = self.domain
        self.seq = [u0(X)]
        return

    def iterate(self,U):
        X = self.domain
        step_size = X[1] - X[0]
        
        g = self.g
        
        N = int((len(X)-1)/2)
        
        pdf = self.pdf_array
        cdf_left = self.cdf_array_left
        cdf_right = self.cdf_array_right
        
        U_center = step_size * np.convolve(pdf, g(U), mode='same')
        U_left = g(U[-1]) * cdf_left
        U_right = g(U[0]) * cdf_right
        
        return U_center + U_left + U_right
    
    def run(self,time_steps):
        
        U_sequence = self.seq
        
        U = U_sequence[0]
        
        for t in range(time_steps):
            U = self.iterate(U)
            U_sequence.append(U)
            
        self.seq = U_sequence
    
    def plot(self, times=[]):
    
        X = self.domain
        U_seq = self.seq
        T = len(U_seq)
        
        if len(times)==0:
            times = np.arange(T)
            
        fig,axs = plt.subplots()
        
        for t in times:
            U = U_seq[t]
            axs.plot(X, U, color=(t/T,1-t/T,0))
        axs.set_xlabel("Spatial position")
        axs.set_ylabel("Population density")
        plt.show()
        
 

class IDEModel:

    def __init__(self, growthFunction, dispersalKernel):
        self.growthFunction = growthFunction
        self.dispersalKernel = dispersalKernel
        
        
class IDESimulation:

    def __init__(self, model):
        self.model = model
    
    def setDomain(self, xmin, xmax, dx):
    
        self.xmin = xmin
        self.xmax = xmax
        self.step_size = dx
        N = int((xmax-xmin)/dx/2)
    
        self.domain = np.linspace(xmin,xmax,2*N+1)
        
        domain_size = xmax-xmin
        self.domain_size = domain_size
        
        domain_large = np.linspace(-2*domain_size, 2*domain_size, 4*N+1)
    
        k = self.model.dispersalKernel.k
        pdf_array_large = k(domain_large)
        self.pdf_array = pdf_array_large[N:(3*N+1)]
        cdf_right = pdf_array_large[0:(2*N+1)]
        cdf_left = pdf_array_large[(2*N+1):(4*N+1)]
        
        #A = np.triu(np.ones((n,n))) - np.eye(n)
        #B = np.tril(np.ones((n,n))) - np.eye(n)
        
        #upper_triangular = np.ones(
        
        cdf_array_left = [0]
        for i in range(2*N):
            cdf_array_left.append(pdf_array_large[i]+cdf_array_left[i])
        
        cdf_array_right = [0]
        for i in range(2*N):
            cdf_array_right.append(pdf_array_large[4*N-i]+cdf_array_right[i])
        cdf_array_right = cdf_array_right[::-1]
        
        self.cdf_array_right = 2*dx*np.array(cdf_array_left)
        self.cdf_array_left = 2*dx*np.array(cdf_array_right)
        
        
        
    
    
    
    def setInitialCondition(self, u0):
    
        X = self.domain
        self.timeSeries = [u0(X)]
        
        self.leftBoundary = u0(self.xmin)
        self.rightBoundary = u0(self.xmax)
        
    
    def setBoundaryCondition(self, boundary):
        self.boundaryCondition = boundary
        
    def run(self,time_steps):
        
        def Q(self,U):
            X = self.domain
            step_size = self.step_size
            
            g = self.model.growthFunction.g
            
            N = int((len(X)-1)/2)
            
            pdf = self.pdf_array
            cdf_left = self.cdf_array_left
            cdf_right = self.cdf_array_right
            
            #U_center = np.fft.ifft(np.fft.fft(pdf) * np.fft.fft(g(U)))
            #U_center = 2*step_size*U_center[::-1]
            U_center = 2*step_size*np.convolve(pdf, g(U), mode='same')
            
            if self.boundaryCondition == "dynamic":
                self.leftBoundary = g(self.leftBoundary)
                self.rightBoundary = g(self.rightBoundary)
            elif self.boundaryCondition == "static":
                self.leftBoundary = g(self.leftBoundary)
                self.rightBoundary = self.rightBoundary
            
            U_left = self.leftBoundary * cdf_left
            U_right = self.rightBoundary * cdf_right
            
            return U_center + U_left + U_right
        self.Q = Q
        
        U = self.timeSeries[0]
        Q = self.Q
        
        for t in range(time_steps):
        
            U = Q(self, U)
            
            self.timeSeries.append(U)
            
    def plot_wavespeed(self, predictedcstar, predictedc1, predictedc2, level=0.1, times=[], file=None):
    
        X = self.domain
        U_seq = self.timeSeries
        T = len(U_seq)
        
        if len(times)==0:
            times = np.arange(T)
            
        fig,axs = plt.subplots()
        
        d0 = self.step_size*np.sum(U_seq[0]>level)
        C = []
        for t in times[1:]:
            
            U = U_seq[t]
            d1 = self.step_size*np.sum(U>level)
            C.append((d1-d0))
            d0 = d1
            
        axs.plot(times[1:], C, color='black')
        #axs.plot(times[1:], C, 'o', color='blue')
        axs.set_xlabel('$n$')
        axs.set_ylabel('$c_n(a)$', rotation=0, labelpad=40)
        
        plt.plot([1, max(times)], [predictedcstar, predictedcstar], 'b--')
        plt.plot([1, max(times)], [predictedc1, predictedc1], 'b:')
        plt.plot([1, max(times)], [predictedc2, predictedc2], 'b:')
        
        ax2 = axs.twinx() 
  
        axs.set_ylim(0.5, 1)
        ax2.set_ylim(0.5, 1)
    
        #color = 'tab:green'
        #ax2.set_ylabel('Y2-axis', color = color) 
        ax2.plot(times[1:], C, 'o', color='black')
        #ax2.tick_params(axis ='y', labelcolor = color) 
        ax2.set_xticks([])
        ax2.set_yticks([predictedc1,predictedcstar,predictedc2])
        ax2.set_yticklabels(['$c_1^*(a)$','$c^*$','$c_2^*(a)$'])

        try:
            plt.savefig(file,bbox_inches='tight')
        except:
            pass
        return plt
        
    def plot(self, times=[], file=None):
    
        X = self.domain
        U_seq = self.timeSeries
        T = len(U_seq)
        
        if len(times)==0:
            times = np.arange(T)
            
        fig,axs = plt.subplots()
        
        for t in times:
            U = U_seq[t]
            axs.plot(X, U, color=(t/T,1-t/T,0))
        axs.set_xlabel('$x$')
        axs.set_ylabel('$u_n(x)$', rotation=0, labelpad=40)
        try:
            plt.savefig(file,bbox_inches='tight')
        except:
            pass
        return plt
    