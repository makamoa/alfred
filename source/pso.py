# ------------------------------------------------

# Andrea Fratalocchi
# Particle Swarm Optimization: Matlab version with adaptive inertia
# Aug 2017

# ------------------------------------------------

# IMPORT DEPENDENCIES

import numpy as np
from matplotlib import pyplot as plt


# CLASSES

# ------------- SINGLE PARTICLE ------------------

class particle(object):


    # constructor
    def __init__(self,dim,fun,bond,init):
        """init a single particle of size dim with:
         fitness function fun and boundary function bond        
         """

        ####### INIT PARTICLES
        self.dim=dim
        self.x=np.zeros(self.dim)
        self.v=np.zeros(self.dim)
        init(self.x,self.v)

        #setup fitness & boundary
        self.fitfun=fun
        self.boundary=bond
        
        #evaluate fitness & best
        self.fitness=self.fitfun(self.x)
        self.fbest=self.fitness             
        self.xbest=self.x
        
    def update_fit(self):
        """evaluate current fitness and fbest"""
        self.fitness=self.fitfun(self.x)
        #update best
        if self.fitness < self.fbest:
            self.fbest=self.fitness
            self.xbest=self.x

    def update_vel(self,wcs,gxbest):
        """update particle velocity with:
        wcs[0] inertia coefficient
        wcs[1] cognitive coefficient
        wcs[2] social coefficient
        gxbest group best position found 
        """
        u=np.random.rand(self.dim)
        vel_cog=wcs[1]*u*(self.xbest-self.x)
        u=np.random.rand(self.dim)
        vel_soc=wcs[2]*u*(gxbest-self.x)
        self.v=wcs[0]*self.v+vel_cog+vel_soc

    def update_pos(self):
        """update position and enforce bounds
        if outside it put it equal to the bound"""
        #update
        self.x=self.x+self.v
        ###enforce bounds    
        self.boundary(self.x)

    def prt(self):
        """print particle info"""
        print('pos',self.x)
        print('vel',self.v)
        print('xbest',self.xbest)
        print('fbest',self.fbest)
        print('fcurr',self.fitness)


# ------------- PSO SEARCH ------------------

class pswarm(object):
    """particle swarm optimization class""" 
        
    def __init__(self,opts):
        """PSO solver with options in dictionary opts
        opts['dim'] #particle size
        opts['maxIter'] maxiter
        opts['inertiaR'] inertia range
        opts['minNfrac'] min neighborhood fraction
        opts['cognitive'] cognitive coefficient
        opts['social'] social coefficient
        opts['fun'] #fitness function
        opts['swarmSize'] #swarm size
        opts['boundary'] #boundary function
        opts['init_part'] #init particle function
        opts['rng'] #init seed of random number generator
        """
        
        #init internal
        self.rng=opts['rng'] #init seed of random number generator
        #np.random.seed(self.rng)
        self.dim=opts['dim'] #particle size
        self.maxIter=opts['maxIter']

        self.swarmSize=opts['swarmSize'] #swarm size
        
        self.minNfrac=opts['minNfrac']
        self.minNsize=max([1,np.int(np.floor(self.swarmSize*self.minNfrac))]) #min neghborhood size
        self.N=self.minNsize 

        self.wcs=np.zeros((3,1)); # w inertia c cognitive s social
        self.inertiaR=opts['inertiaR']
        self.wcs[0]=self.inertiaR.max() # set inertia
        self.wcs[1]=opts['cognitive']
        self.wcs[2]=opts['social']
        
        self.sc=0 #stall counter

        self.fun=opts['fun'] #fitness function
        self.boundary=opts['boundary'] #boundary function
        self.init_part=opts['init_part'] #init particle function
        
        # init the particles        
        self.swarm=np.empty(self.swarmSize,dtype=object) #array of particles
        for i in np.arange(0,self.swarmSize): 
            self.swarm[i]=particle(self.dim,self.fun,self.boundary,self.init_part)

        # global best function and location
        self.fbest=self.swarm[0].fbest
        self.gxbest=self.swarm[0].x
        for i in np.arange(1,self.swarmSize):
            tmp=self.swarm[i].fbest
            if tmp<self.fbest:
                self.fbest=tmp
                self.gxbest=self.swarm[i].x

        # output evolution
        self.x=np.zeros((self.dim,self.swarmSize,self.maxIter)) #positions
        self.currit=0 #current iteration
        self.update_stats()
    
        # optimization loop
        parts=np.arange(0,self.swarmSize-1)
        go=True
        while go:
            #main cycle            
            improved =False
            for i in np.arange(0,self.swarmSize):

                #Choose a random subset S of N particles other than i         
                np.random.shuffle(parts)
                subset=parts[0:self.N].copy()
                msk=[subset>=i]
                subset[msk]=subset[msk]+1

                #Find fbest(S), the best objective function among 
                #the neighbors, and g(S), the position of the 
                #neighbor with the best objective function.
                fbest=self.swarm[i].fbest
                gxbest=self.swarm[i].x
                for k in subset:
                    tmp=self.swarm[k].fbest
                    if tmp<fbest:
                        fbest=tmp
                        gxbest=self.swarm[k].x
                
                #update particle i velocity
                self.swarm[i].update_vel(self.wcs,gxbest)
                
                #Update the position x = x + v.
                #Enforce the bounds. If any component of x is outside 
                #a bound, set it equal to that bound.
                self.swarm[i].update_pos()

                #Evaluate the objective function f = fun(x).
                #If f < fun(p), then set p = x. This step 
                #ensures p has the best position the particle has seen.
                self.swarm[i].update_fit()
                
                #If f < b, then set b = f and d = x. 
                #This step ensures b has the best 
                #objective function in the swarm, 
                #and d has the best location.                
                if self.swarm[i].fbest<self.fbest:
                    self.fbest=self.swarm[i].fbest
                    self.gxbest=self.swarm[i].x
                    improved =True

            if(improved):
                self.if_improved()
                #Update the neighborhood.
                #Set c = max(0,c-1).
                self.sc=np.max([0,self.sc-1])
                    
                #Set N to minNeighborhoodSize.
                self.N=self.minNsize

                #If c < 2, then set W = 2*W.
                if self.sc<2:
                    self.wcs[0]=2.*self.wcs[0]
                
                #If c > 5, then set W = W/2.
                if self.sc>5:
                    self.wcs[0]=0.5*self.wcs[0]
    
                #Ensure that W is in the bounds of 
                #the InertiaRange option.
                if self.wcs[0]>self.inertiaR[1]:
                    self.wcs[0]=self.inertiaR[1]
                if self.wcs[0]<self.inertiaR[0]:
                    self.wcs[0]=self.inertiaR[0]

            else:
                    
                #Set c = c+1.
                self.sc=self.sc+1            

                self.N=np.min([self.N+self.minNsize,self.swarmSize])

            #update stats
            self.update_stats()

            #check output
            self.currit=self.currit+1
            if self.currit==self.maxIter:
                go=False #exit
                self.outputFlag=0                        
            self.show()
        # output info
        #self.plot_output()
    def if_improved(self):
        pass

    def show(self):
        print("current iteration = " + str(self.currit))
        print("current fbest = " + str(self.fbest))

    def update_stats(self):
        """update output statistics"""
        for i in np.arange(0,self.swarmSize):
            self.x[:,i,self.currit]=self.swarm[i].x
