import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

df = pd.read_csv('global.1751_2017.csv')

# Display the first few rows of the dataframe

second_column = df.iloc[:, 1]

# Convert the column to numeric, forcing errors to NaN
second_column_numeric = pd.to_numeric(second_column, errors='coerce')

# Remove potential NaN values
second_column_numeric = second_column_numeric.dropna()

# Convert to NumPy array
second_column_array = second_column_numeric.to_numpy()

# Slice and scale the array
emission_rate = second_column_array[50:] / (10**3)

C_s0 = 1500 #1500*(10**12)
C_ve0 = 550 #550*(10**12)
C_a0 = 596 #596*(10**12)
C_oc0 = 1.5 * 10**5 #1.5*(10**17)

f_gtm = 8.3259*(10**13)
K_a = 1.773*(10**20)
K_p = 0.184
K_A = 8.7039 * (10**9)

K_mm = 1.478
K_c = 29*(10**-6)
K_m = 120*(10**-6)
T_0 = 288.15
K_r = 0.092


E_a = 54.83
R = 8.314
K_sr = 0.034

K_b = 157.072
K_t = 0.092
F_0 = 2.5*(10**-2)
xi = 0.3

zeta = 50

H = 0.5915
P_0 = 1.4*(10**11)
Latent = 43655
A = 0.225
S = 1368
cappacity = 4.69*(10**23) #23
sigma = 5.67*(10**-8)
a_E = 5.101*(10**14)
f_max =  5
omega =  3
T_c = 1.5 #0.1


epsilon_max = 6
saturation = 50


r_0 = 5
def simulate(k,beta,delta,r_max,T_critical,index):


  def epsilon(t):
    t = int(t)

    if t<216:
      return emission_rate[t]
    else:
      return (((t-216)*epsilon_max)/(t-216+saturation)) + emission_rate[216]


  def runaway(T,r_max):
    if index == 1:
      return 0
    else:
      return r_max/(1 + np.exp(-r_0*(T-T_critical)))

  def P_co2(C_a):

    return f_gtm*(C_a+C_a0)/K_a

  def P(C_a,T):
    a = K_p*C_ve0*K_mm
    b = (P_co2(C_a) - K_c)/(K_m + P_co2(C_a) - K_c)
    c = ((15+T)**2)*(25-(T))/5625
    if P_co2(C_a)-K_c>0 and T > -15 and T<25:
      return a*b*c
    else:
      return 0

  def R_veg(C_v,T):
    a = K_r*(C_v)*K_A
    b = np.exp(-E_a/(R*(T+T_0)))
    return a*b

  def R_so(T,C_so):
    a = K_sr*(C_so)*K_b
    b = np.exp(-308.56/(T+T_0-227.13))
    return a*b

  def L(C_v):
    return K_t*(C_v)

  def F_oc(C_a,C_oc):
    a = F_0*xi
    b = C_a
    c = zeta * C_a0*(C_oc)/C_oc0
    return a*(b-c)

  def tau(C_a,T):
    a = 1.73*(P_co2(C_a))**(0.263)
    b = 0.0126*(H*P_0*np.exp(-Latent/(R*(T+T_0))))**(0.503)
    c = 0.0231

    return a+b+c


  def F_d(C_a,T):
    a = (1-A)*S/4
    b = (1+ 0.75*tau(C_a,T))

    return a*b

  def diff_C_a(t,z):
    a = epsilon(t)*(1-z[5])
    b = P(z[0],z[4])

    c = R_veg(z[2],z[4])
    d = R_so(z[4],z[3])
    e = F_oc(z[0],z[1])
    f = runaway(z[4],r_max)
    return a-b+c+d-e+f

  def diff_C_o(t,z):
    return F_oc(z[0],z[1])

  def diff_C_v(t,z):

    a = P(z[0],z[4])
    b = R_veg(z[2],z[4])
    c = L(z[2])
    return a-b-c

  def diff_C_so(t,z):
    a = L(z[2])
    b = R_so(z[4],z[3])

    return a-b

  def diff_T(t,z):
    a = a_E/cappacity
    b = (F_d(z[0],z[4])-(sigma*(z[4]+T_0)**4))

    return a*b*3.14*10**7

  def f_T(T):
      a = f_max
      b = 1 + np.exp(-omega*(T-T_c))
      return a/b

  Diff_x = []
  def diff_x(t,z):
    if t<216:
      return 0
    else:
      a = k*z[5]*(1-z[5])
      b = (-beta + f_T(z[4]) + delta*(2*z[5]-1))
      Diff_x.append(a*b)
    return a*b


 

  def model(t, z):
      return np.array([
          diff_C_a(t, z),
          diff_C_o(t, z),
          diff_C_v(t,z),
          diff_C_so(t,z),
          diff_T(t,z),
          diff_x(t,z),

      ])


  z0 = np.array([0, 0, 0, 0, 0, 0.05])

  # Time span for the simulation
  simulation_time = 400
  t_span = (0, simulation_time)

  # Solve the system of ODEs
  sol = solve_ivp(model, t_span, z0, method='BDF', t_eval=np.linspace(0, simulation_time, simulation_time*100))

  # Plot the results
  # plt.plot(sol.t, sol.y.T)  # Note that sol.y still refers to the solution array
  # plt.xlabel('Time')
  # plt.ylabel('z(t)')
  # plt.title('Simulation of 10 Differential Equations Using z Variables')
  # plt.legend(['z1', 'z2', 'z3', 'z4', 'z5'])
  # plt.show()
  return sol.t,sol.y.T[:,0],sol.y.T[:,1],sol.y.T[:,2],sol.y.T[:,3],sol.y.T[:,4],sol.y.T[:,5]


#first plot of the manuscript

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    'font.size': 16,       # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # X and Y labels
    'xtick.labelsize': 14, # X-axis ticks
    'ytick.labelsize': 14, # Y-axis ticks
    'legend.fontsize': 14  # Legend
})

def plot_both(k, r_max):
    time1, C_a1, C_o1, C_v1, C_so1, Temp1, X1, Y11, Y21, Y31, Y41 = simulate(k, 1, 1, r_max, 2, 1)
    time2, C_a2, C_o2, C_v2, C_so2, Temp2, X2, Y12, Y22, Y32, Y42 = simulate(k, 1, 1, r_max, 2, 2)

    plt.figure(figsize=(10, 6))
    plt.plot(time1 + 1800, Temp1, label='Baseline Model')
    plt.plot(time2 + 1800, Temp2, label='Modified Model')
    plt.legend(loc='upper left', fontsize=14)
    plt.xlabel('Time (year)', fontsize=16)
    plt.ylabel('Temperature Anomaly (celsius)', fontsize=16)

    ax = plt.gca()  # Get current axis
    ax.set_ylim(top=5)  # Extend the vertical axis to a maximum of 5

    # Create an inset of the main plot for X1 and X2 with bbox_to_anchor
    inset_ax = inset_axes(ax, width="30%", height="30%",
                          bbox_to_anchor=(-0.25, 0, 1, 1),  # (x0, y0, width, height)
                          bbox_transform=ax.transAxes, loc='center')

    inset_ax.plot(time1 + 1800, X1)
    inset_ax.plot(time2 + 1800, X2)
    inset_ax.set_xlabel('Time (year)', fontsize=14)
    inset_ax.set_ylabel('X', fontsize=14)
    #inset_ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('figure_for_paper.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_both(1*10**(-1),1)

#second plot of the manuscript panel A and D

def AUC(Y,X):
    dx = X[1] - X[0]
    auc = np.trapz(Y, dx=dx)

    return auc

K = np.linspace(0,0.2,50) #0.2
R_max = np.linspace(0,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_k = 0
for r_max in R_max:
  for k in K:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(k,1,1,r_max,3,2)
    time2,C_a2,C_o2,C_v2,C_so2,Temp2,X2,Y12,Y22,Y32,Y42 = simulate(k,1,1,0,3,2)
    auc1 = AUC(Temp1,time1)
    auc2 = AUC(Temp2,time2)
    diff_auc = auc1 - auc2
    max_info[count_r,count_k] = diff_auc
    count_k = count_k + 1
  count_k = 0
  count_r = count_r + 1

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 16,       # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # X and Y labels
    'xtick.labelsize': 14, # X-axis ticks
    'ytick.labelsize': 14, # Y-axis ticks
    'legend.fontsize': 14  # Legend
})
# Assuming K, R_max, and max_info are defined elsewhere
X, Y = np.meshgrid(K, R_max)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Difference in Area Under the Curve (AUC)')

# Add contour lines
contour = plt.contour(X, Y, max_info, colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)

plt.xlabel('K',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Difference in Area Under the Curve Vs $R_{\mathrm{max}}$ and Social Learning Rate',fontsize= 12)
plt.show()


#Second plot of the manuscript panel B,E

def AUC(Y,X):
    dx = X[1] - X[0]
    auc = np.trapz(Y, dx=dx)

    return auc

Beta = np.linspace(0,2,50)
R_max = np.linspace(0,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_k = 0
for r_max in R_max:
  for beta in Beta:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(0.01,beta,1,r_max,2)
    auc1 = AUC(Temp1,time1)
    diff_auc = auc1
    max_info[count_r,count_k] = diff_auc
    count_k = count_k + 1
  count_k = 0
  count_r = count_r + 1

plt.rcParams.update({
    'font.size': 16,       # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # X and Y labels
    'xtick.labelsize': 14, # X-axis ticks
    'ytick.labelsize': 14, # Y-axis ticks
    'legend.fontsize': 14  # Legend
})
X, Y = np.meshgrid(Beta,R_max)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Difference in Area Under the Curve (AUC)')

contour = plt.contour(X, Y, max_info, colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)


plt.xlabel('Beta',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Difference in Area Under the Curve Vs $R_{\mathrm{max}}$ and Net Cost of Mitigation',fontsize=12)
plt.show()


#Second plot of the manuscript panel C,F


def AUC(Y,X):
    dx = X[1] - X[0]
    auc = np.trapz(Y, dx=dx)

    return auc

Delta = np.linspace(0,2,50)
R_max = np.linspace(0,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_beta = 0
for r_max in R_max:
  for delta in Delta:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(0.01,1,delta,r_max,3,2)
    time2,C_a2,C_o2,C_v2,C_so2,Temp2,X2,Y12,Y22,Y32,Y42 = simulate(0.01,1,delta,0,3,2)
    auc1 = AUC(Temp1,time1)
    auc2 = AUC(Temp2,time2)
    diff_auc = auc1 - auc2
    max_info[count_r,count_beta] = diff_auc
    count_beta = count_beta + 1
  count_beta = 0
  count_r = count_r + 1


plt.rcParams.update({
    'font.size': 16,       # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # X and Y labels
    'xtick.labelsize': 14, # X-axis ticks
    'ytick.labelsize': 14, # Y-axis ticks
    'legend.fontsize': 14  # Legend
})
X, Y = np.meshgrid(Delta,R_max)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Difference in Area Under the Curve (AUC)')

contour = plt.contour(X, Y, max_info, colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)

plt.xlabel('Delta',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Difference in Area Under the Curve Vs $R_{\mathrm{max}}$ and Strength of Social Norms',fontsize = 12)
plt.show()


#Figure 5

def AUC(Y,X):
    dx = X[1] - X[0]
    auc = np.trapz(Y, dx=dx)

    return auc

K = np.linspace(0,0.1,50)
T_critical = np.linspace(1.5,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_k = 0
for t_critical in T_critical:
  for k in K:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(k,1,1,5,t_critical,2)
    time2,C_a2,C_o2,C_v2,C_so2,Temp2,X2,Y12,Y22,Y32,Y42 = simulate(k,1,1,0,t_critical,2)
    auc1 = AUC(Temp1,time1)
    auc2 = AUC(Temp2,time2)
    diff_auc = auc1 - auc2
    max_info[count_r,count_k] = diff_auc
    count_k = count_k + 1
  count_k = 0
  count_r = count_r + 1

plt.rcParams.update({
    'font.size': 16,       # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # X and Y labels
    'xtick.labelsize': 14, # X-axis ticks
    'ytick.labelsize': 14, # Y-axis ticks
    'legend.fontsize': 14  # Legend
})
X, Y = np.meshgrid(K,T_critical)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Difference of Area Under the Curve (AUC)')
contour = plt.contour(X, Y, max_info, colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)

plt.xlabel('K', fontsize = 16)
plt.ylabel('Critical Temperature', fontsize = 16)
plt.title('Difference of AUC Vs critical temperature and social learning rate',fontsize = 12)
plt.show()


#figure 3


K = np.linspace(0,0.2,50)
R_max = np.linspace(0,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_k = 0
for r_max in R_max:
  for k in K:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(k,1,1,r_max,3,2)
    time2,C_a2,C_o2,C_v2,C_so2,Temp2,X2,Y12,Y22,Y32,Y42 = simulate(k,1,1,0,3,2)
    condition = Temp1[1:] >= 1.1 * Temp2[1:]
    index = ((np.argmax(condition) + 1 )/100)+1800
    if not np.any(Temp1[1:] >= 1.1 * Temp2[1:]):
       index = None
       #index = 0
    max_info[count_r,count_k] = index
    count_k = count_k + 1
  count_k = 0
  count_r = count_r + 1

X, Y = np.meshgrid(K,R_max)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Time to Tipping Point')

plt.xlabel('K',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Time to Tipping Point Vs $R_{\mathrm{max}}$ and Social Learning Rate',fontsize = 12)
plt.show()

#Figure 4



K = np.linspace(0,0.1,50)
R_max = np.linspace(0,5,50)
max_info = np.zeros((50,50))
count_r = 0
count_k = 0
for r_max in R_max:
  for k in K:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(k,1,1,r_max,3,2)

    max_info[count_r,count_k] = np.max(Temp1)
    count_k = count_k + 1
  count_k = 0
  count_r = count_r + 1


# Assuming K, R_max, and max_info are defined elsewhere
X, Y = np.meshgrid(K, R_max)

plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Peak of Temperature')

# Add contour lines
contour = plt.contour(X, Y, max_info, colors='w', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)

plt.xlabel('K',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Peak of Temperature Vs $R_{\mathrm{max}}$ and Social Learning Rate',fontsize = 12)
plt.show()

#figure 6

def AUC(Y,X):
    dx = X[1] - X[0]
    auc = np.trapz(Y, dx=dx)

    return auc

Beta = np.linspace(0,2,70)
R_max = np.linspace(0,6,70)
max_info = np.zeros((70,70))
count_r = 0
count_beta = 0
for r_max in R_max:
  for beta in Beta:
    time1,C_a1,C_o1,C_v1,C_so1,Temp1,X1,Y11,Y21,Y31,Y41 = simulate(0.1,beta,3,r_max,2,2)
    time2,C_a2,C_o2,C_v2,C_so2,Temp2,X2,Y12,Y22,Y32,Y42 = simulate(0.1,beta,3,0,2,2)
    auc1 = AUC(Temp1,time1)
    auc2 = AUC(Temp2,time2)
    diff_auc = auc1 - auc2
    max_info[count_r,count_beta] = diff_auc
    count_beta = count_beta + 1
  count_beta = 0
  count_r = count_r + 1


X, Y = np.meshgrid(Beta,R_max)
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, max_info, shading='auto')
plt.colorbar(label='Difference in Area Under the Curve (AUC)')

contour = plt.contour(X, Y, max_info, colors='k', linewidths=0.5)
plt.clabel(contour, inline=True, fontsize=8)


plt.xlabel('Beta',fontsize = 16)
plt.ylabel(r'$R_{\mathrm{max}}$',fontsize = 16)
plt.title('Difference in Area Under the Curve Vs $R_{\mathrm{max}}$ and Net Cost of Mitigation',fontsize=12)
plt.show()



