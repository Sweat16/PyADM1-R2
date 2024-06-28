#%% import Packages
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
#%% ADM1_interp_INPUT
def ADM1_interp_INPUT(IN_Data, t):
    """
    Interpolate INPUT values

    Function corresponds to:
    q_in = np.interp(t, INPUT[:, 0], INPUT[:, 1], left=INPUT[0, 1], right=INPUT[-1, 1])
    x_in = np.interp(t, INPUT[:, 0], INPUT[:, 2:], left=INPUT[0, 2:], right=INPUT[-1, 2:])

    Parameters:
    INPUT (numpy.ndarray): INPUT array with time and INPUT values
    t (float): Time at which INPUT values need to be interpolated

    Returns:
    x_in (numpy.ndarray): Interpolated INPUT values
    q_in (float): Interpolated INPUT value
    """
    # Get number of INPUT sample points
    num_t = IN_Data.shape[0]
    # Get number model components
    num_INPUT = IN_Data.shape[1] - 2
    # Find the last (previous) INPUT value at t
    for i in range(num_t - 1, -1, -1):
        if t >= float(IN_Data.iloc[i, 0]):
            q_in = IN_Data.iloc[i, 1]
            x_in = IN_Data.iloc[i, 2:num_INPUT+2]
            break
        elif i == 0:
            q_in = 0
            x_in = np.zeros(num_INPUT)
    # Interpolate the INPUT values
    if t < IN_Data.iloc[0, 0]:
        q_in = IN_Data.iloc[0, 1]
        x_in = IN_Data.iloc[0, 2:]
    elif t > IN_Data.iloc[-1, 0]:
        q_in = IN_Data.iloc[-1, 1]
        x_in = IN_Data.iloc[-1, 2:]
    return x_in, q_in
#%% ADM1_interp_INPUT
# def ADM1_interp_INPUT(input, t):
#     # Get number of input sample points
#     num_t = input.shape[0]
#     # Get number model components
#     num_input = input.shape[1] - 2
#     # Find the last (previous) input value at t
#     for i in range(num_t-1, -1, -1):
#         if t >= input.iloc[i, 0]:
#             q_in = input.iloc[i, 1]
#             x_in = input.iloc[i, 2:num_input+2]
#             break
#         elif i == 0:
#             q_in = 0
#             x_in = np.zeros(num_input)
#     return x_in, q_in
#%% ADM1_R2_mass
def ADM1_R2_mass(t, x, s, IN_Data, parameter):
    
    # Interpolate INPUT parameters
    x_in, q_in = ADM1_interp_INPUT(IN_Data, t)
    
    # System parameters
    V_liq = s[0]
    V_gas = s[1]
    p_atm = s[2]
    
    # Model parameters
    K_H_ch4 = parameter.iloc[0,0]
    K_H_co2 = parameter.iloc[0,1]
    K_I_IN = parameter.iloc[0,2]
    K_I_nh3 = parameter.iloc[0,3]
    K_a_IN = parameter.iloc[0,4]
    K_a_ac = parameter.iloc[0,5]
    K_a_bu = parameter.iloc[0,6]
    K_a_co2 = parameter.iloc[0,7]
    K_a_pro = parameter.iloc[0,8]
    K_a_va = parameter.iloc[0,9]
    K_ac = parameter.iloc[0,10]
    K_bu = parameter.iloc[0,11]
    K_pro = parameter.iloc[0,12]
    K_va = parameter.iloc[0,13]
    K_w = parameter.iloc[0,14]
    R = parameter.iloc[0,15]
    T = parameter.iloc[0,16]
    k_AB_IN = parameter.iloc[0,17]
    k_AB_ac = parameter.iloc[0,18]
    k_AB_bu = parameter.iloc[0,19]
    k_AB_co2 = parameter.iloc[0,20]
    k_AB_pro = parameter.iloc[0,21]
    k_AB_va = parameter.iloc[0,22]
    k_La = parameter.iloc[0,23]
    k_ch = parameter.iloc[0,24]
    k_dec = parameter.iloc[0,25]
    k_li = parameter.iloc[0,26]
    k_m_ac = parameter.iloc[0,27]
    k_m_bu = parameter.iloc[0,28]
    k_m_pro = parameter.iloc[0,29]
    k_m_va = parameter.iloc[0,30]
    k_p = parameter.iloc[0,31]
    k_pr = parameter.iloc[0,32]
    pK_l_aa = parameter.iloc[0,33]
    pK_l_ac = parameter.iloc[0,34]
    pK_u_aa = parameter.iloc[0,35]
    pK_u_ac = parameter.iloc[0,36]
    p_h2o = parameter.iloc[0,37]
    
    # Define algebraic equations
    S_nh4_i = x[6] - x[23]
    S_co2 = x[5] - x[22]
    phi = x[16] + S_nh4_i/17 - x[22]/44 - x[21]/60 - x[20]/74 - x[19]/88 - x[18]/102 - x[17]
    S_H = -phi*0.5 + 0.5*np.sqrt(phi*phi+4*K_w)
    pH = -np.log10(S_H)
    p_ch4 = x[24]*R*T/16
    p_co2 = x[25]*R*T/44
    p_gas = p_ch4 + p_co2 + p_h2o
    q_gas = k_p*(p_gas-p_atm)*p_gas/p_atm
    
    # Define inhibition functions
    
    inhibition = np.zeros(4)
    inhibition[0] = x[6] / (x[6] + K_I_IN)
    inhibition[1] = 10**(-(3/(pK_u_aa - pK_l_aa))*(pK_l_aa+pK_u_aa)/2) / (S_H**(3/(pK_u_aa - pK_l_aa)) + 10**(-(3/(pK_u_aa - pK_l_aa))*(pK_l_aa+pK_u_aa)/2))
    inhibition[2] = 10**(-(3/(pK_u_ac - pK_l_ac))*(pK_l_ac+pK_u_ac)/2) / (S_H**(3/(pK_u_ac - pK_l_ac)) + 10**(-(3/(pK_u_ac - pK_l_ac))*(pK_l_ac+pK_u_ac)/2))
    inhibition[3] = K_I_nh3 / (K_I_nh3 + x[23])
    
    # Define rate equations
    
    rate = np.zeros(20)
    rate[0] = k_ch * x[8]
    rate[1] = k_pr * x[9]
    rate[2] = k_li * x[10]
    rate[3] = k_m_va * x[0] / (K_va + x[0]) * x[12] * x[0] / (x[1] + x[0] + 1e-8) * inhibition[0] * inhibition[1]
    rate[4] = k_m_bu * x[1] / (K_bu + x[1]) * x[13] * x[1] / (x[1] + x[0] + 1e-8) * inhibition[0] * inhibition[1]
    rate[5] = k_m_pro * x[2] / (K_pro + x[2]) * x[14] * inhibition[0] * inhibition[1]
    rate[6] = k_m_ac * x[3] / (K_ac + x[3]) * x[15] * inhibition[0] * inhibition[2] * inhibition[3]
    rate[7] = k_dec * x[11]
    rate[8] = k_dec * x[12]
    rate[9] = k_dec * x[13]
    rate[10] = k_dec * x[14]
    rate[11] = k_dec * x[15]
    rate[12] = k_AB_va * (x[18] * (K_a_va + S_H) - K_a_va * x[0])
    rate[13] = k_AB_bu * (x[19] * (K_a_bu + S_H) - K_a_bu * x[1])
    rate[14] = k_AB_pro * (x[20] * (K_a_pro + S_H) - K_a_pro * x[2])
    rate[15] = k_AB_ac * (x[21] * (K_a_ac + S_H) - K_a_ac * x[3])
    rate[16] = k_AB_co2 * (x[22] * (K_a_co2 + S_H) - K_a_co2 * x[5])
    rate[17] = k_AB_IN * (x[23] * (K_a_IN + S_H) - K_a_IN * x[6])
    rate[18] = k_La * (x[4] - 16 * (K_H_ch4 * p_ch4))
    rate[19] = k_La * (S_co2 - 44 * (K_H_co2 * p_co2))

    # Define process equations
    # rate = x
    process = np.zeros((26, 1))
    process[0] = 0.15883 * rate[1] - 10.1452 * rate[3]
    process[1] = 0.076292 * rate[0] + 0.20135 * rate[1] + 0.0092572 * rate[2] - 10.9274 * rate[4]
    process[2] = 0.19032 * rate[0] + 0.046509 * rate[1] + 0.023094 * rate[2] + 6.9368 * rate[3] - 14.4449 * rate[5]
    process[3] = 0.41 * rate[0] + 0.52784 * rate[1] + 1.7353 * rate[2] + 5.6494 * rate[3] + 14.0023 * rate[4] + 11.2133 * rate[5] - 26.5447 * rate[6]
    process[4] = 0.047712 * rate[0] + 0.019882 * rate[1] + 0.18719 * rate[2] + 0.68644 * rate[3] + 0.87904 * rate[4] + 2.1242 * rate[5] + 6.7367 * rate[6] - rate[18]
    process[5] = 0.22553 * rate[0] + 0.18347 * rate[1] - 0.64703 * rate[2] - 2.6138 * rate[3] - 3.0468 * rate[4] + 1.5366 * rate[5] + 18.4808 * rate[6] - rate[19]
    process[6] = -0.013897 * rate[0] + 0.18131 * rate[1] - 0.024038 * rate[2] - 0.15056 * rate[3] - 0.15056 * rate[4] - 0.15056 * rate[5] - 0.15056 * rate[6]
    process[7] = -0.028264 * rate[0] - 0.40923 * rate[1] - 0.44342 * rate[2] - 1.363 * rate[3] - 1.7566 * rate[4] - 1.2786 * rate[5] + 0.4778 * rate[6]
    process[8] = -rate[0] + 0.18 * rate[7] + 0.18 * rate[8] + 0.18 * rate[9] + 0.18 * rate[10] + 0.18 * rate[11]
    process[9] = -rate[1] + 0.77 * rate[7] + 0.77 * rate[8] + 0.77 * rate[8] + 0.77 * rate[9] + 0.77 * rate[11]
    process[10] = - rate[2] + 0.05 * rate[7] + 0.05 * rate[7] + 0.05 * rate[8] + 0.05 * rate[9] + 0.05 * rate[10]
    process[11] = 0.092305 * rate[0] + 0.090036 * rate[1] + 0.15966 * rate[2] - rate[7]
    process[12] = rate[3] - rate[8]
    process[13] = rate[4] - rate[9]
    process[14] = rate[5] - rate[10]
    process[15] = rate[6] - rate[11]
    process[16] = 0
    process[17] = 0
    process[18] = - rate[12]
    process[19] = - rate[13]
    process[20] = - rate[14]
    process[21] = - rate[15]
    process[22] = - rate[16]
    process[23] = - rate[17]
    process[24] = (V_liq / V_gas) * rate[18]
    process[25] = (V_liq / V_gas) * rate[19]
  
    # Define differential equations
    dx = np.zeros((26,))
    dx[0] = q_in * (x_in[0] - x[0]) / V_liq + process[0]
    dx[1] = q_in * (x_in[1] - x[1]) / V_liq + process[1]
    dx[2] = q_in * (x_in[2] - x[2]) / V_liq + process[2]
    dx[3] = q_in * (x_in[3] - x[3]) / V_liq + process[3]
    dx[4] = q_in * (x_in[4] - x[4]) / V_liq + process[4]
    dx[5] = q_in * (x_in[5] - x[5]) / V_liq + process[5]
    dx[6] = q_in * (x_in[6] - x[6]) / V_liq + process[6]
    dx[7] = q_in * (x_in[7] - x[7]) / V_liq + process[7]
    dx[8] = q_in * (x_in[8] - x[8]) / V_liq + process[8]
    dx[9] = q_in * (x_in[9] - x[9]) / V_liq + process[9]
    dx[10] = q_in*(x_in[10] - x[10])/V_liq + process[10]
    dx[11] = q_in*(x_in[11] - x[11])/V_liq + process[11]
    dx[12] = q_in*(x_in[12] - x[12])/V_liq + process[12]
    dx[13] = q_in*(x_in[13] - x[13])/V_liq + process[13]
    dx[14] = q_in*(x_in[14] - x[14])/V_liq + process[14]
    dx[15] = q_in*(x_in[15] - x[15])/V_liq + process[15]
    dx[16] = q_in*(x_in[16] - x[16])/V_liq + process[16]
    dx[17] = q_in*(x_in[17] - x[17])/V_liq + process[17]
    dx[18] = process[18]
    dx[19] = process[19]
    dx[20] = process[20]
    dx[21] = process[21]
    dx[22] = process[22]
    dx[23] = process[23]
    dx[24] = -x[24] * q_gas / V_gas + process[24]
    dx[25] = -x[25] * q_gas / V_gas + process[25]
    
    return dx
#%% ADM1_R2_mass_output

def ADM1_R2_mass_output(t, x, system, parameter):
    
    # System parameters
    V_liq = system[0]
    V_gas = system[1]
    p_atm = system[2]

    # Model parameters
    K_H_ch4 = parameter.iloc[0,0]
    K_H_co2 = parameter.iloc[0,1]
    K_I_IN = parameter.iloc[0,2]
    K_I_nh3 = parameter.iloc[0,3]
    K_a_IN = parameter.iloc[0,4]
    K_a_ac = parameter.iloc[0,5]
    K_a_bu = parameter.iloc[0,6]
    K_a_co2 = parameter.iloc[0,7]
    K_a_pro = parameter.iloc[0,8]
    K_a_va = parameter.iloc[0,9]
    K_ac = parameter.iloc[0,10]
    K_bu = parameter.iloc[0,11]
    K_pro = parameter.iloc[0,12]
    K_va = parameter.iloc[0,13]
    K_w = parameter.iloc[0,14]
    R = parameter.iloc[0,15]
    T = parameter.iloc[0,16]
    k_AB_IN = parameter.iloc[0,17]
    k_AB_ac = parameter.iloc[0,18]
    k_AB_bu = parameter.iloc[0,19]
    k_AB_co2 = parameter.iloc[0,20]
    k_AB_pro = parameter.iloc[0,21]
    k_AB_va = parameter.iloc[0,22]
    k_La = parameter.iloc[0,23]
    k_ch = parameter.iloc[0,24]
    k_dec = parameter.iloc[0,25]
    k_li = parameter.iloc[0,26]
    k_m_ac = parameter.iloc[0,27]
    k_m_bu = parameter.iloc[0,28]
    k_m_pro = parameter.iloc[0,29]
    k_m_va = parameter.iloc[0,30]
    k_p = parameter.iloc[0,31]
    k_pr = parameter.iloc[0,32]
    pK_l_aa = parameter.iloc[0,33]
    pK_l_ac = parameter.iloc[0,34]
    pK_u_aa = parameter.iloc[0,35]
    pK_u_ac = parameter.iloc[0,36]
    p_h2o = parameter.iloc[0,37]

    # Define algebraic equations
    S_nh4_i = x[6] - x[23]
    S_co2 = x[5] - x[22]
    phi = x[16] + S_nh4_i / 17 - x[22] / 44 - x[21] / 60 - x[20] / 74 - x[19] / 88 - x[18] / 102 - x[17]
    S_H = -phi * 0.5 + 0.5 * np.sqrt(phi * phi + 4 * K_w)
    pH = -np.log10(S_H)
    p_ch4 = x[24] * R * T / 16
    p_co2 = x[25] * R * T / 44
    p_gas = p_ch4 + p_co2 + p_h2o
    q_gas = k_p * (p_gas - p_atm) * p_gas / p_atm

    # Define output (states)
    y = np.zeros(35)
    for i in range(25):
        y[i] = x[i]

    # Define output (algebraic components)
    y[26] = S_nh4_i
    y[27] = S_co2
    y[28] = phi
    y[29] = S_H
    y[30] = pH
    y[31] = p_ch4
    y[32] = p_co2
    y[33] = p_gas
    y[34] = q_gas
    
    return (t, y)
#%% reading INPUTs
model = 'ADM1_R2'

time_range = [0, 10]
# time_range = pd.DataFrame({'time_range': time_range})

# initial = pd.read_excel("ADM_R2_initial.xlsx")
initial = np.array([0.0114223117132854,0.0119548880579878,0.00914883050228461,0.0492236924718528,0.0117038560549238,4.94195160138739,0.957774774658787,957.027354942148,2.96226304085042,0.950970720516194,0.412307183267109,1.53441898238245,0.0551231732279092,0.162337407079022,0.206029738476229,0.526534356229475,0.05625,0.0075,0.0113837287995512,0.0119180483969296,0.00911647579579492,0.0490915273041594,4.51417296384279,0.0222025050985267,0.359367962677795,0.657500376975363])

In_Data = pd.read_excel('Data.xlsx', sheet_name='Input_10')

# parameter = [0.0011, 0.025, 0.0017, 0.0306, 1.11E-09, 1.74E-05, 1.51E-05, 4.94E-07, 1.32E-05, 1.38E-05, 0.14, 0.1101, 0.07, 0.1, 2.08E-14, 0.08315, 311, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 200, 0.25, 0.02, 0.1, 0.4, 1.2, 0.52, 1.2, 50000, 0.2, 4, 6, 5.5, 7, 0.0657]
parameter = pd.read_excel('Data.xlsx', sheet_name='Parameter')

system = [100, 10, 1.0313]
# system = pd.DataFrame({'system': system})

# solvermethod = 'DOP853'
#%% Function
# Solve ODE of mass-based ADM1-R2

# Solve the ODE system
def f(t,x):
    return ADM1_R2_mass(t, x, system, In_Data, parameter)

#%% ODE
start_time = time.time()
t_eval = np.linspace(0, 10, 241)
sol = solve_ivp(f, time_range, initial, method='LSODA', t_eval=t_eval)
#%% Output
num_t = sol.y.shape[1]
t = np.zeros(num_t)
y = np.zeros((num_t, 35))
for i in range(num_t):
    t[i], y[i,:] = ADM1_R2_mass_output(sol.t[i], sol.y[:,i], system, parameter)
end_time = time.time()
runtime = end_time - start_time
#%%
# sol = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
# sol.set_initial_value(initial, time_range[0])
# t = [time_range[0]]
# y = initial
# u = 0
# while sol.successful() and sol.t < time_range[1]:
#     u =+ u
#     sol.integrate(time_range[1], step=True)
#     # t.append(sol.t)
#     # y.append(sol.y)
#     np.append(t, sol.t)
#     np.append(y, sol.y)
#     print(u)
# t = np.linspace(0, 100)
# sol = odeint(f, initial, time_range)
# sol = solve_ivp(f, time_range, initial, method='RK23')
# num_t = sol.x.shape[1]
# for i in range(num_t):
#     t[i],y[i,:] = ADM1_R2_mass_output(slo.t,sol.x[:,i],system,parameter)

# sol = solve_ivp(ADM1_R2_mass(t, x, system, INPUT, parameter), 
#                 time_range, 
#                 initial, 
#                 method='LSODA')
#%% ODE(2)
# def myode15s(f, tspan, y0, options=None):
#     # Solves the stiff ODE y' = f(t,y) using ode.
#     # f is a function that takes (t, y) as input and returns y'.
#     # tspan is a two-element tuple (t0, tf) specifying the start and end times.
#     # y0 is a column vector of initial conditions.
#     # options is an optional dict of integration options.
#     # Returns a tuple (t, y) where t is a 1D array of time values at which the solution was computed,
#     # and y is a 2D array whose columns are the solution values at the corresponding times.
    
#     # Set default options if none are specified.
#     if options is None:
#         options = {}
    
#     # Define the ODE function that takes (t, y) as input and returns y'.
#     def f_ode(t, y):
#         return f(t, y).reshape(y.shape)
    
#     # Create an instance of the ode class.
#     solver = ode(f_ode)
    
#     # Set the integration options.
#     for key, value in options.items():
#         solver.set_integrator_option(key, value)
    
#     # Set the initial condition and integration range.
#     solver.set_initial_value(y0, tspan[0])
    
#     # Integrate the ODE.
#     t = [tspan[0]]
#     y = [y0]
#     while solver.successful() and solver.t < tspan[1]:
#         t_next = solver.t + solver.integrate(tspan[1])
#         y_next = solver.y
#         t.append(t_next)
#         y.append(y_next)
    
#     # Convert the solution to arrays.
#     t = np.array(t)
#     y = np.array(y).T
    
#     return t, y
#%% ODE(3)
# def myode15s(f, tspan, y0, options=None):
#     # Solves the stiff ODE y' = f(t,y) using ode.
#     # f is a function that takes (t, y) as input and returns y'.
#     # tspan is a two-element tuple (t0, tf) specifying the start and end times.
#     # y0 is a 2D array of initial conditions, where each column corresponds to a different initial condition.
#     # options is an optional dict of integration options.
#     # Returns a tuple (t, y) where t is a 1D array of time values at which the solution was computed,
#     # and y is a 3D array whose dimensions are (n, m, l) where n is the number of time values,
#     # m is the number of initial conditions, and l is the dimension of the solution vector at each time.
    
#     # Set default options if none are specified.
#     if options is None:
#         options = {}
    
#     # Define the ODE function that takes (t, y) as input and returns y'.
#     def f_ode(t, y):
#         return f(t, y).reshape(y.shape)
    
#     # Create an instance of the ode class.
#     solver = ode(f_ode)
    
#     # Set the integration options.
#     for key, value in options.items():
#         solver.set_integrator_option(key, value)
    
#     # Set the initial conditions and integration range.
#     y0 = np.array(y0)
#     solver.set_initial_value(y0, tspan[0])
    
#     # Integrate the ODE for each initial condition.
#     t = [tspan[0]]
#     y = [y0.T]
#     for i in range(1, y0.shape[0]):
#         solver.set_initial_value(y0, tspan[0])
#         y_i = [solver.y]
#         while solver.successful() and solver.t < tspan[1]:
#             t_next = solver.t + solver.integrate(tspan[1])
#             y_next = solver.y
#             y_i.append(y_next)
#             t.append(t_next)
#         y.append(np.array(y_i))
    
#     # Convert the solution to arrays.
#     t = np.array(t)
#     # y = np.array(y).transpose((1, 0, 2))
#     y = np.array(y)
#     return t, y
#%%
# t, y = myode15s(f, time_range, initial)
# t = t[1][0][:]

# # Calculate model output
# num_t = 199
# #t = sol.t
# y = np.zeros((num_t, len(y)))

# for i in range(num_t):
#     t[i], y[i,:] = ADM1_R2_mass_output(t[i], y[:,i], system, parameter)
            
# # Set model output
# output = []
# output[0:num_t,:] = np.column_stack((t, y))

#%% Plot model output
# =============================================================================
# plt.plot(output.iloc[:,0], output.iloc[:,1:])
# plt.plotbrowser(True)
# 
# # Set legend
# l = plt.legend(output.columns[1:], loc='upper right')
# l.set_visible(False)
# l.set_bbox_to_anchor((1.05, 1))
# 
# # Set title
# t = plt.title(f'Simulation results of the {model}')
# t.set_visible(False)
# 
# # Set axis labels
# plt.xlabel('Time [d]')
# plt.ylabel('Model output')
# 
# # Clear variables
# del i, num_t, t, y, l, time_range
# =============================================================================
