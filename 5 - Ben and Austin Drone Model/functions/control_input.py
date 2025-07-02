import drone_config
FLAG = drone_config.FLAG
import numpy as np

def get_u(u_sim):
    if FLAG == '1':
        U = np.array(np.vstack([u_sim['PWM1'], u_sim['PWM2'], u_sim['PWM3'], u_sim['PWM4']]))
    elif FLAG == '2' or FLAG == '3':
        U = np.array(np.vstack([u_sim['u_x'], u_sim['u_y'], u_sim['u_z'], u_sim['u_psi']]))
    elif FLAG == '4':
        U = np.array(np.vstack([u_sim['u_thrust'], u_sim['u_phi'], u_sim['u_theta'], u_sim['u_psi']]))
    return U