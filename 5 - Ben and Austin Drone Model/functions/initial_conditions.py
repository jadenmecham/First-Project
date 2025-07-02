import drone_config
FLAG = drone_config.FLAG
import numpy as np

def init_conds(X0_sim):
    x0_x_offset = 1.0
    x0_y_offset = 1.0
    x0_z_offset = 1.0
    x0_v_x_offset = 1.0
    x0_v_y_offset = 1.0
    x0_v_z_offset = 1.0
    x0_psi_offset = 1.0
    x0_theta_offset = 1.0
    x0_phi_offset = 1.0
    x0_omega_x_offset = 1.0
    x0_omega_y_offset = 1.0
    x0_omega_z_offset = 1.0
    x0_wx_offset = 1.0
    x0_wy_offset = 1.0

    if FLAG == '1':
        Xo= np.atleast_2d(np.vstack((
        X0_sim[0] + x0_x_offset,X0_sim[1] + x0_y_offset,X0_sim[2] + x0_z_offset,
        X0_sim[3] + x0_v_x_offset,X0_sim[4] + x0_v_y_offset,X0_sim[5] + x0_v_z_offset,
        X0_sim[6] + x0_phi_offset,X0_sim[7] + x0_theta_offset,X0_sim[8] + x0_psi_offset,
        X0_sim[9] + x0_omega_x_offset,X0_sim[10] + x0_omega_y_offset,X0_sim[11] + x0_omega_z_offset,
        X0_sim[12] + x0_wx_offset,X0_sim[13] + x0_wy_offset
        )))
    elif FLAG == '2' or FLAG == '3':
        Xo= np.atleast_2d(np.vstack((
            X0_sim[0] + x0_x_offset, X0_sim[1] + x0_y_offset, X0_sim[2] + x0_z_offset,
            X0_sim[3] + x0_v_x_offset, X0_sim[4] + x0_v_y_offset, X0_sim[5] + x0_v_z_offset,
            X0_sim[6] + x0_psi_offset,
            X0_sim[7] + x0_wx_offset, X0_sim[8] + x0_wy_offset
        )))
    elif FLAG == '4':
        Xo= np.atleast_2d(np.vstack((
            X0_sim[0] + x0_x_offset, X0_sim[1] + x0_y_offset, X0_sim[2] + x0_z_offset,
            X0_sim[3] + x0_v_x_offset, X0_sim[4] + x0_v_y_offset, X0_sim[5] + x0_v_z_offset,
            X0_sim[6] + x0_phi_offset, X0_sim[7] + x0_theta_offset, X0_sim[8] + x0_psi_offset,
            X0_sim[9] + x0_wx_offset, X0_sim[10] + x0_wy_offset
        )))

    return x0_x_offset, x0_y_offset, x0_z_offset, x0_v_x_offset, x0_v_y_offset, x0_v_z_offset, \
           x0_psi_offset, x0_theta_offset, x0_phi_offset, x0_omega_x_offset, x0_omega_y_offset, \
           x0_omega_z_offset, x0_wx_offset, x0_wy_offset, Xo

def covariance_q():
    if FLAG == '1':
        Q = np.diag(np.array([
        1e-20, 1e-20, 1e-20,  # x, y, z
        1e-20, 1e-20, 1e-20,  # v_x, v_y, v_z
        1e-20, 1e-20, 1e-20,  # phi, theta, psi
        1e-20, 1e-20, 1e-20,  # omega_x, omega_y, omega_z
        1e-20, 1e-20        # wx, wy
        ]))
    elif FLAG == '2' or FLAG == '3':
        Q = np.diag(np.array([
            1e-2, 1e-2, 1e-2,  # x, y, z
            1e-2, 1e-2, 1e-2,  # v_x, v_y, v_z
            1e-2,                # psi
            1e-2, 1e-2          # wx, wy
        ]))
    elif FLAG == '4':
        Q = np.diag(np.array([
            1e-2, 1e-2, 1e-2,  # x, y, z
            1e-2, 1e-2, 1e-2,  # v_x, v_y, v_z
            1e-2, 1e-2, 1e-2,  # phi, theta, psi
            1e-2, 1e-2          # wx, wy
        ]))
    return Q

def estimate_covariance_p(x0_x_offset, x0_y_offset, x0_z_offset, x0_v_x_offset, x0_v_y_offset, x0_v_z_offset, \
x0_psi_offset, x0_theta_offset, x0_phi_offset, x0_omega_x_offset, x0_omega_y_offset, \
x0_omega_z_offset, x0_wx_offset, x0_wy_offset):
    if FLAG == '1':
        # Initial state covariance matrix
        P0 = np.diag(np.array([x0_x_offset, x0_y_offset, x0_z_offset,
                            x0_v_x_offset, x0_v_y_offset, x0_v_z_offset,
                            x0_phi_offset, x0_theta_offset, x0_psi_offset,
                            x0_omega_x_offset, x0_omega_y_offset, x0_omega_z_offset,
                            x0_wx_offset, x0_wy_offset]) ** 2)
    elif FLAG == '2' or FLAG == '3':
        P0 = np.diag(np.array([x0_x_offset, x0_y_offset, x0_z_offset,
                            x0_v_x_offset, x0_v_y_offset, x0_v_z_offset,
                            x0_psi_offset,
                            x0_wx_offset, x0_wy_offset]) ** 2)
    elif FLAG == '4':
        P0 = np.diag(np.array([x0_x_offset, x0_y_offset, x0_z_offset,
                            x0_v_x_offset, x0_v_y_offset, x0_v_z_offset,
                            x0_phi_offset, x0_theta_offset, x0_psi_offset,
                            x0_wx_offset, x0_wy_offset]) ** 2)   
    return P0