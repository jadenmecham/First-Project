import numpy as np
import drone_config
FLAG = drone_config.FLAG

def noise(y_sim):
    # creat new dictionary that is a copy of y_sim called y_sim_noise
    y_sim_noise = y_sim.copy()
    # add noise to y_sim_noise
    # mocap postion noise in standard deviations
    if FLAG == '1':
        pose_noise_std = 0.1
        y_sim_noise['x'] = y_sim['x'] + np.random.normal(0, pose_noise_std, size=y_sim['x'].shape)
        y_sim_noise['y'] = y_sim['y'] + np.random.normal(0, pose_noise_std, size=y_sim['y'].shape)
        y_sim_noise['z'] = y_sim['z'] + np.random.normal(0, pose_noise_std, size=y_sim['z'].shape)
        velocity_noise_std = 0.1
        y_sim_noise['v_x'] = y_sim['v_x'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_x'].shape)
        y_sim_noise['v_y'] = y_sim['v_y'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_y'].shape)
        y_sim_noise['v_z'] = y_sim['v_z'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_z'].shape)
        attitude_noise_std = 0.1
        y_sim_noise['phi'] = y_sim['phi'] + np.random.normal(0, attitude_noise_std, size=y_sim['phi'].shape)
        y_sim_noise['theta'] = y_sim['theta'] + np.random.normal(0, attitude_noise_std, size=y_sim['theta'].shape)
        y_sim_noise['psi'] = y_sim['psi'] + np.random.normal(0, attitude_noise_std, size=y_sim['psi'].shape)
        omega_noise_std = 0.1
        y_sim_noise['omega_x'] = y_sim['omega_x'] + np.random.normal(0, omega_noise_std, size=y_sim['omega_x'].shape)
        y_sim_noise['omega_y'] = y_sim['omega_y'] + np.random.normal(0, omega_noise_std, size=y_sim['omega_y'].shape)
        y_sim_noise['omega_z'] = y_sim['omega_z'] + np.random.normal(0, omega_noise_std, size=y_sim['omega_z'].shape)
        acc_noise_std = 0.1
        y_sim_noise['Axt'] = y_sim['Axt'] + np.random.normal(0, acc_noise_std, size=y_sim['Axt'].shape)
        y_sim_noise['Ayt'] = y_sim['Ayt'] + np.random.normal(0, acc_noise_std, size=y_sim['Ayt'].shape)
        y_sim_noise['Az'] = y_sim['Az'] + np.random.normal(0, acc_noise_std, size=y_sim['Az'].shape)
        wind_noise_std = 0.1
        y_sim_noise['Wax'] = y_sim['Wax'] + np.random.normal(0, wind_noise_std, size=y_sim['Wax'].shape)
        y_sim_noise['Way'] = y_sim['Way'] + np.random.normal(0, wind_noise_std, size=y_sim['Way'].shape)
        of_noise_std = 0.1
        y_sim_noise['rx'] = y_sim['rx'] + np.random.normal(0, of_noise_std, size=y_sim['rx'].shape)
        y_sim_noise['ry'] = y_sim['ry'] + np.random.normal(0, of_noise_std, size=y_sim['ry'].shape)
    elif FLAG == '2' or FLAG == '3':
        pose_noise_std = 0.1
        y_sim_noise['x'] = y_sim['x'] + np.random.normal(0, pose_noise_std, size=y_sim['x'].shape)
        y_sim_noise['y'] = y_sim['y'] + np.random.normal(0, pose_noise_std, size=y_sim['y'].shape)
        y_sim_noise['z'] = y_sim['z'] + np.random.normal(0, pose_noise_std, size=y_sim['z'].shape)
        velocity_noise_std = 0.1
        y_sim_noise['v_x'] = y_sim['v_x'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_x'].shape)
        y_sim_noise['v_y'] = y_sim['v_y'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_y'].shape)
        y_sim_noise['v_z'] = y_sim['v_z'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_z'].shape)
        attitude_noise_std = 0.1
        y_sim_noise['psi'] = y_sim['psi'] + np.random.normal(0, attitude_noise_std, size=y_sim['psi'].shape)
        acc_noise_std = 0.1
        y_sim_noise['Ax'] = y_sim['Ax'] + np.random.normal(0, acc_noise_std, size=y_sim['Ax'].shape)
        y_sim_noise['Ay'] = y_sim['Ay'] + np.random.normal(0, acc_noise_std, size=y_sim['Ay'].shape)
        y_sim_noise['Az'] = y_sim['Az'] + np.random.normal(0, acc_noise_std, size=y_sim['Az'].shape)
        wind_noise_std = 0.1
        y_sim_noise['Wax'] = y_sim['Wax'] + np.random.normal(0, wind_noise_std, size=y_sim['Wax'].shape)
        y_sim_noise['Way'] = y_sim['Way'] + np.random.normal(0, wind_noise_std, size=y_sim['Way'].shape)
        of_noise_std = 0.1
        y_sim_noise['rx'] = y_sim['rx'] + np.random.normal(0, of_noise_std, size=y_sim['rx'].shape)
        y_sim_noise['ry'] = y_sim['ry'] + np.random.normal(0, of_noise_std, size=y_sim['ry'].shape)
    elif FLAG == '4':
        pose_noise_std = 0.1
        y_sim_noise['x'] = y_sim['x'] + np.random.normal(0, pose_noise_std, size=y_sim['x'].shape)
        y_sim_noise['y'] = y_sim['y'] + np.random.normal(0, pose_noise_std, size=y_sim['y'].shape)
        y_sim_noise['z'] = y_sim['z'] + np.random.normal(0, pose_noise_std, size=y_sim['z'].shape)
        velocity_noise_std = 0.1
        y_sim_noise['v_x'] = y_sim['v_x'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_x'].shape)
        y_sim_noise['v_y'] = y_sim['v_y'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_y'].shape)
        y_sim_noise['v_z'] = y_sim['v_z'] + np.random.normal(0, velocity_noise_std, size=y_sim['v_z'].shape)
        attitude_noise_std = 0.1
        y_sim_noise['phi'] = y_sim['phi'] + np.random.normal(0, attitude_noise_std, size=y_sim['phi'].shape)
        y_sim_noise['theta'] = y_sim['theta'] + np.random.normal(0, attitude_noise_std, size=y_sim['theta'].shape)
        y_sim_noise['psi'] = y_sim['psi'] + np.random.normal(0, attitude_noise_std, size=y_sim['psi'].shape)
        acc_noise_std = 0.1
        y_sim_noise['Ax'] = y_sim['Ax'] + np.random.normal(0, acc_noise_std, size=y_sim['Ax'].shape)
        y_sim_noise['Ay'] = y_sim['Ay'] + np.random.normal(0, acc_noise_std, size=y_sim['Ay'].shape)
        y_sim_noise['Az'] = y_sim['Az'] + np.random.normal(0, acc_noise_std, size=y_sim['Az'].shape)
        wind_noise_std = 0.1
        y_sim_noise['Wax'] = y_sim['Wax'] + np.random.normal(0, wind_noise_std, size=y_sim['Wax'].shape)
        y_sim_noise['Way'] = y_sim['Way'] + np.random.normal(0, wind_noise_std, size=y_sim['Way'].shape)
        of_noise_std = 0.1
        y_sim_noise['rx'] = y_sim['rx'] + np.random.normal(0, of_noise_std, size=y_sim['rx'].shape)
        y_sim_noise['ry'] = y_sim['ry'] + np.random.normal(0, of_noise_std, size=y_sim['ry'].shape)
    
    return y_sim_noise