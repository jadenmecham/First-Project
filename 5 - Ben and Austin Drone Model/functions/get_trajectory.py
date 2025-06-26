from trajectory_gen import TrajectoryGenerator
import drone_config
FLAG = drone_config.FLAG
from drone_model_with_PWM import DroneSimulator
import time
import pandas as pd
import numpy as np

def gen_traj(motif_type, path):
    # generates and plots the trajectory that the drone will follow
    g = 9.81  # gravity [m/s^2]
    m = 0.086  # mass [kg]
    M = 2.529  # mass [kg]
    Mm = 4 * m + M  # total mass [kg]
    L = 0.2032  # length [m]
    R = 0.1778  # average body radius [m]
    I_x = .491 # moment of inertia about x
    I_y = .387 # moment of inertia about y
    I_z = .667 # moment of inertia about z
    b = 1.34 # thrust coefficient
    d = 1.0 # drag constant
    C = 0.1 # drag coefficient from ground speed plus air speed

    PARAMS = {
            'g': g,
            'm': m,
            'M': M,
            'Mm': Mm,
            'L': L,
            'R': R,
            'I_x': I_x,
            'I_y': I_y,
            'I_z': I_z,
            'b': b,
            'd': d,
            'C': C
                    }
    EVERYTHING= ['constant_velocity' , 'constant_velocity_yaw', 'constant_accel', 'sine','sine_constant_heading']

    traj_gen = TrajectoryGenerator()

    # motif_type= 'sine'  # 'sine', 'constant_accel', 'constant_velocity', 'constant_velocity_yaw', or 'EVERYTHING
    if motif_type == 'EVERYTHING':
        # run all motifs except 'all' save in dictionary for each output ie tsim['constant_velocity'] and tsim['sine_wave'] ect
        tsim = {}
        dt = {}
        v_x = {}
        v_y = {}
        psi = {}
        psi_global = {}
        x_dot = {}
        y_dot = {}
        z = {}
        X0_sim = {}
        for motif in EVERYTHING:
            print(f'Generating motif: {motif}')
            tsim[motif], dt[motif], v_x[motif], v_y[motif], psi[motif], psi_global[motif], x_dot[motif], y_dot[motif], z[motif], X0_sim[motif] = traj_gen.motifs(motif=motif)
            traj_gen.plot_trajectory(motif=motif)
    else:
        print(f'Generating motif: {motif_type}')
        tsim, dt, v_x, v_y, psi, psi_global, x_dot, y_dot, z, X0_sim = traj_gen.motifs(motif=motif_type)
        traj_gen.plot_trajectory()
    
    if FLAG == '1':
        pass
    elif FLAG == '2' or FLAG == '3':
        X0_sim = np.delete(X0_sim, [6, 7, 9, 10, 11], axis=0)
    elif FLAG == '4':
        X0_sim = np.delete(X0_sim, [9, 10, 11], axis=0)
    X0_sim
    print(X0_sim[-2], X0_sim[-1])

    if motif_type == 'EVERYTHING':
        print('Running simulation for all motifs...')
        simulator = {}
        t_sim = {}
        x_sim = {}
        u_sim = {}
        y_sim = {}
        for motif in EVERYTHING:
            print(f'Running simulation for motif: {motif}')
            # Create simulator
            simulator[motif] = DroneSimulator(dt=dt[motif], mpc_horizon=50, r_u=1e-4, control_mode='velocity_body_level',params=PARAMS)
            # Update the setpoints
            simulator[motif].update_setpoint(v_x=v_x[motif], v_y=v_y[motif], psi=psi[motif], z=z[motif], wx=np.ones_like(z[motif])*X0_sim[motif][-2], wy=np.ones_like(z[motif])*X0_sim[motif][-1])
            # Run simulation
            st = time.time()
            t_sim[motif], x_sim[motif], u_sim[motif], y_sim[motif] = simulator[motif].simulate(x0=X0_sim[motif], mpc=True, return_full_output=True)
            et = time.time()
            print('elapsed time:', et-st)
    else:
        print(f'Running simulation for just motif: {motif_type}')
        # Create simulator
        simulator = DroneSimulator(dt=dt, mpc_horizon=50, r_u=1e-4, control_mode='velocity_body_level',params=PARAMS)
        # Update the setpoints
        simulator.update_setpoint(v_x=v_x, v_y=v_y, psi=psi, z=z, wx=np.ones_like(z)*X0_sim[-2], wy=np.ones_like(z)*X0_sim[-1])
        # Run simulation
        st = time.time()
        # t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=X0_sim, u=U_real, mpc=False, return_full_output=True)
        t_sim, x_sim, u_sim, y_sim = simulator.simulate(x0=X0_sim, mpc=True, return_full_output=True)
        et = time.time()
        print('elapsed time:', et-st)
    
    # dictionary to csv 
    y_sim.keys()
    y_sim_df = pd.DataFrame(y_sim)
    # add 'output' to the names of the columns
    y_sim_df.columns = [f'output_{col}' for col in y_sim_df.columns]
    # y_sim_df.to_csv(f'/home/austinlopez/Drone_AFRL/simulation/simulated_outputs_{motif_type}.csv', index=False)
    x_sim_df = pd.DataFrame(x_sim)
    # add 'state' to the names of the columns
    x_sim_df.columns = [f'state_{col}' for col in x_sim_df.columns]
    # x_sim_df.to_csv(f'/home/austinlopez/Drone_AFRL/simulation/simulated_states_{motif_type}.csv', index=False)
    u_sim_df = pd.DataFrame(u_sim)
    # add 'control' to the names of the columns
    u_sim_df.columns = [f'control_{col}' for col in u_sim_df.columns]
    # u_sim_df.to_csv(f'/home/austinlopez/Drone_AFRL/simulation/simulated_inputs_{motif_type}.csv', index=False)
    t_sim_df = pd.DataFrame(t_sim, columns=['time'])

    #combine all dataframes into one
    simulated_data_df = pd.concat([t_sim_df, x_sim_df, u_sim_df, y_sim_df], axis=1)

    # print( simulated_data_df.head() )
    # save the dataframe to a csv file
    simulated_data_df.to_csv(path, index=False)

    return simulated_data_df