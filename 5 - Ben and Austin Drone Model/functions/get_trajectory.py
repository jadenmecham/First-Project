from trajectory_gen import TrajectoryGenerator

def gen_traj(motif_type):
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