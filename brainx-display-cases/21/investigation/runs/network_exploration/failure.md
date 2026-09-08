# Failed exploratory launch

This launch failed before producing results: Expon's default conductance State had mS units while the modeled membrane and communication used mS/cm2. The corrected code supplies `g_initializer=Constant(0 * GD)`. `network_exploration_v2` reruns the configuration. Do not use this directory as completed evidence. Earlier source hashes are retained; immutable source snapshots are supplied for subsequent named runs.
