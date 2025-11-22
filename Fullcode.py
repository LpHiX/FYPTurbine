import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Union, Callable, Any
from numpy.typing import NDArray

# --- Type Definitions ---
FloatOrArray = Union[float, NDArray[np.float64]]

# --- Material Classes ---
@dataclass
class Gas:
    R: float # J/(kg·K)
    gamma: float

@dataclass
class Liquid:
    density: float

# --- Flow Formulas ---
def gasFlowrate(Kv, P_1, P_2, T_1, gas, assumeChoked=False):
    P_1 = np.atleast_1d(P_1).astype(float)
    P_2 = np.atleast_1d(P_2).astype(float)
    T_1 = np.atleast_1d(T_1).astype(float)
    Kv  = np.atleast_1d(Kv).astype(float)
    
    try:
        P_1, P_2, T_1, Kv = np.broadcast_arrays(P_1, P_2, T_1, Kv)
    except ValueError:
        raise ValueError("Input shapes (P, T, Kv) are incompatible.")

    P_N_Pa = 101325.0
    T_N_K  = 273.15
    rho_n  = P_N_Pa / (gas.R * T_N_K)

    p1_bar = P_1 / 1e5
    p2_bar = P_2 / 1e5

    reverse_mask = p2_bar > p1_bar
    pu = np.where(reverse_mask, p2_bar, p1_bar)
    pd = np.where(reverse_mask, p1_bar, p2_bar)

    critical_ratio = 0.5
    assumeChoked = np.atleast_1d(assumeChoked)
    is_choked = (pd < (pu * critical_ratio)) | ((assumeChoked == True) | (assumeChoked == 1))

    q_n = np.zeros_like(pu)

    sub_idx = ~is_choked
    if np.any(sub_idx):
        dp_sub = pu[sub_idx] - pd[sub_idx]
        term = np.maximum(dp_sub * pd[sub_idx], 0) 
        q_n[sub_idx] = 514 * Kv[sub_idx] * np.sqrt(term / (rho_n * T_1[sub_idx]))

    choked_idx = is_choked
    if np.any(choked_idx):
        q_n[choked_idx] = 257 * Kv[choked_idx] * pu[choked_idx] * np.sqrt(1.0 / (rho_n * T_1[choked_idx]))

    m_dot = q_n * rho_n / 3600.0
    m_dot = np.where(reverse_mask, -m_dot, m_dot)
    return m_dot

def liquidFlowrate(Kv, P_1, P_2, liquid):
    P_1 = np.atleast_1d(P_1).astype(float)
    P_2 = np.atleast_1d(P_2).astype(float)
    Kv  = np.atleast_1d(Kv).astype(float)
    
    try:
        P_1, P_2, Kv = np.broadcast_arrays(P_1, P_2, Kv)
    except ValueError:
        raise ValueError("Input shapes are incompatible.")

    dp = P_1 - P_2
    CONST_LIQUID = 1 / 36000.0
    m_dot = (np.sign(dp) * CONST_LIQUID * Kv * np.sqrt(np.abs(dp) * liquid.density))
    return m_dot

# --- Component Logic ---
def timed_valve_kv(t, maxKv, t_open, t_close, t_ramp, leak_kv=1e-6):
    is_scalar = np.isscalar(t) or np.ndim(t) == 0
    t = np.atleast_1d(t).astype(float)
    k = 12.0 / t_ramp
    center_open = t_open + (t_ramp / 2.0)
    center_close = t_close + (t_ramp / 2.0)
    sig_open = 1.0 / (1.0 + np.exp(-k * (t - center_open)))
    sig_close = 1.0 - (1.0 / (1.0 + np.exp(-k * (t - center_close))))
    Kv = maxKv * sig_open * sig_close
    Kv = np.maximum(Kv, leak_kv)
    if is_scalar: return Kv.item()
    return Kv

def regulator_kv(P_up, P_down, set_pressure, reg_constant=300, leak_kv=1e-6):
    is_scalar = np.isscalar(P_up) or np.ndim(P_up) == 0
    P_up = np.atleast_1d(P_up).astype(float)
    P_down = np.atleast_1d(P_down).astype(float)

    reg_dp = set_pressure - P_down
    
    # FIX: Removed (P_up > set_pressure) condition. 
    # If P_up < set_pressure, the valve should remain open (droop), not close.
    reg_kv = np.where(reg_dp > 0, reg_dp / reg_constant / 1e5, leak_kv)
    
    reg_kv = np.maximum(reg_kv, leak_kv)
    if is_scalar: return reg_kv.item()
    return reg_kv

# --- System Classes ---
@dataclass
class Tank:
    name: str
    volume: float
    pressure: float
    gas: Gas | None = None
    gas_temp: float | None = None
    liquid: Liquid | None = None
    mass_liquid: float = 0
    liquid_temp: float | None = None
    mass_gas: float = field(init=False)
    
    def __post_init__(self):
        if self.gas is not None:
            if self.gas_temp is None: raise ValueError("gas_temp required")
            liquid_vol = (self.mass_liquid / self.liquid.density) if (self.mass_liquid and self.liquid) else 0.0
            self.mass_gas = (self.pressure * (self.volume - liquid_vol)) / (self.gas.R * self.gas_temp)
        else:
            self.mass_gas = 0.0
        
        if self.liquid and self.liquid_temp is None:
            self.liquid_temp = self.gas_temp

@dataclass
class FlowComponent:
    name: str
    kv: Union[FloatOrArray, Callable[[FloatOrArray, FloatOrArray, FloatOrArray], FloatOrArray]]

    def get_kv(self, t, P_up, P_down):
        if callable(self.kv): return self.kv(t, P_up, P_down)
        return self.kv

@dataclass
class FluidNode:
    name: str
    pressure: float | None = None
    temperature: float | None = None
    fluid: Gas | Liquid | None = None
    tank: Tank | None = None
    constant_pressure: bool = False
    nodeComponentTuples: list[tuple["FluidNode", FlowComponent, bool]] = field(default_factory=list)

    def connect_nodes(self, connections):
        self.nodeComponentTuples.extend(connections)

# --- Solver Logic ---
class SystemModel:
    def __init__(self, tanks, nodes):
        self.tanks = tanks
        self.nodes = nodes
        self.variable_nodes = [n for n in nodes if n.tank is None and not n.constant_pressure]

    def get_component_flows(self, t, node_up, node_down, component):
        P_up, P_down = node_up.pressure, node_down.pressure
        kv = component.get_kv(t, P_up, P_down)
        
        # Determine fluid based on pressure direction
        fluid = node_up.fluid if P_up >= P_down else node_down.fluid
        
        if fluid is None:
            # Fallback if fluid propagation failed (should not happen with sorting)
            return 0.0, None

        if isinstance(fluid, Gas):
            T = node_up.temperature if P_up >= P_down else node_down.temperature
            m_dot = gasFlowrate(kv, P_up, P_down, T, fluid)
        else:
            m_dot = liquidFlowrate(kv, P_up, P_down, fluid)
            
        return m_dot, fluid

    def solve_pressures(self, t):
        if not self.variable_nodes: return

        # Initial guess from previous state
        x0 = np.array([n.pressure if n.pressure else 1e5 for n in self.variable_nodes])

        def residuals(pressures):
            # 1. Apply Pressures
            for i, node in enumerate(self.variable_nodes):
                node.pressure = pressures[i]

            # 2. Propagate Fluids (FIX: Sort by Pressure)
            # Sorting ensures we process high-pressure nodes first, propagating fluid info downstream
            sorted_nodes = sorted(self.nodes, key=lambda n: n.pressure if n.pressure is not None else -np.inf, reverse=True)
            
            for node in sorted_nodes:
                if node.tank or node.constant_pressure: continue
                
                # Find best upstream neighbor
                best_neighbor = None
                max_p = -np.inf
                
                for neighbor, _, _ in node.nodeComponentTuples:
                    if neighbor.pressure is not None and neighbor.pressure > node.pressure:
                        if neighbor.pressure > max_p:
                            max_p = neighbor.pressure
                            best_neighbor = neighbor
                
                if best_neighbor and best_neighbor.fluid:
                    node.fluid = best_neighbor.fluid
                    node.temperature = best_neighbor.temperature

            # 3. Calculate Net Flows
            net_flows = []
            for node in self.variable_nodes:
                net_m = 0.0
                for connected, comp, is_upstream in node.nodeComponentTuples:
                    if is_upstream:
                        m, _ = self.get_component_flows(t, connected, node, comp)
                        net_m += m
                    else:
                        m, _ = self.get_component_flows(t, node, connected, comp)
                        net_m -= m
                net_flows.append(net_m)
            
            return np.array(net_flows)

        sol = root(residuals, x0, method='hybr', tol=1e-6)
        for i, node in enumerate(self.variable_nodes):
            node.pressure = sol.x[i]

    def calculate_state(self, t, y, return_aux=False):
        # 1. Update Tanks from State Vector
        for i, tank in enumerate(self.tanks):
            mass_gas = y[2*i]
            mass_liq = y[2*i + 1]
            
            tank.mass_gas = mass_gas
            tank.mass_liquid = mass_liq if tank.liquid else 0.0
            
            vol_liq = tank.mass_liquid / tank.liquid.density if tank.liquid else 0.0
            vol_gas = tank.volume - vol_liq
            
            if vol_gas > 0 and tank.gas:
                tank.pressure = (mass_gas / vol_gas) * tank.gas.R * tank.gas_temp

        # 2. Update Tank Nodes
        for node in self.nodes:
            if node.tank:
                node.pressure = node.tank.pressure
                node.temperature = node.tank.gas_temp
                node.fluid = node.tank.liquid if (node.tank.liquid and node.tank.mass_liquid > 0) else node.tank.gas

        # 3. Solve Network
        self.solve_pressures(t)

        # 4. Calculate Derivatives (and Aux Data)
        dydt = []
        aux_data = {} if return_aux else None
        
        for tank in self.tanks:
            d_gas, d_liq = 0.0, 0.0
            tank_node = next(n for n in self.nodes if n.tank == tank)
            
            for connected, comp, is_upstream in tank_node.nodeComponentTuples:
                if is_upstream:
                    m, fluid = self.get_component_flows(t, connected, tank_node, comp)
                    if isinstance(fluid, Gas): d_gas += m
                    else: d_liq += m
                else:
                    m, fluid = self.get_component_flows(t, tank_node, connected, comp)
                    if isinstance(fluid, Gas): d_gas -= m
                    else: d_liq -= m
                
                if return_aux:
                    # Store Kv and Pressure for plotting
                    # Note: This overwrites if multiple components have same name, but works for this setup
                    aux_data[comp.name + "_kv"] = comp.get_kv(t, 0, 0) # Dummy P for timed valves
                    if callable(comp.kv) and "Regulator" in comp.name:
                         # Re-calculate specific regulator Kv with actual pressures
                         p_u = connected.pressure if is_upstream else tank_node.pressure
                         p_d = tank_node.pressure if is_upstream else connected.pressure
                         aux_data[comp.name + "_kv"] = comp.get_kv(t, p_u, p_d)

            dydt.extend([d_gas, d_liq])
            
        if return_aux:
            # Add intermediate pressures
            for node in self.variable_nodes:
                aux_data[node.name + "_pressure"] = node.pressure
            return dydt, aux_data
            
        return dydt

# --- Setup & Execution ---

N2 = Gas(R=296.8, gamma=1.4)
ipa = Liquid(density=800)

hp_tank = Tank("High Pressure Tank", 0.001, 300e5, N2, 300)
lp_tank = Tank("Low Pressure Tank", 0.01, 32e5, N2, 300, ipa, 8 * 0.001 * 800)

n2_valve = FlowComponent("N2 Time Valve", lambda t, P_up, P_down: timed_valve_kv(t, 0.05, 1, 15, 1))
regulator = FlowComponent("Pressure Regulator", lambda t, P_up, P_down: regulator_kv(P_up, P_down, 55e5))
fuel_valve = FlowComponent("Fuel Valve", lambda t, P_up, P_down: timed_valve_kv(t, 0.5, 2, 20, 1))

hp_node = FluidNode("HP tank", tank=hp_tank)
hpv_node = FluidNode("HPV Reg Node")
lp_node = FluidNode("LP tank", tank=lp_tank)
atm_node = FluidNode("Atmosphere", 101325, 300, N2, constant_pressure=True)

hp_node.connect_nodes([(hpv_node, n2_valve, False)])
hpv_node.connect_nodes([(hp_node, n2_valve, True), (lp_node, regulator, False)])
lp_node.connect_nodes([(hpv_node, regulator, True), (atm_node, fuel_valve, False)])
atm_node.connect_nodes([(lp_node, fuel_valve, True)])

tanks = [hp_tank, lp_tank]
nodes = [hp_node, hpv_node, lp_node, atm_node]
model = SystemModel(tanks, nodes)

# Initial Conditions
y0 = []
for t in tanks:
    y0.extend([t.mass_gas, t.mass_liquid if t.liquid else 0.0])

# Solve
sol = solve_ivp(
    lambda t, y: model.calculate_state(t, y),
    (0, 20), y0, method='LSODA', t_eval=np.linspace(0, 20, 300)
)

# --- Post-Processing (Vectorized / De-duplicated) ---
results = {
    "t": sol.t,
    "masses": sol.y,
    "pressures": [],
    "kvs": {}
}

# Pre-initialize lists for aux data
aux_keys = ["HPV Reg Node_pressure", "N2 Time Valve_kv", "Pressure Regulator_kv", "Fuel Valve_kv"]
aux_lists = {k: [] for k in aux_keys}

for i, t in enumerate(sol.t):
    # Re-run the state calculation to get auxiliary data (pressures, Kvs)
    # This uses the EXACT same logic as the solver, ensuring consistency.
    _, aux = model.calculate_state(t, sol.y[:, i], return_aux=True)
    
    for k in aux_keys:
        if k in aux: aux_lists[k].append(aux[k])
        else: aux_lists[k].append(0) # Fallback

# Plotting
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# 1. Masses
ax[0, 0].plot(sol.t, sol.y[0], 'b-', label='HP Gas')
ax[0, 0].set_ylabel('HP Mass (kg)', color='b')
ax2 = ax[0, 0].twinx()
ax2.plot(sol.t, sol.y[2], 'r-', label='LP Gas')
ax2.plot(sol.t, sol.y[3], 'g-', label='LP Liquid')
ax2.set_ylabel('LP Mass (kg)')
ax[0, 0].set_title("Tank Masses")
ax[0, 0].legend(loc='upper left')
ax2.legend(loc='upper right')

# 2. Pressures
# Calculate tank pressures for plotting
p_hp = []
p_lp = []
for i in range(len(sol.t)):
    # Quick recalc of tank P
    m_g, m_l = sol.y[0, i], sol.y[1, i] # HP
    p_hp.append((m_g / 0.001) * N2.R * 300 / 1e5)
    
    m_g, m_l = sol.y[2, i], sol.y[3, i] # LP
    vol_l = m_l / 800
    p_lp.append((m_g / (0.01 - vol_l)) * N2.R * 300 / 1e5)

ax[0, 1].plot(sol.t, p_hp, 'b-', label='HP Tank')
ax[0, 1].plot(sol.t, np.array(aux_lists["HPV Reg Node_pressure"])/1e5, 'g-', label='HPV Node')
ax[0, 1].plot(sol.t, p_lp, 'r-', label='LP Tank')
ax[0, 1].set_ylabel('Pressure (bar)')
ax[0, 1].legend()
ax[0, 1].set_title("System Pressures")

# 3. Kvs
ax[1, 0].plot(sol.t, aux_lists["N2 Time Valve_kv"], label='N2 Valve')
ax[1, 0].plot(sol.t, aux_lists["Pressure Regulator_kv"], label='Regulator')
ax[1, 0].plot(sol.t, aux_lists["Fuel Valve_kv"], label='Fuel Valve')
ax[1, 0].set_ylabel('Kv')
ax[1, 0].legend()
ax[1, 0].set_title("Valve Kv")

# 4. LP Pressure Zoom
ax[1, 1].plot(sol.t, p_lp, 'r-', label='LP Tank')
# ax[1,1].plot(t, pressure_gas_results[2, :] / 1e5, label='LP Tank Pressure (bar)', color='red')
ax[1,1].set_ylabel('Pressure (bar)', color='red')
ax[1,1].tick_params(axis='y', labelcolor='red')
ax[1,1].set_xlabel('Time (s)')
ax[1,1].legend()

fig.tight_layout()
plt.show()
