import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import math
from typing import Union, Callable

# --- PHYSICS CLASSES ---

@dataclass
class Gas:
    R: float # J/(kg·K)
    gamma: float

@dataclass
class Liquid:
    density: float

# --- ROBUST FLOW CALCULATIONS ---

def continuous_sqrt(x, epsilon=1e-5):
    """
    Calculates sign(x) * sqrt(abs(x)) with a linear transition near zero
    to avoid infinite gradients (singularities) for the solver.
    """
    # If x is very small, use linear approximation: sqrt(x) ~ x / sqrt(epsilon)
    # This ensures derivative is finite at 0.
    mask = np.abs(x) < epsilon
    
    # Standard sqrt for normal flow
    safe_sqrt = np.sign(x) * np.sqrt(np.abs(x))
    
    # Linearized flow for tiny pressure drops
    linear_flow = x / np.sqrt(epsilon)
    
    return np.where(mask, linear_flow, safe_sqrt)

def gasFlowrate(Kv, P_1, P_2, T_1, gas, assumeChoked=False):
    """
    Calculates gas mass flow rate with regularization for near-zero pressure drops.
    """
    # Broadcast if necessary
    if not np.isscalar(P_1) or not np.isscalar(P_2):
        try:
            P_1, P_2, T_1, Kv = np.broadcast_arrays(P_1, P_2, T_1, Kv)
        except ValueError:
            raise ValueError("Input shapes incompatible.")

    # Constants
    P_N_Pa = 101325.0
    T_N_K  = 273.15
    rho_n  = P_N_Pa / (gas.R * T_N_K)

    p1_bar = P_1 / 1e5
    p2_bar = P_2 / 1e5

    # Determine flow direction based on pressure difference
    # We calculate DP and allow negative results for reverse flow
    dp_bar = p1_bar - p2_bar
    
    # Upstream/Downstream determination for density/choking logic
    # (Note: For reverse flow, we swap just for density calculation purposes)
    reverse = p2_bar > p1_bar
    pu = np.where(reverse, p2_bar, p1_bar)
    pd = np.where(reverse, p1_bar, p2_bar)
    
    # Choked flow logic
    critical_ratio = 0.5
    is_choked = (pd < (pu * critical_ratio)) | (np.atleast_1d(assumeChoked) == True)

    # Flow Calculation
    q_n = np.zeros_like(pu, dtype=float)

    # 1. Choked Flow
    if np.any(is_choked):
        # Pure choked flow doesn't happen in reverse for this simplified model 
        # unless we are careful, but usually high dP implies valid flow direction.
        q_n[is_choked] = 257.0 * Kv[is_choked] * pu[is_choked] * np.sqrt(1.0 / (rho_n * T_1[is_choked]))

    # 2. Subcritical Flow (Regularized)
    sub_idx = ~is_choked
    if np.any(sub_idx):
        # Use the signed difference for direction
        # Term = dp * p_downstream (approximation)
        # We use a continuous sqrt function to handle the 0 crossing smoothly
        
        dp_sub = pu[sub_idx] - pd[sub_idx]
        avg_p = pd[sub_idx] # Using downstream is standard for this ISO simplified model
        
        term = dp_sub * avg_p
        
        # Calculate using continuous sqrt to allow solver to cross zero smoothly
        root_term = continuous_sqrt(term) 
        
        q_n[sub_idx] = 514.0 * Kv[sub_idx] * root_term / np.sqrt(rho_n * T_1[sub_idx])

    # Convert to kg/s
    m_dot = q_n * rho_n / 3600.0
    
    # Apply sign (if we swapped pu/pd, the logic above yielded positive Q, 
    # but we need to respect the original P1-P2 direction)
    
    # If flow was calculated using absolute dP, re-apply sign:
    final_sign = np.sign(dp_bar)
    
    return m_dot * final_sign

def liquidFlowrate(Kv, P_1, P_2, liquid):
    """
    Calculates liquid mass flow rate with regularization.
    """
    if not np.isscalar(P_1):
        P_1, P_2, Kv = np.broadcast_arrays(P_1, P_2, Kv)

    dp = P_1 - P_2
    CONST_LIQUID = 1.0 / 36000.0
    
    # Use robust sqrt
    # continuous_sqrt handles the sign and the singularity at 0
    m_dot = CONST_LIQUID * Kv * continuous_sqrt(dp * liquid.density)
    return m_dot

# --- COMPONENT DEFINITIONS ---

@dataclass
class Tank:
    name: str
    volume: float  # m^3
    pressure: float # Pa
    gas: Gas | None= None 
    gas_temp: float | None = None
    liquid: Liquid | None= None
    mass_liquid: float = 0 
    liquid_temp: float | None = None
    mass_gas: float = field(init=False)
    
    def __post_init__(self):
        if self.gas is not None:
            if self.gas_temp is None:
                raise ValueError("gas_temp must be provided")
            liquid_vol = 0.0
            if self.mass_liquid > 0 and self.liquid:
                liquid_vol = self.mass_liquid / self.liquid.density
            
            self.mass_gas = (self.pressure * (self.volume - liquid_vol)) / (self.gas.R * self.gas_temp)
        else:
            self.mass_gas = 0.0
        if self.liquid and self.liquid_temp is None:
            self.liquid_temp = self.gas_temp

FloatOrArray = Union[float, np.typing.NDArray[np.float64]]

@dataclass
class FlowComponent:
    name: str
    kv: Union[FloatOrArray, Callable[[FloatOrArray, FloatOrArray, FloatOrArray], FloatOrArray]]

    def get_kv(self, t, P_up, P_down):
        if callable(self.kv):
            return self.kv(t, P_up, P_down)
        return self.kv

def timed_valve_kv(t, maxKv, t_open, t_close, t_ramp, leak_kv=1e-8):
    # Vectorized or scalar handling
    t = np.atleast_1d(t).astype(float)
    k = 12.0 / t_ramp
    center_open = t_open + (t_ramp / 2.0)
    center_close = t_close + (t_ramp / 2.0)
    
    sig_open = 1.0 / (1.0 + np.exp(-k * (t - center_open)))
    sig_close = 1.0 - (1.0 / (1.0 + np.exp(-k * (t - center_close))))
    
    Kv = maxKv * sig_open * sig_close
    return np.maximum(Kv, leak_kv)

def regulator_kv(P_up, P_down, set_pressure, reg_constant=300, leak_kv=1e-8):
    """
    Robust Regulator Logic:
    Decouples mechanical valve lift from flow direction.
    Even if P_down > P_up (reverse flow), the regulator stays physically open
    if P_down < set_pressure. 
    """
    P_up = np.atleast_1d(P_up)
    P_down = np.atleast_1d(P_down)
    
    # Calculate mechanical opening based on Downstream pressure vs Setpoint
    # The regulator tries to open if Downstream < Setpoint
    pressure_deficit = set_pressure - P_down
    
    # If pressure_deficit > 0, valve opens. 
    # We smooth the transition at 0 slightly or just use max(0, ...)
    # Note: Dividing by 1e5 keeps the constant in bar units
    opening_kv = np.maximum(pressure_deficit, 0.0) / reg_constant / 1e5
    
    # Note: We REMOVED the (P_up > P_down) check. 
    # If P_up < P_down, the valve is still mechanically open, allowing reverse flow.
    # This prevents the solver discontinuity.
    
    return np.maximum(opening_kv, leak_kv)

@dataclass
class FluidNode:
    name: str
    pressure: float | None = None
    temperature: float| None = None
    fluid: Gas | Liquid | None = None
    tank: Tank | None = None
    constant_pressure: bool = False
    nodeComponentTuples: list = None 

    def connect_nodes(self, tuples):
        if self.nodeComponentTuples is None: self.nodeComponentTuples = []
        self.nodeComponentTuples.extend(tuples)

# --- NETWORK SOLVER ---

def solve_network_pressures(all_nodes: list[FluidNode], t: float):
    variable_nodes = [n for n in all_nodes if n.tank is None and not n.constant_pressure]
    if not variable_nodes: return

    # SCALING: Solve for pressure in BAR (or units of 1e5 Pa)
    # This keeps variable values ~1.0 to 300.0 instead of 10^5 to 10^7
    SCALE_P = 1e5
    
    # Initial guess (in scaled units)
    x0 = np.array([n.pressure/SCALE_P if n.pressure else 1.0 for n in variable_nodes])

    def residuals(p_scaled_array):
        # 1. Update node pressures from solver guess
        for i, node in enumerate(variable_nodes):
            node.pressure = p_scaled_array[i] * SCALE_P
            
            # Propagate fluid/temp properties (Simple heuristic)
            # Find the neighbor with highest pressure to inherit properties
            # (In a real solver, this is an enthalpy balance, but this suffices here)
            best_neighbor = None
            max_p = -1e9
            for neighbor, _, _ in node.nodeComponentTuples:
                p_neighbor = neighbor.pressure
                if p_neighbor is not None and p_neighbor > max_p:
                    max_p = p_neighbor
                    best_neighbor = neighbor
            
            if best_neighbor and best_neighbor.fluid:
                node.fluid = best_neighbor.fluid
                node.temperature = best_neighbor.temperature

        # 2. Calculate Net Mass Flow for each variable node
        net_flows = []
        for node in variable_nodes:
            net_m = 0.0
            for connected_node, component, is_upstream in node.nodeComponentTuples:
                # Note: get_component_flows handles the physics
                m_dot, _ = get_component_flows(t, 
                                             connected_node if is_upstream else node, 
                                             node if is_upstream else connected_node, 
                                             component)
                
                # If component is upstream (Connected -> Node), add flow
                # If component is downstream (Node -> Connected), subtract flow
                if is_upstream:
                    net_m += float(m_dot)
                else:
                    net_m -= float(m_dot)
            
            # SCALING THE RESIDUAL:
            # Mass flows are often small (e.g. 0.01 kg/s). 
            # Least Squares likes residuals ~ 1.0. 
            # Multiply by a constant to stiffen the gradient for the solver.
            net_flows.append(net_m * 100.0) 

        return np.array(net_flows)

    # Use 'trf' (Trust Region Reflective) with bounds to prevent negative pressures
    # Bounds in scaled units (0.01 bar to 1000 bar)
    lower_bounds = np.full_like(x0, 0.01)
    upper_bounds = np.full_like(x0, 1000.0)
    
    res = least_squares(residuals, x0, bounds=(lower_bounds, upper_bounds), 
                        ftol=1e-6, xtol=1e-6, method='trf')
    
    # Update final pressures
    for i, node in enumerate(variable_nodes):
        node.pressure = res.x[i] * SCALE_P

def get_component_flows(t, node_up: FluidNode, node_down: FluidNode, component: FlowComponent):
    P_up = node_up.pressure
    P_down = node_down.pressure
    
    # Determine fluid based on high pressure side
    fluid_obj = node_up.fluid if P_up >= P_down else node_down.fluid
    
    kv_val = component.get_kv(t, P_up, P_down)
    
    if isinstance(fluid_obj, Gas):
        T_up = node_up.temperature if P_up >= P_down else node_down.temperature
        m_dot = gasFlowrate(kv_val, P_up, P_down, T_up, fluid_obj)
    else:
        m_dot = liquidFlowrate(kv_val, P_up, P_down, fluid_obj)
        
    return m_dot, fluid_obj

# --- MAIN SIMULATION SETUP ---

N2 = Gas(R=296.8, gamma=1.4)
ipa = Liquid(density=800)

hp_tank = Tank("HP Tank", volume=0.001, pressure=300e5, gas=N2, gas_temp=300)
lp_tank = Tank("LP Tank", volume=0.01, pressure=32e5, gas=N2, gas_temp=300, 
               liquid=ipa, mass_liquid=8*0.001*800)

n2_valve = FlowComponent("N2 Valve", 
    lambda t, Pu, Pd: timed_valve_kv(t, 0.1, 1, 15, 1))

regulator = FlowComponent("Regulator", 
    lambda t, Pu, Pd: regulator_kv(Pu, Pd, 55e5))

fuel_valve = FlowComponent("Fuel Valve", 
    lambda t, Pu, Pd: timed_valve_kv(t, 1.0, 2, 20, 1))

test_valve_1 = FlowComponent("Test Valve 1", 1.0)
test_valve_2 = FlowComponent("Test Valve 2", 1.0)

# Nodes
hp_node = FluidNode("HP", tank=hp_tank)
reg_node = FluidNode("RegNode", pressure=32e5) # Initial guess helps
lp_node = FluidNode("LP", tank=lp_tank)
mid1_node = FluidNode("Mid1", pressure=1e5)
mid2_node = FluidNode("Mid2", pressure=1e5)
atm_node = FluidNode("Atm", pressure=101325, constant_pressure=True, fluid=N2, temperature=300)

# Connections (Upstream Node, Component, is_component_upstream_relative_to_self_arg)
# To simplify, let's just manually build the list passed to solver
hp_node.connect_nodes([(reg_node, n2_valve, False)])
reg_node.connect_nodes([(hp_node, n2_valve, True), (lp_node, regulator, False)])
lp_node.connect_nodes([(reg_node, regulator, True), (mid1_node, fuel_valve, False)])
mid1_node.connect_nodes([(lp_node, fuel_valve, True), (mid2_node, test_valve_1, False)])
mid2_node.connect_nodes([(mid1_node, test_valve_1, True), (atm_node, test_valve_2, False)])
atm_node.connect_nodes([(mid2_node, test_valve_2, True)])

all_nodes = [hp_node, reg_node, lp_node, mid1_node, mid2_node, atm_node]
tanks = [hp_tank, lp_tank]

def system_derivs(t, y):
    # Unpack state
    idx = 0
    for tank in tanks:
        tank.mass_gas = y[idx]
        tank.mass_liquid = y[idx+1]
        
        # Update Tank Pressure
        vol_liq = 0.0
        if tank.liquid: vol_liq = tank.mass_liquid / tank.liquid.density
        vol_gas = max(tank.volume - vol_liq, 1e-6) # prevent zero vol
        rho_gas = tank.mass_gas / vol_gas
        tank.pressure = rho_gas * tank.gas.R * tank.gas_temp
        
        idx += 2
        
    # Update Node Pressures from Tanks
    for node in all_nodes:
        if node.tank:
            node.pressure = node.tank.pressure
            node.fluid = node.tank.liquid if node.tank.mass_liquid > 0 else node.tank.gas
            node.temperature = node.tank.gas_temp

    # Solve Algebraics (Intermediate Pressures)
    solve_network_pressures(all_nodes, t)
    
    # Calculate Derivatives (Mass Flows)
    dydt = []
    for tank in tanks:
        tank_node = next(n for n in all_nodes if n.tank == tank)
        dm_gas = 0.0
        dm_liq = 0.0
        
        for conn_node, comp, is_upstream_input in tank_node.nodeComponentTuples:
            # If is_upstream_input is True, comp is upstream OF tank_node (Flow IN)
            # If False, comp is downstream (Flow OUT)
            
            if is_upstream_input:
                mdot, fluid = get_component_flows(t, conn_node, tank_node, comp)
                flow = float(mdot) # Positive means In
            else:
                mdot, fluid = get_component_flows(t, tank_node, conn_node, comp)
                flow = -float(mdot) # Negative means Out
            
            if isinstance(fluid, Gas):
                dm_gas += flow
            else:
                dm_liq += flow
        
        dydt.extend([dm_gas, dm_liq])
        
    return dydt

def main():
    # Initial State
    y0 = []
    for t in tanks:
        y0.extend([t.mass_gas, t.mass_liquid])
        
    print("Starting Simulation...")
    sol = solve_ivp(system_derivs, (0, 20), y0, method='LSODA', rtol=1e-4, atol=1e-6)
    print(f"Solved successfully: {sol.success}")
    
    # Plotting
    time = sol.t
    hp_p = []
    lp_p = []
    reg_p = []
    
    for i in range(len(time)):
        # Reconstruct pressures for plotting
        # (Ideally return this from solver, but for brevity we recalculate or just inspect states)
        # Simplification: Just plot the tank pressures calculated from mass
        m_hp_gas = sol.y[0][i]
        p_hp = (m_hp_gas / hp_tank.volume) * N2.R * 300
        hp_p.append(p_hp/1e5)
        
        m_lp_gas = sol.y[2][i]
        m_lp_liq = sol.y[3][i]
        v_liq = m_lp_liq / ipa.density
        p_lp = (m_lp_gas / (lp_tank.volume - v_liq)) * N2.R * 300
        lp_p.append(p_lp/1e5)

    plt.figure(figsize=(10,6))
    plt.plot(time, hp_p, label='HP Tank (bar)')
    plt.plot(time, lp_p, label='LP Tank (bar)')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (bar)')
    plt.title('Tank Pressures')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()