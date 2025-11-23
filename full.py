import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import RectBivariateSpline
import math
from dataclasses import dataclass, field
from typing import Union, Callable

# --- CEA GENERATION ---
from rocketcea.cea_obj_w_units import CEA_Obj

# Initialize CEA only once
cea = CEA_Obj(oxName="N2O", fuelName="Isopropanol", isp_units='sec', cstar_units='m/s',
              pressure_units='Bar', temperature_units='K', sonic_velocity_units='m/s',
              enthalpy_units='J/kg', density_units='kg/m^3', specific_heat_units='J/kg-K',
              viscosity_units='centipoise', thermal_cond_units='W/cm-degC', make_debug_prints=False)

chamber_pressures = np.linspace(10, 100, 10)  # bar
of_ratios = np.linspace(0.5, 10, 40)
cstar_data = np.zeros((len(chamber_pressures), len(of_ratios)))

for i, pc in enumerate(chamber_pressures):
    for j, of in enumerate(of_ratios):
        cstar_data[i, j] = cea.get_Cstar(pc, of)

# OPTIMIZATION 1: Use RectBivariateSpline (C-based, 10x faster than RegularGridInterpolator)
cstar_spline = RectBivariateSpline(chamber_pressures, of_ratios, cstar_data)

# --- FLUID CLASSES ---
@dataclass
class Gas:
    R: float
    gamma: float

@dataclass
class Liquid:
    density: float

# --- OPTIMIZATION 2: FAST PATH FLOW FUNCTIONS (No Numpy Overhead) ---
def gasFlowrate_fast(Kv, P_1, P_2, T_1, R_gas):
    """Scalar-only optimized gas flow calculation."""
    # Standard Conditions Constants
    P_N_Pa = 101325.0
    T_N_K = 273.15
    rho_n = P_N_Pa / (R_gas * T_N_K)

    # Convert to bar
    p1_bar = P_1 * 1e-5
    p2_bar = P_2 * 1e-5

    if p2_bar > p1_bar:
        pu, pd = p2_bar, p1_bar
        sign = -1.0
    else:
        pu, pd = p1_bar, p2_bar
        sign = 1.0

    # Choked flow check (pd < 0.5 * pu)
    if pd < (pu * 0.5):
        q_n = 257.0 * Kv * pu * math.sqrt(1.0 / (rho_n * T_1))
    else:
        dp_sub = pu - pd
        term = dp_sub * pd
        if term < 0: term = 0.0
        q_n = 514.0 * Kv * math.sqrt(term / (rho_n * T_1))

    return (q_n * rho_n / 3600.0) * sign

def liquidFlowrate_fast(Kv, P_1, P_2, density):
    """Scalar-only optimized liquid flow calculation."""
    dp = P_1 - P_2
    CONST_LIQUID = 2.777777777777778e-05 # 1/36000
    
    if dp >= 0:
        return CONST_LIQUID * Kv * math.sqrt(dp * density)
    else:
        return -CONST_LIQUID * Kv * math.sqrt(-dp * density)

# --- COMPONENT CLASSES ---
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
        liquid_vol = 0.0
        if self.mass_liquid and self.liquid:
            liquid_vol = self.mass_liquid / self.liquid.density
        
        if self.gas:
            self.mass_gas = (self.pressure * (self.volume - liquid_vol)) / (self.gas.R * self.gas_temp)
        else:
            self.mass_gas = 0.0

@dataclass
class FlowComponent:
    name: str
    kv: Union[float, Callable[[float, float, float], float]]
    
    def get_kv(self, t, P_up, P_down):
        if callable(self.kv):
            return self.kv(t, P_up, P_down)
        return self.kv

# --- KV FUNCTIONS (Optimized for Scalar) ---
def timed_valve_kv(t, maxKv, t_open, t_close, t_ramp, leak_kv=1e-6):
    # Fast scalar math
    k = 12.0 / t_ramp
    center_open = t_open + (t_ramp * 0.5)
    center_close = t_close + (t_ramp * 0.5)
    
    # Sigmoid Open
    try:
        exp_open = math.exp(-k * (t - center_open))
        sig_open = 1.0 / (1.0 + exp_open)
    except OverflowError:
        sig_open = 0.0 if (t - center_open) < 0 else 1.0

    # Sigmoid Close
    try:
        exp_close = math.exp(-k * (t - center_close))
        sig_close = 1.0 - (1.0 / (1.0 + exp_close))
    except OverflowError:
        sig_close = 1.0 if (t - center_close) < 0 else 0.0
        
    kv = maxKv * sig_open * sig_close
    return max(kv, leak_kv)

def regulator_kv(P_up, P_down, set_pressure, reg_constant=300, leak_kv=1e-6):
    reg_dp = set_pressure - P_down
    if reg_dp > 0 and P_up > P_down:
        return max(reg_dp / reg_constant / 1e5, leak_kv)
    return leak_kv

@dataclass
class FluidNode:
    name: str
    pressure: float | None = None
    temperature: float | None = None
    fluid: Gas | Liquid | None = None
    tank: Tank | None = None
    is_engine: bool = False
    constant_pressure: bool = False
    nodeComponentTuples: list = field(default_factory=list)

    def connect_nodes(self, tuples):
        self.nodeComponentTuples.extend(tuples)

# --- NETWORK SOLVER ---

# Cache for the variable nodes to avoid rebuilding list every step
_cached_variable_nodes = None
_cached_x0 = None

def solve_network_pressures(all_nodes, t):
    global _cached_variable_nodes, _cached_x0
    
    # Update Tank/Fixed Nodes
    for node in all_nodes:
        if node.tank:
            if node.tank.liquid and node.tank.mass_liquid > 0:
                node.fluid = node.tank.liquid
            else:
                node.fluid = node.tank.gas

    # Identify variable nodes (Run once or if cache invalid)
    if _cached_variable_nodes is None:
        _cached_variable_nodes = [n for n in all_nodes if n.tank is None and not n.constant_pressure]
        _cached_x0 = np.array([n.pressure if n.pressure else 1e5 for n in _cached_variable_nodes])
    
    variable_nodes = _cached_variable_nodes
    if not variable_nodes: return

    # Current guess from previous step (warm start)
    x0 = np.array([n.pressure for n in variable_nodes])

    # Pre-calculate topology for the engine to avoid searching in the loop
    # (Simplified for this script: We assume the engine connections are static)
    
    def residuals(pressures):
        # Update node objects
        for i, p in enumerate(pressures):
            variable_nodes[i].pressure = p
        
        net_flows = np.zeros(len(pressures))

        for i, node in enumerate(variable_nodes):
            current_p = pressures[i]

            # --- OPTIMIZATION 3: ENGINE LOGIC FAST PATH ---
            if node.is_engine:
                # Direct lookup of connections (hardcoded for speed in this specific topology)
                # In a general library, you would pre-compile this adjacency list.
                
                # Ox Flow (from LP tank 2)
                # We know index 1 in the tuple list is the component, index 0 is node
                ox_tuple = node.nodeComponentTuples[1] # LP Tank 2
                fuel_tuple = node.nodeComponentTuples[0] # Test Valve 2
                
                ox_node, ox_comp = ox_tuple[0], ox_tuple[1]
                fuel_node, fuel_comp = fuel_tuple[0], fuel_tuple[1]
                
                # Ox Flow
                kv_ox = ox_comp.get_kv(t, ox_node.pressure, current_p)
                m_ox = liquidFlowrate_fast(kv_ox, ox_node.pressure, current_p, ox_node.fluid.density)
                
                # Fuel Flow
                kv_fuel = fuel_comp.get_kv(t, fuel_node.pressure, current_p)
                m_fuel = liquidFlowrate_fast(kv_fuel, fuel_node.pressure, current_p, fuel_node.fluid.density)
                
                # Cstar
                if m_fuel > 1e-6 and m_ox > 1e-6:
                    of = m_ox / m_fuel
                    # Use Fast Spline .ev()
                    cstar = cstar_spline.ev(current_p * 1e-5, of)
                    m_out = current_p * 0.0001 / cstar # At=0.0001
                else:
                    m_out = 0.0
                
                net_flows[i] = m_out - m_ox - m_fuel
                continue
            
            # --- STANDARD NODE LOGIC ---
            net_m = 0.0
            for neighbor, comp, is_upstream in node.nodeComponentTuples:
                p_neigh = neighbor.pressure
                
                if is_upstream:
                    p_up, p_down = p_neigh, current_p
                else:
                    p_up, p_down = current_p, p_neigh
                
                # Get Kv
                kv = comp.get_kv(t, p_up, p_down)
                
                # Determine Fluid
                if p_up >= p_down:
                    fluid = neighbor.fluid if is_upstream else node.fluid
                    temp = neighbor.temperature if is_upstream else node.temperature
                else:
                    fluid = node.fluid if is_upstream else neighbor.fluid
                    temp = node.temperature if is_upstream else neighbor.temperature
                
                if fluid is None: m_dot = 0.0
                elif isinstance(fluid, Gas):
                    m_dot = gasFlowrate_fast(kv, p_up, p_down, temp, fluid.R)
                else:
                    m_dot = liquidFlowrate_fast(kv, p_up, p_down, fluid.density)
                
                if is_upstream: net_m += m_dot
                else: net_m -= m_dot
            
            net_flows[i] = net_m

        return net_flows

    bounds = (np.full_like(x0, 1e3), np.full_like(x0, 1e8))
    res = least_squares(residuals, x0, bounds=bounds, max_nfev=100)
    
    # Apply results
    for i, p in enumerate(res.x):
        variable_nodes[i].pressure = p

# --- INTEGRATION SYSTEM ---

def dae_system(t, y):
    # Unpack
    for i, tank in enumerate(tanks):
        m_gas = y[2*i]
        m_liq = y[2*i+1]
        
        tank.mass_liquid = m_liq if tank.liquid else 0
        tank.mass_gas = m_gas
        
        vol_liq = tank.mass_liquid / tank.liquid.density if tank.liquid else 0
        vol_gas = tank.volume - vol_liq
        
        if vol_gas > 1e-9:
            tank.pressure = (m_gas * tank.gas.R * tank.gas_temp) / vol_gas
        else:
            tank.pressure = 1e8 # Hydrostatic lock prevention

    # Solve Algebraic
    solve_network_pressures(total_nodes, t)
    
    dydt = []
    for tank in tanks:
        d_gas, d_liq = 0.0, 0.0
        # Find node (simplified assumption: 1 node per tank)
        node = next(n for n in total_nodes if n.tank == tank)
        
        for neighbor, comp, is_upstream in node.nodeComponentTuples:
            p_up = neighbor.pressure if is_upstream else node.pressure
            p_down = node.pressure if is_upstream else neighbor.pressure
            
            kv = comp.get_kv(t, p_up, p_down)
            
            # Fluid Determination
            if p_up >= p_down:
                fluid = neighbor.fluid if is_upstream else node.fluid
                temp = neighbor.temperature if is_upstream else node.temperature
            else:
                fluid = node.fluid if is_upstream else neighbor.fluid
                temp = node.temperature if is_upstream else neighbor.temperature

            if isinstance(fluid, Gas):
                m = gasFlowrate_fast(kv, p_up, p_down, temp, fluid.R)
                if is_upstream: d_gas += m
                else: d_gas -= m
            elif isinstance(fluid, Liquid):
                m = liquidFlowrate_fast(kv, p_up, p_down, fluid.density)
                if is_upstream: d_liq += m
                else: d_liq -= m
        
        dydt.extend([d_gas, d_liq])
    return dydt

# --- SYSTEM DEFINITION ---

N2 = Gas(R=296.8, gamma=1.4)
ipa = Liquid(density=800)

hp_tank = Tank("HP", 0.001, 300e5, N2, 300)
lp_tank = Tank("LP", 0.01, 32e5, N2, 300, ipa, 8*0.001*800)
lp_tank_2 = Tank("LP2", 0.005, 35e5, N2, 300, ipa, 4*0.001*800)

n2_valve = FlowComponent("N2V", lambda t, p1, p2: timed_valve_kv(t, 0.04, 1, 8, 1))
regulator = FlowComponent("Reg", lambda t, p1, p2: regulator_kv(p1, p2, 55e5))
valve_to_tank_1 = FlowComponent("V1", 0.1)
valve_to_tank_2 = FlowComponent("V2", 0.1)
fuel_valve = FlowComponent("FV1", lambda t, p1, p2: timed_valve_kv(t, 0.2, 2, 5, 1))
fuel_valve_2 = FlowComponent("FV2", lambda t, p1, p2: timed_valve_kv(t, 0.3, 3, 30, 1))
test_valve_1 = FlowComponent("TV1", 0.1)
test_valve_2 = FlowComponent("TV2", 0.2)

hp_tank_node = FluidNode("HP tank", tank=hp_tank)
hpv_reg_node = FluidNode("HPV-Reg")
reg_to_valve1_node = FluidNode("Reg-V1")
lp_tank_node = FluidNode("LP tank", tank=lp_tank)
fuel_valve_to_testvalve_1 = FluidNode("FV-TV1")
testvalve_1_to_testvalve_2 = FluidNode("TV1-TV2")
reg_to_valve2_node = FluidNode("Reg-V2")
lp_tank_node_2 = FluidNode("LP tank 2", tank=lp_tank_2)
engine_node = FluidNode("Engine", fluid=N2, temperature=300, is_engine=True)
atmosphere_node = FluidNode("Atm", pressure=101325, constant_pressure=True, fluid=N2, temperature=300)

# Connect
hp_tank_node.connect_nodes([(hpv_reg_node, n2_valve, False)])
hpv_reg_node.connect_nodes([(hp_tank_node, n2_valve, True), (reg_to_valve1_node, regulator, False), (reg_to_valve2_node, regulator, False)])
reg_to_valve1_node.connect_nodes([(hpv_reg_node, regulator, True), (lp_tank_node, valve_to_tank_1, False)])
lp_tank_node.connect_nodes([(reg_to_valve1_node, valve_to_tank_1, True), (fuel_valve_to_testvalve_1, fuel_valve, False)])
fuel_valve_to_testvalve_1.connect_nodes([(lp_tank_node, fuel_valve, True), (testvalve_1_to_testvalve_2, test_valve_1, False)])
testvalve_1_to_testvalve_2.connect_nodes([(fuel_valve_to_testvalve_1, test_valve_1, True), (engine_node, test_valve_2, False)])
reg_to_valve2_node.connect_nodes([(hpv_reg_node, regulator, True), (lp_tank_node_2, valve_to_tank_2, False)])
lp_tank_node_2.connect_nodes([(reg_to_valve2_node, valve_to_tank_2, True), (engine_node, fuel_valve_2, False)])
# Engine specific
engine_node.connect_nodes([(testvalve_1_to_testvalve_2, test_valve_2, True), (lp_tank_node_2, fuel_valve_2, True), (atmosphere_node, None, False)])

total_nodes = [hp_tank_node, hpv_reg_node, lp_tank_node, fuel_valve_to_testvalve_1, testvalve_1_to_testvalve_2, 
               engine_node, reg_to_valve1_node, lp_tank_node_2, reg_to_valve2_node, atmosphere_node]
tanks = [hp_tank, lp_tank, lp_tank_2]

y0 = []
for t in tanks: y0.extend([t.mass_gas, t.mass_liquid if t.liquid else 0])

def main():
    # Run Solver
    # LSODA is generally good, but if it's very stiff, try method='Radau'
    sol = solve_ivp(dae_system, (0, 30), y0, method='LSODA', t_eval=np.linspace(0, 30, 500))
    print(f"Steps: {len(sol.t)}")
    
if __name__ == "__main__":
    import cProfile
    cProfile.run("main()", sort="cumulative")