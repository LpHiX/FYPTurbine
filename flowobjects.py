from numba import njit
import numpy as np
from dataclasses import dataclass
import math

@dataclass
class Gas:
    name: str
    R: float # J/(kg·K)
    gamma: float 
    viscosity: float 

    
@dataclass
class Liquid:
    name: str
    density: float # kg/m^3
    viscosity: float  # Pa·s
    

import numpy as np

P_N_Pa = 101325.0       # Pa
T_N_K  = 273.15         # K

# Constants pre-calculated for speed
CONST_LIQ = 1.0 / 36000.0
RHO_N_CONST = 101325.0 / 273.15  # ~370.95

@njit(cache=True)
def fast_gas_flow(Kv, P_up, P_down, T_up, gas_R):
    # 1. Flow Direction
    if P_up > P_down:
        pu, pd = P_up, P_down
        sign = 1.0
    else:
        pu, pd = P_down, P_up
        sign = -1.0

    # 2. Constants
    rho_n = RHO_N_CONST / gas_R 
    pu_bar = pu * 1e-5
    pd_bar = pd * 1e-5

    # 3. Choking (Simplified)
    if pd < (pu * 0.5):
        # Choked
        q_n = 257.0 * Kv * pu_bar * math.sqrt(1.0 / (rho_n * T_up))
    else:
        # Subsonic
        dp_sub = pu_bar - pd_bar
        term = dp_sub * pd_bar
        if term < 0: term = 0.0
        q_n = 514.0 * Kv * math.sqrt(term / (rho_n * T_up))

    # 4. Mass Flow (kg/s) -> q_n [Nm3/h] * rho_n [kg/Nm3] / 3600
    return (q_n * rho_n / 3600.0) * sign

@njit(cache=True)
def fast_liquid_flow(Kv, P_up, P_down, liquid_rho):
    dp = P_up - P_down
    if dp >= 0:
        return CONST_LIQ * Kv * math.sqrt(dp * liquid_rho)
    else:
        return -CONST_LIQ * Kv * math.sqrt(-dp * liquid_rho)

@njit(cache=True)
def gasFlowrate(Kv, P_1, P_2, T_1, gas, assumeChoked=False):
    """
    Calculates gas mass flow rate using the Simplified Kv Model (Tameson).
    Includes broadcasting to handle mix of scalar and array inputs.
    """
    
    # Constants for Standard Conditions (NTP: 0°C, 1 atm)

    rho_n  = P_N_Pa / (gas.R * T_N_K) # Normal Density (kg/m^3)

    # Convert Pressures to BAR for the Kv formula
    p1_bar = P_1 / 1e5
    p2_bar = P_2 / 1e5

    if p2_bar > p1_bar:
        pu = p2_bar
        pd = p1_bar
        sign = -1.0
    else:
        pu = p1_bar
        pd = p2_bar
        sign = 1.0

    # Simplified Model assumes choking at P_down < P_up / 2
    critical_ratio = 0.5
    
    # Determine if flow is choked
    is_choked = (pd < (pu * critical_ratio)) or (assumeChoked == True) or (assumeChoked == 1)

    if not is_choked:
        dp_sub = pu - pd
        # Protect against negative sqrt near zero
        term = max(dp_sub * pd, 0.0)
        q_n = 514.0 * Kv * math.sqrt(term / (rho_n * T_1))
    else:
        q_n = 257.0 * Kv * pu * math.sqrt(1.0 / (rho_n * T_1))

    # Final Conversion to Mass Flow (kg/s)
    m_dot = q_n * rho_n / 3600.0
    return m_dot * sign

import numpy as np

CONST_LIQUID = 1.0 / 36000.0

@njit(cache=True)
def liquidFlowrate(Kv, P_1, P_2, liquid):
    """
    Calculates liquid mass flow rate (kg/s) using Kv.
    
    Parameters:
        Kv  : Flow coefficient (m^3/h water @ 1 bar dP)
        P_1 : Upstream pressure (Pa)
        P_2 : Downstream pressure (Pa)
        liquid : Object with .density attribute (kg/m^3)
        
    Returns:
        m_dot : Mass flow rate (kg/s). Negative value indicates reverse flow.
    """
    
    dp = P_1 - P_2
    
    if dp >= 0:
        return CONST_LIQUID * Kv * math.sqrt(dp * liquid.density)
    else:
        return -CONST_LIQUID * Kv * math.sqrt(-dp * liquid.density)


from typing import Union, Callable, Any, List
from numpy.typing import NDArray
from dataclasses import dataclass, field

FloatOrArray = Union[float, NDArray[np.float64]]

@dataclass
class Tank:
    name: str
    volume: float  # m^3
    pressure: float # Pa
    gas: Gas | None= None        # Gas object
    gas_temp: float | None = None       # K
    liquid: Liquid | None= None  # Liquid object
    mass_liquid: float = 0  # kg
    liquid_temp: float | None = None  # K
    mass_gas: float = field(init=False)
    def __post_init__(self):
        if self.gas is not None:
            if self.gas_temp is None:
                raise ValueError("gas_temp must be provided if gas is specified.")
            R = self.gas.R
            T = self.gas_temp
            V = self.volume
            P = self.pressure

            liquid_volume = 0.0
            if self.mass_liquid is not None and self.liquid is not None:
                liquid_volume = self.mass_liquid / self.liquid.density
            
            self.mass_gas = (P * (V - liquid_volume)) / (R * T)
        else:
            self.mass_gas = 0.0

        if self.liquid is not None:
            if self.liquid_temp and self.gas_temp is None:
                raise ValueError("liquid_temp must be provided if liquid is specified and temperature cannot be assumed from gas_temp.")
            
            if self.liquid_temp is None:
                self.liquid_temp = self.gas_temp
                
            if self.mass_liquid is None:
                raise ValueError("mass_liquid must be provided if liquid is specified.")

@dataclass
class FlowComponent:
    name: str
    # kv can be a simple float OR a function that returns a float
    # The function signature is flexible: f(t, P_up, P_down)
    kv: Union[FloatOrArray, Callable, FloatOrArray]

    def get_kv(self, t: FloatOrArray, P_up: FloatOrArray, P_down: FloatOrArray, rho: Union[FloatOrArray, None], mu: Union[FloatOrArray, None], temp: Union[FloatOrArray, None], fluid: Union[Gas, Liquid, None]) -> FloatOrArray:
        """
        Returns the Kv value. 
        If inputs are arrays, returns an array. 
        If inputs are scalars, returns a float (usually).
        """
        if callable(self.kv):
            # The lambda handles the logic
            args = (t, P_up, P_down, rho, mu, temp, fluid)
            return self.kv(args)
        else:
            # If kv is a constant (e.g., a pipe), just return it.
            # Note: If you multiply this scalar return value by an array 
            # downstream (e.g. flow = kv * sqrt(dp_array)), 
            # Numpy broadcasting handles it automatically.
            return self.kv
        
class TubeComponent(FlowComponent):
    def __init__(self, name: str, D: float, L: float, roughness: float, bend_ang: float = 0, K_extra: float = 0): 
        super().__init__(name, kv=lambda args: tube_kv(args, D, L, roughness, bend_ang, K_extra))
        self.D = D
        self.L= L
        self.roughness = roughness
        self.bend_ang = bend_ang
        self.K_extra = K_extra
    

def timed_valve_kv(args, maxKv, t_open, t_close, t_ramp, leak_kv=1e-6):
    """
    Single function for a full valve cycle using Smooth Sigmoids (S-Curves).
    Best for ODE solvers (scipy.solve_ivp) as it has continuous derivatives.
    
    Parameters:
        t       : Current time (scalar or array)
        maxKv   : Maximum Kv
        t_open  : Start time for OPENING
        t_close : Start time for CLOSING
        t_ramp  : Time duration for the transition (approximate)
        leak    : Minimum Kv value (to prevent solver singularities)
    """
    t = args[0]
    k = 12.0 / t_ramp
    center_open = t_open + (t_ramp / 2.0)
    center_close = t_close + (t_ramp / 2.0)
    
    # Opening: standard sigmoid
    sig_open = 1.0 / (1.0 + math.exp(-k * (t - center_open)))
    
    # Closing: inverted sigmoid
    sig_close = 1.0 - (1.0 / (1.0 + math.exp(-k * (t - center_close))))
    
    Kv = maxKv * sig_open * sig_close
    return max(Kv, leak_kv)

def regulator_kv(args, set_pressure, reg_constant=300, leak_kv=1e-6):
    """Calculates the Kv of a pressure regulator based on upstream and downstream pressures.

    Parameters:
        P_up        : Upstream pressure
        P_down      : Downstream pressure
        set_pressure: Set pressure of the regulator
        reg_constant: Regulator constant (default: 300)
        leak_kv     : Minimum Kv value to prevent singularities (default: 1e-8)

    Huge assumptions are made:
        - Regulator only allows flow from high to low pressure
        - Linear relationship between pressure drop and Kv above set pressure
        - Below set pressure, only leak flow is allowed
        - No dynamic behavior (instantaneous response)
    Returns:
        reg_kv      : Calculated Kv value of the regulator
    """
    P_up, P_down = args[1], args[2]
    # P_up = float(P_up)
    # P_down = float(P_down)
    
    reg_dp = set_pressure - P_down
    # if reg_dp > 0 and P_up > set_pressure:
    if reg_dp > 0 and P_up > P_down:
        reg_kv = reg_dp / reg_constant / 1e5
    else:
        reg_kv = leak_kv
        
    return max(reg_kv, leak_kv)


def reg_kv_simple(args, set_pressure, max_kv, p_band, leak_kv=1e-6):
    P_up, P_down = args[1], args[2]
        # P_up = float(P_up)
        # P_down = float(P_down)
        
    error = set_pressure - P_down
    normalized_error = error / p_band
    opening_pct = math.tanh(normalized_error)
    
    kv = opening_pct * max_kv
    return max(kv, leak_kv)

@njit(cache=True)
def fast_tube_kv(P_up, P_down, rho, mu, D, L, roughness, bend_ang, K_extra):
    # Explicit Swamee-Jain (Turbulent)
    dp = abs(P_up - P_down)
    if dp < 1e-5: return 1e-6
    
    g = 9.81
    A = math.pi * (D/2)**2
    
    # K to Length conversion (assuming f=0.025)
    K_minor = (bend_ang / 90.0) * 0.35 + K_extra
    L_eff = L + (K_minor * D / 0.025)

    h_f = dp / (rho * g)
    
    term1 = roughness / (3.7 * D)
    # 2.51 * nu / (D * sqrt(2gD * hf/L))
    # nu = mu/rho
    term2 = (2.51 * (mu/rho) * math.sqrt(L_eff)) / (math.sqrt(2 * g * D**3 * h_f) + 1e-9)

    v = -2.0 * math.sqrt(2 * g * D * h_f / L_eff) * math.log10(term1 + term2)
    
    # v to Kv
    Q_m3h = v * A * 3600.0
    dp_bar = dp * 1e-5
    sg = rho * 0.001
    
    return Q_m3h * math.sqrt(sg / dp_bar)

def tube_kv(args, D, L, roughness, bend_ang, K_extra):
    """
    Calculates Tube Kv assuming fully turbulent flow.
    NON-ITERATIVE: Uses the explicit Swamee-Jain solution for velocity.
    """
    P_up, P_down, rho, mu = args[1], args[2], args[3], args[4]
    # 1. Physics Setup
    dp = abs(P_up - P_down)
    if dp < 1e-5: return 1e-6 # Protect against zero div
    
    g = 9.81
    A = math.pi * (D/2)**2
    nu = mu / rho # Kinematic Viscosity
    
    # 2. Equivalent Length Method
    # Convert bends/tees (K) into extra length of pipe (L_equiv).
    # We assume a reference friction factor f=0.025 for this conversion,
    # which is standard practice when exact f is unknown.
    K_minor = (bend_ang / 90.0) * 0.35 + K_extra
    L_eff = L + (K_minor * D / 0.025)

    # 3. Explicit Solve for Velocity (Swamee-Jain)
    # Head Loss h_f [m]
    h_f = dp / (rho * g)
    
    # Term A: Relative Roughness
    term_roughness = roughness / (3.7 * D)
    
    # Term B: Viscosity / High-Pressure Correction
    # (2.51 * nu) / (D * sqrt(2 * g * D * h_f / L_eff)) simplified:
    term_viscosity = (2.51 * nu * math.sqrt(L_eff)) / (math.sqrt(2 * g * D**3 * h_f) + 1e-9)

    # The Formula
    v = -2.0 * math.sqrt(2 * g * D * h_f / L_eff) * math.log10(term_roughness + term_viscosity)

    # 4. Convert to Kv
    Q_m3h = v * A * 3600.0
    dp_bar = dp / 1e5
    sg = rho / 1000.0
    
    if dp_bar <= 0 or sg <= 0: return 1e-6
    
    kv = Q_m3h * math.sqrt(sg / dp_bar)
    
    return kv


def nozzle_kv(area, discharge_coeff=0.98):
    # Kv = flow of 1000rho at 1 bar dp
    # Q = A * v
    # v = Cd * sqrt(2*dp/rho)
    # sqrt(2*dp/rho) at 1 bar, 1000kg/m3 = 14.142m/s
    # 14.142 * 3600 = 50911.8
    return 50900 * area * discharge_coeff



@dataclass(eq=False)
class FluidNode:
    name: str
    volume: float = 1e-6
    pressure: float | None = None
    temperature: float| None = 300.0
    fluid: Gas | Liquid | None = None
    tank: Tank | None = None
    constant_pressure: bool = False
    nodeComponentTuples: list[tuple["FluidNode", FlowComponent, bool]] | None = None # (Connected Node, Component, component_is_upstream)

    mass: float = field(init=False)

    def connect_nodes(self, nodeComponentTuples: list[tuple["FluidNode", FlowComponent, bool]]):
        if self.nodeComponentTuples is None:
            self.nodeComponentTuples = []
        self.nodeComponentTuples.extend(nodeComponentTuples)

    def initialize_state(self):
            if self.tank: # If attached to a tank, take tank properties
                self.pressure = self.tank.pressure
                self.mass = self.tank.mass_gas
                self.volume = self.tank.volume
            else:
                # P * V = m * R * T  =>  m = P*V / R*T
                self.mass = (self.pressure * self.volume) / (self.fluid.R * self.temperature)

def assign_tube_volumes(nodes):
    """
    Estimates volume for intermediate nodes based on connected components.
    Approximation: Each node gets half the volume of the tubes connected to it.
    """
    for node in nodes:
        # If user manually set a large volume (Tank), skip
        if node.volume > 1e-4: continue
        node.volume = 5.0e-4
        
        # vol = 0.0
        # count = 0
        # for _, comp, _ in node.nodeComponentTuples:
        #     if isinstance(comp, TubeComponent):
        #         # V = Area * Length
        #         r = comp.D / 2.0
        #         v_tube = math.pi * (r**2) * comp.L
        #         vol += v_tube * 0.5 # Assign half to this node, half to neighbor
        #         count += 1
        
        # # If no volume found (e.g. just a valve), keep default tiny volume
        # if vol > 1e-9:
        #     node.volume = vol

from scipy.optimize import least_squares    

def solve_network_pressures(all_nodes: list[FluidNode], t: float):


    sorted_nodes = sorted(all_nodes, key=lambda n: n.pressure if n.pressure is not None else -np.inf, reverse=True)

    for node in sorted_nodes:
        if node.tank or node.constant_pressure:
            continue

        best_neihbor = None
        max_p = -np.inf

        for neighbor, _, _ in node.nodeComponentTuples:
            if neighbor.pressure is not None and neighbor.pressure > max_p:
                max_p = neighbor.pressure
                best_neihbor = neighbor

        if best_neihbor and best_neihbor.fluid:
            node.fluid = best_neihbor.fluid
            if node.temperature is None:
                node.temperature = best_neihbor.temperature
            if node.pressure is None:
                node.pressure = best_neihbor.pressure / 2
                
    # if iterated == 1:
    #     for node in all_nodes:
    #         print(f"Initial pressure guess for node '{node.name}': {node.pressure}")


    variable_nodes = [node for node in all_nodes if node.tank is None and not node.constant_pressure]

    
    # print(f"Solving pressures at time {t:.2f}s for nodes {[node.name for node in variable_nodes]}")

    if not variable_nodes:
        return
    
    # Initial guess for pressures, set undefined pressures to 1 bar
    # x0 = np.array([node.pressure if node.pressure is not None and not np.isnan(node.pressure) else 1e5 for node in variable_nodes])
    x0 = np.array([node.pressure for node in variable_nodes])

    def residuals(pressures):
        
        # for node in all_nodes:
        #     print(f"Node '{node.name}': Pressure={node.pressure}, Fluid={node.fluid}")

        for i, node in enumerate(variable_nodes):
            node.pressure = pressures[i]

        net_flows = []

        for node in variable_nodes:
            net_m = 0.0
            
            if node.nodeComponentTuples is None:
                raise ValueError(f"FluidNode '{node}' has no connected components.")
            
            for connected_node, component, component_is_upstream in node.nodeComponentTuples:
                if component_is_upstream:
                    m_dot, _ = get_component_flows(t, connected_node, node, component)
                    if np.isnan(m_dot): m_dot = 0.0
                    # m_dot = float(m_dot)
                    net_m += m_dot
                else :
                    m_dot, _ = get_component_flows(t, node, connected_node, component)
                    if np.isnan(m_dot): m_dot = 0.0
                    # m_dot = float(m_dot)
                    net_m -= m_dot
            net_flows.append(net_m)

            # Bring flows to 1e5 scale for numerical stability
            # net_flows = [flow * 1e5 for flow in net_flows]
            # net_flows = [flow / 1e2 for flow in net_flows]
        
        # print(f"At time {t:.2f}s, Residuals: {net_flows}")
        # if iterated == 1:
        #     for i, node in enumerate(variable_nodes):
        #         print(f"At time {t:.2f}s, Node '{node.name}': Attempted Pressure={pressures[i]}, Net Flow={net_flows[i]}")
        # print(f"Attemped pressures: {[node.pressure for node in variable_nodes]}")
        return np.array(net_flows)
    
    # sol = root(residuals, x0, method='hybr', tol=1e-8)
    bounds_limits = (np.full_like(x0, 1e3), np.full_like(x0, 1e8))  # Example bounds: 1 kPa to 100 MPa
    sol = least_squares(residuals, x0, bounds=bounds_limits, max_nfev=10000)
    if not sol.success:
        print(f"Warning: Pressure solver did not converge at time {t}s. Message: {sol.message}")

    for i, node in enumerate(variable_nodes):
        node.pressure = sol.x[i]

        


    

# def get_component_flows(t, node_up: FluidNode, node_down: FluidNode, component: FlowComponent):

#     P_up = node_up.pressure
#     P_down = node_down.pressure

#     if P_up is None or P_down is None:
#         raise ValueError("Both upstream and downstream pressures must be defined.")

#     fluid_higherpressure = node_up.fluid if P_up >= P_down else node_down.fluid
#     temp_higherpressure = node_up.temperature if P_up >= P_down else node_down.temperature
#     if isinstance(fluid_higherpressure, Gas):
#         if node_up.temperature is None or fluid_higherpressure is None:
#             kv = component.get_kv(t, P_up, P_down, None, None, None, None)
#         else:
#             if node_up.pressure is None: raise ValueError(f"Upstream pressure is None for node '{node_up.name}' when calculating gas flow.")
#             rho = node_up.pressure / (fluid_higherpressure.R * node_up.temperature)
#             kv = component.get_kv(t, P_up, P_down, rho, fluid_higherpressure.viscosity, node_up.temperature, fluid_higherpressure)
#     else:
#         if temp_higherpressure is None or fluid_higherpressure is None:
#             kv = component.get_kv(t, P_up, P_down, None, None, None, None)
#         else:
#             kv = component.get_kv(t, P_up, P_down, fluid_higherpressure.density, fluid_higherpressure.viscosity, temp_higherpressure, fluid_higherpressure)


#     # print(f"Node_up fluid: {node_up.fluid}, Node_down fluid: {node_down.fluid}, Fluid higher pressure: {fluid_higherpressure}")
#     # print(f"Component '{component.name}': P_up={P_up}, P_down={P_down}, kv={kv}, fluid={fluid_higherpressure}")

#     # NOTE This check fails the solver
#     # if P_up < 0 or P_down < 0:
#     #     raise ValueError(f"Negative pressure encountered: P_up={P_up}, P_down={P_down} on component '{component.name}'")

#     try:
#         if isinstance(fluid_higherpressure, Gas):
#             T_higherpressure = node_up.temperature if P_up >= P_down else node_down.temperature
#             m_dot = gasFlowrate(kv, P_up, P_down, T_higherpressure, fluid_higherpressure)
#             if np.isnan(m_dot):
#                 raise ValueError(f"Calculated NaN mass flow rate for component '{component.name}' with P_up={P_up}, P_down={P_down}, T_higherpressure={T_higherpressure}, kv={kv}, fluid={fluid_higherpressure}")
#         else:
#             m_dot = liquidFlowrate(kv, P_up, P_down, fluid_higherpressure)
#     except AttributeError as e:
#         raise ValueError \
#             (f"No fluid properties defined for component '{component.name}': {e}. Node up name: {node_up.name}, Node down name: {node_down.name}. Node_up fluid: {node_up.fluid}, Node_down fluid: {node_down.fluid} with pressures P_up={P_up}, P_down={P_down}")

#     return m_dot, fluid_higherpressure

def get_component_flows(t, node_up, node_down, component):
    P_up = node_up.pressure
    P_down = node_down.pressure

    # Optimize attribute access
    if P_up >= P_down:
        fluid = node_up.fluid
        T = node_up.temperature
    else:
        fluid = node_down.fluid
        T = node_down.temperature

    # --- FAST PATH ---
    # Check "R" attribute to distinguish Gas vs Liquid without slow isinstance
    if hasattr(fluid, 'R'): 
        # GAS LOGIC
        rho = P_up / (fluid.R * T) if P_up >= P_down else P_down / (fluid.R * T)
        
        # Check if it's a Tube (has 'D' attribute) to run the expensive calculation
        # Otherwise it's a constant/simple Kv component
        if hasattr(component, 'D'):
            kv = fast_tube_kv(P_up, P_down, rho, fluid.viscosity, 
                              component.D, component.L, component.roughness, 
                              component.bend_ang, component.K_extra)
        else:
            # Valve/Regulator/Nozzle logic (handled by lambda or direct value)
            # If it's a lambda, we still have Python overhead here unfortunately
            if callable(component.kv):
                # Unpack args manually for speed
                args = (t, P_up, P_down, rho, fluid.viscosity, T, fluid)
                kv = component.kv(args)
            else:
                kv = component.kv

        m_dot = fast_gas_flow(kv, P_up, P_down, T, fluid.R)
        
    else:
        # LIQUID LOGIC
        if hasattr(component, 'D'):
            kv = fast_tube_kv(P_up, P_down, fluid.density, fluid.viscosity, 
                              component.D, component.L, component.roughness, 
                              component.bend_ang, component.K_extra)
        else:
            if callable(component.kv):
                args = (t, P_up, P_down, fluid.density, fluid.viscosity, T, fluid)
                kv = component.kv(args)
            else:
                kv = component.kv
                
        m_dot = fast_liquid_flow(kv, P_up, P_down, fluid.density)

    return m_dot, fluid

def get_flow_rate(t, node_up, node_down, component):
    P_up = node_up.pressure
    P_down = node_down.pressure
    fluid = node_up.fluid # Assume single fluid type for simplicity
    T = node_up.temperature

    # Density at High Pressure side
    P_high = P_up if P_up > P_down else P_down
    rho = P_high / (fluid.R * T)

    # 1. Calculate Kv
    if isinstance(component, TubeComponent):
        kv = fast_tube_kv(P_up, P_down, rho, fluid.viscosity, 
                          component.D, component.L, component.roughness, 
                          component.bend_ang, component.K_extra)
    elif callable(component.kv):
        # Handle Regulator/Valve Lambda
        # args expected: (t, P_up, P_down)
        args = (t, P_up, P_down) 
        kv = component.kv(args)
    else:
        kv = component.kv

    # 2. Calculate Mass Flow
    m_dot = fast_gas_flow(kv, P_up, P_down, T, fluid.R)
    return m_dot

iterated = 0
def ode_system(t, y, nodes, node_map):
    global iterated
    iterated += 1
    if iterated % 1000 == 0:
        print(f"Time: {t:.2f} s", end='\r')
    dydt = np.zeros(len(nodes))
    
    # 1. Update Pressures from Mass (State)
    for i, node in enumerate(nodes):
        m = max(y[i], 1e-12) # Clamp to avoid vacuum crash
        node.mass = m
        if not node.constant_pressure:
            # P = mRT / V
            node.pressure = (m * node.fluid.R * node.temperature) / node.volume
    
    # 2. Calculate Flows
    # We iterate nodes, but we need to avoid double counting. 
    # Logic: Only calculate if I am the "Upstream" definition of the connection.
    
    for i, node in enumerate(nodes):
        for neighbor, comp, defined_as_upstream in node.nodeComponentTuples:
            
            # Only process this link if 'node' is the one defined as upstream in the setup
            # This ensures we calculate flow for the link A-B exactly once.
            if defined_as_upstream:
                # Calculate flow based on current pressures (Auto-handles reverse flow)
                m_dot = get_flow_rate(t, node, neighbor, comp)
                
                neighbor_idx = node_map[neighbor]
                
                # Apply Mass Balance
                # m_dot is positive if flowing Node -> Neighbor
                if not node.constant_pressure:
                    dydt[i] -= m_dot
                
                if not neighbor.constant_pressure:
                    dydt[neighbor_idx] += m_dot
                    
    return dydt

def dae_system(t, y, total_nodes, tanks):
    masses = y
    print(f"Time: {t:.2f} s, masses: {masses}", end='\r')


    # Update tank masses
    for i, tank in enumerate(tanks):
        mass_gas = max(masses[2*i], 1e-10)
        mass_liquid = max(masses[2*i + 1], 0.0)

        volume_liquid = 0

        if tank.liquid is not None:
            volume_liquid = mass_liquid / tank.liquid.density

        if tank.gas is not None:
            volume_gas = tank.volume - volume_liquid
            density_gas = mass_gas / volume_gas

            tank.gas_temp = tank.gas_temp  # Assuming constant temperature for simplicity, later add isentropic and heat transfers
            tank.pressure = density_gas * tank.gas.R * tank.gas_temp


        tank.mass_gas = mass_gas
        if tank.liquid is not None:
            tank.mass_liquid = mass_liquid

    for node in total_nodes:
        if node.tank:
            node.pressure = node.tank.pressure
            node.temperature = node.tank.gas_temp if node.tank.gas is not None else None

            if node.tank.liquid and node.tank.mass_liquid > 0:
                node.fluid = node.tank.liquid
            else:
                node.fluid = node.tank.gas

    # Calculate intermediete pressures

    solve_network_pressures(total_nodes, t)

    # Calculate mass flow rates for each tank

    dydt = []

    for tank in tanks:
        # print(f"Iteration {iterated}, Time {t:.2f}s, Tank '{tank.name}': P={tank.pressure/1e5:.2f} bar, m_gas={tank.mass_gas:.4f} kg, m_liq={tank.mass_liquid:.4f} kg")

        d_gas = 0.0
        d_liq = 0.0

        tank_node = next(node for node in total_nodes if node.tank == tank)

        if tank_node.nodeComponentTuples is None:
            raise ValueError(f"FluidNode for tank '{tank.name}' has no connected components.")


        for connected_node, component, component_is_upstream in tank_node.nodeComponentTuples:
            if component_is_upstream:
                m_dot, fluid = get_component_flows(t, connected_node, tank_node, component)
                # print(f"Component '{component.name}' upstream to tank '{tank.name}': m_dot={m_dot} kg/s, fluid={fluid}, Upstream Node: '{connected_node.name}' pressure={connected_node.pressure}, Downstream Node: '{tank_node.name}' pressure={tank_node.pressure}")
                # m_dot = float(m_dot)
                if isinstance(fluid, Gas):
                    d_gas += m_dot
                else:
                    d_liq += m_dot
            else:
                m_dot, fluid = get_component_flows(t, tank_node, connected_node, component)
                # print(f"Component '{component.name}' downstream to tank '{tank.name}': m_dot={m_dot} kg/s, fluid={fluid}, Upstream Node: '{tank_node.name}' pressure={tank_node.pressure}, Downstream Node: '{connected_node.name}' pressure={connected_node.pressure}")
                # m_dot = float(m_dot)
                if isinstance(fluid, Gas):
                    d_gas -= m_dot
                else:
                    d_liq -= m_dot


        dydt.append(d_gas)
        dydt.append(d_liq)

    # print(dydt)
    return dydt

def initialize_tanks(tanks: List[Tank]):
    initial_masses = np.array([])
    initial_temperatures = np.array([])
    for tank in tanks:
        initial_masses = np.append(initial_masses, tank.mass_gas)
        if tank.gas_temp is not None:
            initial_temperatures = np.append(initial_temperatures, tank.gas_temp)
        else:
            initial_temperatures = np.append(initial_temperatures, np.nan)

        if tank.liquid is not None:
            initial_masses = np.append(initial_masses, tank.mass_liquid)

            if tank.liquid_temp is not None:
                initial_temperatures = np.append(initial_temperatures, tank.liquid_temp)
            else:
                raise ValueError("Liquid temperature must be provided if liquid is specified.")
        else:
            initial_masses = np.append(initial_masses, 0.0)
        initial_temperatures = np.append(initial_temperatures, np.nan)

    return initial_masses, initial_temperatures

def print_pressure_ladder(nodes: List[FluidNode]):
    # import matplotlib.pyplot as plt

    sorted_nodes = sorted(nodes, key=lambda n: n.pressure if n.pressure is not None else -np.inf, reverse=True)

    for node in sorted_nodes:
        if node.pressure is not None:
            print(f"Node '{node.name}': Pressure={node.pressure/1e5:.2f} bar, Fluid={'Gas' if isinstance(node.fluid, Gas) else 'Liquid' if isinstance(node.fluid, Liquid) else 'None'}")
        else:
            print(f"Node '{node.name}': Pressure=Undefined, Fluid={'Gas' if isinstance(node.fluid, Gas) else 'Liquid' if isinstance(node.fluid, Liquid) else 'None'}")

        if node.nodeComponentTuples is not None:
            for connected_node, component, component_is_upstream in node.nodeComponentTuples:
                if not component_is_upstream:
                    break

                if isinstance(component, TubeComponent):
                    mdot, fluid_higherpressure = get_component_flows(0, connected_node, node, component) # This will print the flow and component details due to the print statements in get_component_flows
                    if isinstance(fluid_higherpressure, Gas):
                        if node.pressure is None or node.temperature is None: 
                            print(f"Cannot calculate density for node '{node.name}' due to undefined pressure or temperature.")
                        else:
                            density = node.temperature* fluid_higherpressure.R / node.pressure
                            v = mdot / (density * math.pi * (component.D/2)**2)
                            print(f"-> Tube '{component.name} from '{connected_node.name}' is carrying {fluid_higherpressure.name} at velocity {v:.2f} m/s")

                    elif isinstance(fluid_higherpressure, Liquid):
                        if node.pressure is None: 
                            print(f"Cannot calculate density for node '{node.name}' due to undefined pressure.")
                        else:
                            density = fluid_higherpressure.density
                            v = mdot / (density * math.pi * (component.D/2)**2)
                            print(f"-> Tube '{component.name} from '{connected_node.name}' is carrying {fluid_higherpressure.name} at velocity {v:.2f} m/s")




# Ensure you have your classes imported: FluidNode, Gas, Liquid, TubeComponent, etc.

def plot_pressure_network(nodes: list[FluidNode]):
    """
    Visualizes the pressure network as a Hydraulic Grade Line plot.
    Nodes are sorted by pressure to create a "Ladder" effect.
    Connecting tubes are drawn as arrows with velocity annotations.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    
    # 1. Sort nodes by pressure (High -> Low) to define the X-axis order
    #    Filter out undefined pressures
    valid_nodes = [n for n in nodes if n.pressure is not None]
    sorted_nodes = sorted(valid_nodes, key=lambda n: n.pressure, reverse=True)
    
    # Create a mapping of Node Object -> X-axis Index
    node_x_map = {node: i for i, node in enumerate(sorted_nodes)}
    node_names = [n.name for n in sorted_nodes]
    pressures_bar = [n.pressure / 1e5 for n in sorted_nodes]

    # --- PLOTTING SETUP ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 2. Plot the Nodes (Scatter points)
    #    Color code: Red for Gas, Blue for Liquid
    colors = []
    for n in sorted_nodes:
        if n.fluid and isinstance(n.fluid, Gas):
            colors.append('#d62728') # Tab:Red
        elif n.fluid and isinstance(n.fluid, Liquid):
            colors.append('#1f77b4') # Tab:Blue
        else:
            colors.append('gray')

    ax.scatter(range(len(sorted_nodes)), pressures_bar, color=colors, s=100, zorder=5)

    # 3. Draw Connections (Arrows)
    #    We iterate through every node and look for its DOWNSTREAM connections
    for node in valid_nodes:
        if not node.nodeComponentTuples:
            continue
            
        for neighbor, component, component_is_upstream_of_me in node.nodeComponentTuples:
            
            # We only draw lines for flow leaving "node" and going to "neighbor"
            # If component is upstream of ME, flow is coming IN. Skip it.
            if component_is_upstream_of_me:
                continue

            # Skip if neighbor has no pressure (can't plot)
            if neighbor not in node_x_map:
                continue

            # --- CALCULATE VELOCITY (Your Logic) ---
            velocity = 0.0
            
            # Get mass flow (assuming get_component_flows handles the physics)
            try:
                # Note: get_component_flows returns (mdot, fluid_object)
                # Since we are node -> neighbor, node is UP
                mdot, fluid_obj = get_component_flows(0, node, neighbor, component)
                
                # Check for Tube to calculate velocity
                # (Assuming 'D' is an attribute of the component, like in TubeComponent)
                if hasattr(component, 'D') and component.D > 0:
                    rho = 0.0
                    if isinstance(fluid_obj, Gas):
                        if node.pressure and node.temperature:
                            rho = node.pressure / (fluid_obj.R * node.temperature)
                    elif isinstance(fluid_obj, Liquid):
                        rho = fluid_obj.density
                    
                    if rho > 0:
                        area = math.pi * (component.D / 2)**2
                        velocity = mdot / (rho * area)

            except Exception as e:
                print(f"Viz Warning: Could not calc flow for {component.name}: {e}")
                velocity = 0.0

            # --- DRAW ARROW ---
            start_x = node_x_map[node]
            end_x = node_x_map[neighbor]
            start_y = node.pressure / 1e5
            end_y = neighbor.pressure / 1e5
            
            # Draw line
            ax.annotate("", 
                        xy=(end_x, end_y), 
                        xytext=(start_x, start_y), 
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5, shrinkA=10, shrinkB=10))

            # Add Velocity Label (Midpoint)
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # Offset text slightly to avoid overlapping the line
            ax.text(mid_x, mid_y, f"P={node.pressure/1e5:.2f} bar\n{velocity:.1f} m/s, {mdot*1000:.1f} g/s\n({component.name})", 
                    fontsize=8, ha='center', va='bottom', color='darkgreen', rotation=0, 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))

    # 4. Formatting
    ax.set_xticks(range(len(sorted_nodes)))
    ax.set_xticklabels(node_names, rotation=45, ha='right')
    ax.set_ylabel("Pressure (Bar)")
    ax.set_title("System Pressure Gradient & Flow Velocities")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='Gas Node', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Liquid Node', markersize=10),
        Line2D([0], [0], color='black', lw=1.5, label='Flow Path')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    # make plot 5 tall 5 wide
    # plt.gcf().set_size_inches(25, 25)
    plt.show()

def create_series_nodes(upstream_node: FluidNode, components: list[FlowComponent], downstream_node: FluidNode):
    """
    Creates intermediate nodes between components in series.
    [Comp1, Comp2, Comp3] -> Upstream --(C1)--> Node1 --(C2)--> Node2 --(C3)--> Downstream
    """
    intermediate_nodes = []
    current_node = upstream_node

    for i, component in enumerate(components):
        if i == len(components) - 1:
            target_node = downstream_node
        else:
            target_node = FluidNode(name=f"Node_{i}_{component.name}")
            intermediate_nodes.append(target_node)
        current_node.connect_nodes([(target_node, component, False)])
        target_node.connect_nodes([(current_node, component, True)])
        current_node = target_node

    return intermediate_nodes