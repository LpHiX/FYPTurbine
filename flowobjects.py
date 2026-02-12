from platform import node
from unicodedata import name
from narwhals import Unknown
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

def gasFlowrate(Kv, P_1, P_2, T_1, gas, assumeChoked=False):
    """
    Calculates gas mass flow rate using the Simplified Kv Model (Tameson).
    Includes broadcasting to handle mix of scalar and array inputs.
    """
    
    if np.isscalar(P_1) and np.isscalar(P_2) and np.isscalar(Kv) and np.isscalar(T_1):
            P_1 = float(P_1)
            P_2 = float(P_2)
            Kv = float(Kv)
            T_1 = float(T_1)
            
            # Constants for Standard Conditions (NTP: 0°C, 1 atm)
            P_N_Pa = 101325.0       # Pa
            T_N_K  = 273.15         # K
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
    
    # --- BROADCASTING FIX ---
    # This ensures that if Kv is [1] and P_1 is length 5, Kv becomes length 5
    try:
        P_1, P_2, T_1, Kv = np.broadcast_arrays(P_1, P_2, T_1, Kv)
    except ValueError:
        raise ValueError("Input shapes (P, T, Kv) are incompatible and cannot be broadcast together.")

    # Constants for Standard Conditions (NTP: 0°C, 1 atm)
    P_N_Pa = 101325.0       # Pa
    T_N_K  = 273.15         # K
    rho_n  = P_N_Pa / (gas.R * T_N_K) # Normal Density (kg/m^3)

    # Convert Pressures to BAR for the Kv formula
    p1_bar = P_1 / 1e5
    p2_bar = P_2 / 1e5

    # --- 2. Flow Direction Logic ---
    # Identify reverse flow (P2 > P1) and swap variables for calculation
    reverse_mask = p2_bar > p1_bar
    
    pu = np.where(reverse_mask, p2_bar, p1_bar) # Upstream Pressure (bar)
    pd = np.where(reverse_mask, p1_bar, p2_bar) # Downstream Pressure (bar)

    # --- 3. Choked Flow Determination ---
    # Simplified Model assumes choking at P_down < P_up / 2
    critical_ratio = 0.5
    
    # Handle assumeChoked (scalar or array)
    assumeChoked = np.atleast_1d(assumeChoked)
    # Broadcast assumeChoked to shape of P_1 if necessary, or let numpy broadcasting handle the 'or' logic
    
    # Determine if flow is choked (physically choked OR forced by user)
    # Note: We use bitwise OR (|) for boolean arrays
    is_choked = (pd < (pu * critical_ratio)) | ((assumeChoked == True) | (assumeChoked == 1))

    # --- 4. Flow Rate Calculation (Q_n in Nm^3/h) ---
    q_n = np.zeros_like(pu)

    # A. Subcritical Flow (P_down > P_up / 2)
    sub_idx = ~is_choked
    if np.any(sub_idx):
        dp_sub = pu[sub_idx] - pd[sub_idx]
        # Protect against negative sqrt near zero
        term = np.maximum(dp_sub * pd[sub_idx], 0) 
        
        q_n[sub_idx] = 514 * Kv[sub_idx] * np.sqrt(
            term / (rho_n * T_1[sub_idx])
        )

    # B. Supercritical / Choked Flow (P_down < P_up / 2)
    choked_idx = is_choked
    if np.any(choked_idx):
        q_n[choked_idx] = 257 * Kv[choked_idx] * pu[choked_idx] * np.sqrt(
            1.0 / (rho_n * T_1[choked_idx])
        )

    # --- 5. Final Conversion to Mass Flow (kg/s) ---
    m_dot = q_n * rho_n / 3600.0

    # Apply sign for reverse flow
    m_dot = np.where(reverse_mask, -m_dot, m_dot)

    return m_dot

import numpy as np

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
    # --- 1. Broadcast Inputs ---
    # Ensure inputs match shapes (handles scalar Kv with array Pressure)

        # --- OPTIMIZATION: Scalar Fast Path ---
    if np.isscalar(P_1) and np.isscalar(P_2) and np.isscalar(Kv):
        P_1 = float(P_1)
        P_2 = float(P_2)
        Kv = float(Kv)
        
        dp = P_1 - P_2
        CONST_LIQUID = 1.0 / 36000.0
        
        if dp >= 0:
            return CONST_LIQUID * Kv * math.sqrt(dp * liquid.density)
        else:
            return -CONST_LIQUID * Kv * math.sqrt(-dp * liquid.density)

    P_1 = np.atleast_1d(P_1).astype(float)
    P_2 = np.atleast_1d(P_2).astype(float)
    Kv  = np.atleast_1d(Kv).astype(float)
    
    try:
        P_1, P_2, Kv = np.broadcast_arrays(P_1, P_2, Kv)
    except ValueError:
        raise ValueError("Input shapes are incompatible.")

    # --- 2. Calculate Pressure Drop ---
    dp = P_1 - P_2
    
    # --- 3. Calculate Mass Flow ---
    # Formula: m_dot = (Kv / 36000) * sqrt(rho * abs(dp))
    # We use np.sign(dp) to handle reverse flow direction automatically
    
    # Constant derived from: 0.1 (unit conversions) / 3600 (hours to seconds)
    CONST_LIQUID = 1 / 36000.0
    
    m_dot = (np.sign(dp) * CONST_LIQUID * Kv * np.sqrt(np.abs(dp) * liquid.density))

    return m_dot

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
    kv: Union[FloatOrArray, Callable[[FloatOrArray, FloatOrArray, FloatOrArray, Union[FloatOrArray, None], Union[FloatOrArray, None]], FloatOrArray]]

    def get_kv(self, t: FloatOrArray, P_up: FloatOrArray, P_down: FloatOrArray, rho: Union[FloatOrArray, None], mu: Union[FloatOrArray, None]) -> FloatOrArray:
        """
        Returns the Kv value. 
        If inputs are arrays, returns an array. 
        If inputs are scalars, returns a float (usually).
        """
        if callable(self.kv):
            # The lambda handles the logic
            return self.kv(t, P_up, P_down, rho, mu)
        else:
            # If kv is a constant (e.g., a pipe), just return it.
            # Note: If you multiply this scalar return value by an array 
            # downstream (e.g. flow = kv * sqrt(dp_array)), 
            # Numpy broadcasting handles it automatically.
            return self.kv
        
class TubeComponent(FlowComponent):
    def __init__(self, name: str, D: float, L: float, roughness: float, bend_ang: float, K_extra: float): 
        super().__init__(name, kv=lambda t, P_up, P_down, rho, mu: tube_kv(P_up, P_down, rho, mu, D, L, roughness, bend_ang, K_extra))
        self.D = D
        self.L= L
        self.roughness = roughness
        self.bend_ang = bend_ang
        self.K_extra = K_extra
    

def timed_valve_kv(t, maxKv, t_open, t_close, t_ramp, leak_kv=1e-6):
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
    # 1. Standardize input
    if np.isscalar(t):
        t = float(t)
        k = 12.0 / t_ramp
        center_open = t_open + (t_ramp / 2.0)
        center_close = t_close + (t_ramp / 2.0)
        
        # Opening: standard sigmoid
        sig_open = 1.0 / (1.0 + math.exp(-k * (t - center_open)))
        
        # Closing: inverted sigmoid
        sig_close = 1.0 - (1.0 / (1.0 + math.exp(-k * (t - center_close))))
        
        Kv = maxKv * sig_open * sig_close
        return max(Kv, leak_kv)

    # 1. Standardize input
    is_scalar = np.isscalar(t) or np.ndim(t) == 0


    is_scalar = np.isscalar(t) or np.ndim(t) == 0
    t = np.atleast_1d(t).astype(float)
    
    # 2. Calculate Steepness (k)
    # A 'k' factor of 12/t_ramp ensures the curve completes ~99% of its 
    # transition within the t_ramp window.
    k = 12.0 / t_ramp
    
    # 3. Calculate Centers
    # We center the S-curve at the midpoint of the ramp
    center_open = t_open + (t_ramp / 2.0)
    center_close = t_close + (t_ramp / 2.0)
    
    # 4. Generate Curves (0.0 to 1.0)
    # Opening: standard sigmoid (low -> high)
    # Equation: 1 / (1 + e^-k(t - center))
    sig_open = 1.0 / (1.0 + np.exp(-k * (t - center_open)))
    
    # Closing: inverted sigmoid (high -> low)
    # Equation: 1 - sigmoid
    sig_close = 1.0 - (1.0 / (1.0 + np.exp(-k * (t - center_close))))
    
    # 5. Combine
    # We multiply them. If either is 0 (closed), the result is 0.
    # This naturally creates a smooth bell-shape if the valve closes 
    # before it finishes opening.
    Kv = maxKv * sig_open * sig_close
    
    # 6. Apply Leak (Clip)
    Kv = np.maximum(Kv, leak_kv)
    
    if is_scalar:
        return Kv.item()
    return Kv

def regulator_kv(P_up, P_down, set_pressure, reg_constant=300, leak_kv=1e-6):
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
    if np.isscalar(P_up) and np.isscalar(P_down):
        P_up = float(P_up)
        P_down = float(P_down)
        
        reg_dp = set_pressure - P_down
        # if reg_dp > 0 and P_up > set_pressure:
        if reg_dp > 0 and P_up > P_down:
            reg_kv = reg_dp / reg_constant / 1e5
        else:
            reg_kv = leak_kv
            
        return max(reg_kv, leak_kv)

    is_scalar = np.isscalar(P_up) or np.ndim(P_up) == 0
    P_up = np.atleast_1d(P_up).astype(float)
    P_down = np.atleast_1d(P_down).astype(float)

    reg_dp = set_pressure - P_down
    # reg_kv = np.where((reg_dp > 0) & (P_up > set_pressure), reg_dp / reg_constant / 1e5, leak_kv)
    reg_kv = np.where((reg_dp > 0) & (P_up > P_down), reg_dp / reg_constant / 1e5, leak_kv)
    reg_kv = np.maximum(reg_kv, leak_kv)
    if is_scalar:
        return reg_kv.item()
    return reg_kv

def tube_kv(P_up, P_down, rho, mu, D, L, roughness, bend_ang, K_extra):
    dp = abs(P_up - P_down)
    if dp < 1e-4:
        return 1e-6 # Return a tiny leak if no pressure drop to avoid singularity

    A = math.pi * (D/2)**2
    K_minor = (bend_ang / 90.0) * 0.35 + K_extra # 0.35 is a typical K for a 90 degree bend, scaled by angle, should be from Crane TN410, but need to check

    v = 5.0
    rel_diff = 1
    iterations = 0

    while rel_diff > 1e-4:

        Re = rho * v * D / mu

        if Re < 1e-3: Re = 1e-3

        f_lam = 64.0 / Re
        f_turb = 0.25 / (math.log10((roughness/(3.7*D)) + (5.74/(Re**0.9))))**2

        if Re < 2000: f = f_lam
        elif Re > 4000: f = f_turb
        else: f = f_lam + (f_turb - f_lam) * ((Re - 2000) / 2000)


        K_total= K_minor + f * (L / D)

        v_new = math.sqrt(2 * abs(dp) / (rho * K_total))
        rel_diff = abs(v_new - v) / v

        # print(f"tube_kv iteration {iterations}: P_up={P_up}, P_down={P_down}, rho={rho}, mu={mu}, D={D}, L={L}, roughness={roughness}, bend_ang={bend_ang}, K_extra={K_extra}, Re={Re:.2e}, v={v:.4f} m/s, f={f:.4e}, K_total={K_total:.4f}")
        
        v = v_new
        # v = 0.5 * v + 0.5 * v_new
        
        if iterations > 100:
            print(f"Warning: tube_kv did not converge after 100 iterations. Returning last Kv value. P_up={P_up}, P_down={P_down}, rho={rho}, mu={mu}, D={D}, L={L}, roughness={roughness}, bend_ang={bend_ang}, K_extra={K_extra}, Re={Re}, v={v}, f={f}, K_total={K_total}")
            break

        # Damping to ensure convergence
        iterations += 1

    # print(iterations)
    Q_m3h = v * A * 3600.0
    dp_bar = abs(dp) / 1e5
    sg = rho / 1000.0
    kv = Q_m3h * math.sqrt(sg / dp_bar)
    return kv

# def tube_kv(P_up, P_down, rho, mu, D, L, roughness, bend_ang, K_extra): 
#     dp = P_up - P_down
#     if abs(dp) < 1e-4: return 1e-6
    
#     # Constants
#     g = 9.81
#     nu = mu / rho
#     A = math.pi * (D/2)**2
    
#     # 1. Calculate Head Loss (h_f)
#     # We must account for minor losses (K_minor) which makes it tricky.
#     # Standard Swamee-Jain assumes ONLY pipe friction.
#     # If K_minor is large, this approximation fails. 
#     # If K_minor is small, we can just add an equivalent length: L_eq = L + (K_minor * D / 0.02)
    
#     K_minor = (bend_ang / 90.0) * 0.35 + K_extra
#     L_equiv = L + (K_minor * D / 0.02) # approx friction factor 0.02 for converting K to L
    
#     h_f = abs(dp) / (rho * g)
    
#     # 2. Check Laminar Regime Limit (Re ~ 2000)
#     # Laminar flow: Q = (pi * D^4 * dp) / (128 * mu * L)
#     # v_lam = Q / A
#     v_lam = (abs(dp) * D**2) / (32 * mu * L_equiv)
    
#     # 3. Turbulent Explicit Calculation (Swamee-Jain inversed)
#     term1 = roughness / (3.7 * D)
#     term2 = (2.51 * nu * L_equiv**0.5) / ((2 * g * D**3 * h_f)**0.5)
    
#     v_turb = -2 * math.sqrt(2 * g * D * h_f / L_equiv) * math.log10(term1 + term2)
    
#     # 4. Blend (Simple Min/Max logic usually works for speed)
#     # If laminar velocity is lower than turbulent prediction, we are likely laminar.
#     v = (v_lam**-2 + v_turb**-2)**(-0.5) # Harmonic mean to blend velocities

#     # 5. Convert to Kv
#     Q_m3h = v * A * 3600.0
#     dp_bar = abs(dp) / 1e5
#     sg = rho / 1000.0
    
#     if dp_bar <= 0 or sg <= 0: return 1e-6
    
#     return Q_m3h * math.sqrt(sg / dp_bar)

@dataclass(eq=False)
class FluidNode:
    name: str
    pressure: float | None = None
    temperature: float| None = None
    fluid: Gas | Liquid | None = None
    tank: Tank | None = None
    constant_pressure: bool = False
    nodeComponentTuples: list[tuple["FluidNode", FlowComponent, bool]] | None = None # (Connected Node, Component, component_is_upstream)

    def connect_nodes(self, nodeComponentTuples: list[tuple["FluidNode", FlowComponent, bool]]):
        if self.nodeComponentTuples is None:
            self.nodeComponentTuples = []
        self.nodeComponentTuples.extend(nodeComponentTuples)

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
                    m_dot = float(m_dot)
                    net_m += m_dot
                else :
                    m_dot, _ = get_component_flows(t, node, connected_node, component)
                    if np.isnan(m_dot): m_dot = 0.0
                    m_dot = float(m_dot)
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

        


    

def get_component_flows(t, node_up: FluidNode, node_down: FluidNode, component: FlowComponent):

    P_up = node_up.pressure
    P_down = node_down.pressure

    if P_up is None or P_down is None:
        raise ValueError("Both upstream and downstream pressures must be defined.")

    fluid_higherpressure = node_up.fluid if P_up >= P_down else node_down.fluid
    temp_higherpressure = node_up.temperature if P_up >= P_down else node_down.temperature
    if isinstance(fluid_higherpressure, Gas):
        if node_up.temperature is None or fluid_higherpressure is None:
            kv = component.get_kv(t, P_up, P_down, None, None)
        else:
            if node_up.pressure is None: raise ValueError(f"Upstream pressure is None for node '{node_up.name}' when calculating gas flow.")
            rho = node_up.pressure / (fluid_higherpressure.R * node_up.temperature)
            kv = component.get_kv(t, P_up, P_down, rho, fluid_higherpressure.viscosity)
    else:
        if temp_higherpressure is None or fluid_higherpressure is None:
            kv = component.get_kv(t, P_up, P_down, None, None)
        else:
            kv = component.get_kv(t, P_up, P_down, fluid_higherpressure.density, fluid_higherpressure.viscosity)


    # print(f"Node_up fluid: {node_up.fluid}, Node_down fluid: {node_down.fluid}, Fluid higher pressure: {fluid_higherpressure}")
    # print(f"Component '{component.name}': P_up={P_up}, P_down={P_down}, kv={kv}, fluid={fluid_higherpressure}")

    # NOTE This check fails the solver
    # if P_up < 0 or P_down < 0:
    #     raise ValueError(f"Negative pressure encountered: P_up={P_up}, P_down={P_down} on component '{component.name}'")

    try:
        if isinstance(fluid_higherpressure, Gas):
            T_higherpressure = node_up.temperature if P_up >= P_down else node_down.temperature
            m_dot = gasFlowrate(kv, P_up, P_down, T_higherpressure, fluid_higherpressure)
            if np.isnan(m_dot):
                raise ValueError(f"Calculated NaN mass flow rate for component '{component.name}' with P_up={P_up}, P_down={P_down}, T_higherpressure={T_higherpressure}, kv={kv}, fluid={fluid_higherpressure}")
        else:
            m_dot = liquidFlowrate(kv, P_up, P_down, fluid_higherpressure)
    except AttributeError as e:
        raise ValueError \
            (f"No fluid properties defined for component '{component.name}': {e}. Node up name: {node_up.name}, Node down name: {node_down.name}. Node_up fluid: {node_up.fluid}, Node_down fluid: {node_down.fluid} with pressures P_up={P_up}, P_down={P_down}")

    return m_dot, fluid_higherpressure
        


def dae_system(t, y, total_nodes, tanks):
    # print(f"--- Iteration {iterated}, Time {t:.2f}s ---")
    masses = y

    # Update tank masses
    for i, tank in enumerate(tanks):
        mass_gas = masses[2*i]
        mass_liquid = masses[2*i + 1]

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
                m_dot = float(m_dot)
                if isinstance(fluid, Gas):
                    d_gas += m_dot
                else:
                    d_liq += m_dot
            else:
                m_dot, fluid = get_component_flows(t, tank_node, connected_node, component)
                # print(f"Component '{component.name}' downstream to tank '{tank.name}': m_dot={m_dot} kg/s, fluid={fluid}, Upstream Node: '{tank_node.name}' pressure={tank_node.pressure}, Downstream Node: '{connected_node.name}' pressure={connected_node.pressure}")
                m_dot = float(m_dot)
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
            ax.text(mid_x, mid_y, f"{velocity:.1f} m/s\n({component.name})", 
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