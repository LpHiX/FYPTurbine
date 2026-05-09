from rocketcea.cea_obj_w_units import CEA_Obj


class Engine:
    def __init__(self, ox: str, fuel: str):
        self.ox = ox
        self.fuel = fuel
        self.cea = CEA_Obj(
            oxName = self.ox,
            fuelName = self.fuel,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            # fac_CR=self.cr,
            make_debug_prints=False)
        
    def size(self, OF, pc, pe, thrust):
        eps = self.cea.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=pc/pe)
        _, cf, _ = self.cea.get_PambCf(Pamb=pe, Pc=pc, MR=OF, eps=eps)
        at = thrust / (cf * pc * 1e5)
        cstar = self.cea.get_Cstar(Pc=pc, MR=OF)
        mdot = pc * 1e5 * at / cstar
        isp_sea, _ = self.cea.estimate_Ambient_Isp(Pamb=pe, Pc=pc, MR=OF, eps=eps)
        return eps, cf, at, cstar, mdot, isp_sea