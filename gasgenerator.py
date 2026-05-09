from rocketcea.cea_obj_w_units import CEA_Obj


class GasGenerator:
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
            enthalpy_units='J/kg',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            # fac_CR=self.cr,
            make_debug_prints=False)
    def combustion_sim(self, OF, pc):
        Tg_c = self.cea.get_Tcomb(Pc=pc, MR=OF)
        [mw_c, gam_c] = self.cea.get_Chamber_MolWt_gamma(Pc=pc, MR=OF, eps=40)
        R_c = 8314.5 / mw_c  # J/kg-K
        Cp_c = self.cea.get_Chamber_Cp(Pc=pc, MR=OF, eps=40)
        # print(f"mw={mw_c}, R={R_c}, gam={gam_c}, Cp={Cp_c}")
        return Tg_c, Cp_c, R_c, gam_c