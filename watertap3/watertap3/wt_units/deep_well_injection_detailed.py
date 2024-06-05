from pyomo.environ import Block, Expression, NonNegativeReals, Var, units as pyunits
from watertap3.utils import financials
from watertap3.wt_units.wt_unit import WT3UnitProcess
import numpy as np

## REFERENCE:
# Using model data from 'Membrane Concentrate Disposal: Practices and Regulation'


module_name = 'deep_well_injection_detailed'
basis_year = 2001
tpec_or_tic = 'TPEC'


class UnitProcess(WT3UnitProcess):

    def fixed_cap(self, unit_params):
        '''
        Fixed capital cost for deep well injection.

        :param unit_params: Input parameter dictionary from input sheet.
        :type unit_params: dict
        :param lift_height: Lift height for pump [ft]
        :type lift_height: float
        :param pipe_distance: Piping distance to deep well injection site
        :type pipe_distance: float
        :return: Fixed capital cost for deep well injection [$MM]
        '''
        time = self.flowsheet().config.time
        t = self.flowsheet().config.time.first()
        self.lift_height = Var(time, initialize=400, domain=NonNegativeReals, units=pyunits.ft, doc='Lift height for pump [ft]')
        self.flow_in = pyunits.convert(self.flow_vol_in[t], to_units=pyunits.megagallons/ pyunits.day) 

        self.velocity = Var(initialize = 10, domain=NonNegativeReals, units=pyunits.dimensionless, doc='Flow velocity')
        self.depth = Var( initialize = 2500, domain=NonNegativeReals, units=pyunits.ft, doc='Depth [ft]')
        self.chem_dict = {}

        try:
            self.lift_height.fix(unit_params['lift_height'])
        except:
            self.lift_height.fix(400)

        # Check if velocity is listed in the input variables
        try:
            self.velocity.fix(unit_params['velocity'])
        except:
            self.velocity.fix(10)
        
        if self.velocity == 5:
            self.pipe_diameter = (7.5386 * self.flow_in**0.4997) * pyunits.inches
        elif self.velocity == 8:
            self.pipe_diameter = (5.9747 * self.flow_in**0.498) * pyunits.inches
        if self.velocity == 10:
            self.pipe_diameter = (5.329 * self.flow_in**0.4998) * pyunits.inches

        # Check if deep well injection well depth is listed
        try:
            self.depth.fix(unit_params['depth'])  #in ft
        except:
            self.depth.fix(2500)  #in ft

        if self.depth == 2500:
            self.logging = (4.5193 * self.pipe_diameter + 251.48) * 1e-3   # Converted to million $
            self.drilling = (17.611 * self.pipe_diameter + 338.05) * 1e-3   # Converted to million $
            self.tubing = (116.65*self.pipe_diameter**0.3786) * 1e-3   # Converted to million $ 
            self.casing = (20.545*self.pipe_diameter + 384.36) * 1e-3   # Converted to million $
            self.grouting = (1.1657*self.pipe_diameter**2 - 2.8548*self.pipe_diameter + 198.83) * 1e-3   # Converted to million $

        
        elif self.depth == 5000:
            self.logging = (5.5898 * self.pipe_diameter + 305.73) * 1e-3   # Converted to million $
            self.drilling = (31.881 * self.pipe_diameter + 616.09) * 1e-3   # Converted to million $
            self.tubing = (217.5*self.pipe_diameter**0.3724) * 1e-3   # Converted to million $ 
            self.casing = (37.45*self.pipe_diameter + 696.11) * 1e-3   # Converted to million $
            self.grouting = (1.7818*self.pipe_diameter**2 +4.0854*self.pipe_diameter + 325.9) * 1e-3   # Converted to million $        

        elif self.depth == 7500:
            self.logging = (6.5248 * self.pipe_diameter + 361.97) * 1e-3   # Converted to million $
            self.drilling = (46.591 * self.pipe_diameter + 882.27) * 1e-3   # Converted to million $
            self.tubing = (308.83*self.pipe_diameter**0.3767) * 1e-3   # Converted to million $ 
            self.casing = (51.211*self.pipe_diameter + 962.52) * 1e-3   # Converted to million $
            self.grouting = (2.4141*self.pipe_diameter**2 +6.5144*self.pipe_diameter + 440.68) * 1e-3   # Converted to million $

        elif self.depth == 1000:
            self.logging = (7.5694 * self.pipe_diameter + 415.55) * 1e-3   # Converted to million $
            self.drilling = (60.407 * self.pipe_diameter + 1166.5) * 1e-3   # Converted to million $
            self.tubing = (407.38*self.pipe_diameter**0.3741) * 1e-3   # Converted to million $ 
            self.casing = (65.52*self.pipe_diameter + 1228.1) * 1e-3   # Converted to million $
            self.grouting = (3.67*self.pipe_diameter**2 - 7.8567*self.pipe_diameter + 632.72) * 1e-3   # Converted to million $

        
        self.packing = (579.85 * np.log(self.pipe_diameter()) - 596.64) * 1e-3   # Converted to million $
        self.monitoring = (31.337*self.depth**0.3742) * 1e-3   # Converted to million $
        self.mobilizing = (32.225*self.depth**0.3711) * 1e-3   # Converted to million $

        self.deep_well_cap = self.logging + self.drilling + self.tubing + self.casing + self. grouting + self.packing + self.monitoring + self.mobilizing
         
        return self.deep_well_cap



    def elect(self):
        '''
        Electricity intensity for deep well injection [kWh/m3]

        :param lift_height: Lift height for pump [ft]
        :type lift_height: float
        :return: Electricity intensity [kWh/m3]
        '''
        t = self.flowsheet().config.time.first()
        time = self.flowsheet().config.time
        self.pump_eff = 0.9
        self.motor_eff = 0.9
        self.flow_in_gpm = pyunits.convert(self.flow_in, to_units=(pyunits.gallon / pyunits.minute))
        electricity = (0.746 * self.flow_in_gpm * self.lift_height[t] / (3960 * self.pump_eff * self.motor_eff)) / pyunits.convert(self.flow_in ,  to_units=pyunits.m ** 3 / pyunits.hr)
        return electricity

    def get_costing(self, unit_params=None, year=None):
        '''
        Initialize the unit in WaterTAP3.
        '''
        financials.create_costing_block(self, basis_year, tpec_or_tic)
        self.costing.fixed_cap_inv_unadjusted = Expression(expr=self.fixed_cap(unit_params),
                                                           doc='Unadjusted fixed capital investment')
        self.electricity = Expression(expr=self.elect(),
                                      doc='Electricity intensity [kwh/m3]')
        financials.get_complete_costing(self.costing)