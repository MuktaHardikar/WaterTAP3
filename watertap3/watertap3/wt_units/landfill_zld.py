from pyomo.environ import Block, Expression, units as pyunits
from watertap3.utils import financials
from watertap3.wt_units.wt_unit import WT3UnitProcess

## REFERENCE
## CAPITAL:
# Survey of High-Recovery and Zero Liquid Discharge Technologies for Water Utilities (2008).
# WateReuse Foundation
# https://www.waterboards.ca.gov/water_issues/programs/grants_loans/water_recycling/research/02_006a_01.pdf
# data in Table A2.1, Table A2.2

module_name = 'landfill_zld'
basis_year = 2007
tpec_or_tic = 'TPEC'


class UnitProcess(WT3UnitProcess):

    def fixed_cap(self,unit_params):
        time = self.flowsheet().config.time.first()
        self.flow_in = pyunits.convert(self.flow_vol_in[time], to_units=pyunits.m ** 3 / pyunits.day)
        self.capacity_basis = 302096*pyunits.kg/pyunits.day
        self.cap_scaling_exp = 0.7
        self.conc_mass_tot = 0
        for constituent in self.config.property_package.component_list:
            self.conc_mass_tot += self.conc_mass_in[time, constituent]
        self.density = 0.6312 * self.conc_mass_tot + 997.86  #kg/m3
        self.total_mass = total_mass = self.density * self.flow_in  #kg/day
        self.chem_dict = {}
        landfill_cap = (self.total_mass / self.capacity_basis) ** self.cap_scaling_exp

        # Calculate fixed required land area
        self.wet_solids_flow_rate = total_mass/1000 # Convert kg/day to metric tonne per day
        ratio_acre_flow_rate = 0.3 # acre/ (MT/day) Assuming 85% solids ratio
        self.required_area = ratio_acre_flow_rate*self.wet_solids_flow_rate
        cost_per_acre = 0.25   #MM$/acre based on 2007
        self.fixed_capital_basis = cost_per_acre*self.required_area  #MM$
        # transport_fixed_capital = 3.42
        # if unit_params['distance'] == 0:
        return self.fixed_capital_basis*landfill_cap
        # else:
            # return landfill_cap + transport_fixed_capital

    def elect(self):
        electricity = 0
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