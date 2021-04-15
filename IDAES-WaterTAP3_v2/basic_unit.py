#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:22:42 2021

@author: ksitterl
"""

##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Demonstration zeroth-order model for WaterTAP3
"""

# Import IDAES cores
from idaes.core import (declare_process_block_class, UnitModelBlockData, useDefault)
from idaes.core.util.config import is_physical_parameter_block
# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.environ import value
# Import WaterTAP# financials module
import financials
from financials import *  # ARIEL ADDED
from cost_curves import basic_unit

# Import properties and units from "WaterTAP Library"

##########################################
####### UNIT PARAMETERS ######
# At this point (outside the unit), we define the unit parameters that do not change across case studies or analyses ######.
# Below (in the unit), we define the parameters that we may want to change across case studies or analyses. Those parameters should be set as variables (eventually) and atttributed to the unit model (i.e. m.fs.UNIT_NAME.PARAMETERNAME). Anything specific to the costing only should be in  m.fs.UNIT_NAME.costing.PARAMETERNAME ######
##########################################

## REFERENCE: Cost Estimating Manual for Water Treatment Facilities (McGivney/Kawamura)

### MODULE NAME ###
module_name = "basic_unit"

# Cost assumptions for the unit, based on the method #
# this is either cost curve or equation. if cost curve then reads in data from file.
unit_cost_method = "cost_curve"
tpec_or_tic = "TPEC"


# You don't really want to know what this decorator does
# Suffice to say it automates a lot of Pyomo boilerplate for you
@declare_process_block_class("UnitProcess")
class UnitProcessData(UnitModelBlockData):
    """
    This class describes the rules for a zeroth-order model for a unit

    The Config Block is used tpo process arguments from when the model is
    instantiated. In IDAES, this serves two purposes:
         1. Allows us to separate physical properties from unit models
         2. Lets us give users options for configuring complex units
    The dynamic and has_holdup options are expected arguments which must exist
    The property package arguments let us define different sets of contaminants
    without needing to write a new model.
    """

    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(domain=In([False]), default=False, description="Dynamic model flag - must be False", doc="""Indicates whether this model will be dynamic or not,
**default** = False. Equilibrium Reactors do not support dynamic behavior."""))
    CONFIG.declare("has_holdup", ConfigValue(default=False, domain=In([False]), description="Holdup construction flag - must be False", doc="""Indicates whether holdup terms should be constructed or not.
**default** - False. Equilibrium reactors do not have defined volume, thus
this must be False."""))
    CONFIG.declare("property_package", ConfigValue(default=useDefault, domain=is_physical_parameter_block, description="Property package to use for control volume", doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))
    CONFIG.declare("property_package_args", ConfigBlock(implicit=True, description="Arguments to use for constructing property packages", doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))

    def build(self):
        import unit_process_equations
        return unit_process_equations.build_up(self, up_name_test=module_name)

    # NOTE ---> THIS SHOULD EVENTUaLLY BE JUST FOR COSTING INFO/EQUATIONS/FUNCTIONS. EVERYTHING ELSE IN ABOVE.
    def get_costing(self, module=financials, cost_method="wt", year=None, unit_params=None):

        unit_process_name = unit_params['unit_process_name']
        try:
            case_specific = unit_params['case_specific']
        except:
            case_specific = None

        if case_specific:
            basis, cap_basis, cap_exp, elect, unit_year, kind = basic_unit(unit_process_name, case_specific=case_specific)
        else:
            basis, cap_basis, cap_exp, elect, unit_year, kind = basic_unit(unit_process_name)
        if not hasattr(self.flowsheet(), "costing"):
            self.flowsheet().get_costing(module=module, year=year)

        self.costing = Block()

        time = self.flowsheet().config.time.first()
        flow_in = pyunits.convert(self.flow_vol_in[time], to_units=pyunits.m ** 3 / pyunits.hour)


        sys_cost_params = self.parent_block().costing_param
        self.costing.tpec_tic = sys_cost_params.tpec if tpec_or_tic == "TPEC" else sys_cost_params.tic

        self.costing.basis_year = unit_year

        chem_dict = {}
        self.chem_dict = chem_dict

        def fixed_cap():
            if kind == 'flow':
                flow_basis = basis * (pyunits.m ** 3 / pyunits.hour)
                flow_factor = flow_in / flow_basis
                basic_cap = cap_basis * flow_factor ** cap_exp
                return basic_cap

            if kind == 'mass':
                mass_basis = basis * (pyunits.kg / pyunits.hour)
                mass_in = 0
                for constituent in self.config.property_package.component_list:
                    mass_in += self.conc_mass_in[time, constituent]
                density = 0.6312 * mass_in + 997.86  # kg / m3
                total_mass_in = density * flow_in  # kg / hr
                mass_factor = total_mass_in / mass_basis
                basic_cap = cap_basis * mass_factor ** cap_exp
                return basic_cap

        self.deltaP_outlet.unfix()
        self.deltaP_waste.unfix()
        self.pressure_out.fix(1)
        self.pressure_waste.fix(1)

        ## fixed_cap_inv_unadjusted ##
        self.costing.fixed_cap_inv_unadjusted = Expression(expr=fixed_cap(), doc="Unadjusted fixed capital investment")  # $M

        ## electricity consumption ##
        self.electricity = elect  # kwh/m3

        ##########################################
        ####### GET REST OF UNIT COSTS ######
        ##########################################

        module.get_complete_costing(self.costing)
