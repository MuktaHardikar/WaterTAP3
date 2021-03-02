#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyomo.environ import ConcreteModel, SolverFactory, TransformationFactory
from pyomo.network import Arc

from idaes.core import FlowsheetBlock

# Import properties and units from "WaterTAP Library"
from water_props import WaterParameterBlock

import financials

from pyomo.environ import Block, Constraint, Var, units as pyunits
import module_import
from source_example import Source
from mixer_example import Mixer1
from split_test2 import Separator1
from pyomo.network import Arc, SequentialDecomposition

import pandas as pd

import generate_constituent_list

def add_unit_process(m=None, unit_process_name=None, unit_process_type=None, with_connection=False, from_splitter=False, splitter_tream=None, link_to=None, link_from=None, connection_type="series",
                     stream_name=None):  # in design

    import constituent_removal_water_recovery

    up_module = module_import.get_module(unit_process_type)

    unit_params = m.fs.pfd_dict[unit_process_name]["Parameter"]

    if 'basic_unit' in unit_process_name:
        unit_process_name = unit_params['unit_process_name']
        setattr(m.fs, unit_process_name, up_module.UnitProcess(default={"property_package": m.fs.water}))
        m = constituent_removal_water_recovery.create(m, unit_process_name, unit_process_name)

    else:
        setattr(m.fs, unit_process_name, up_module.UnitProcess(default={"property_package": m.fs.water}))
        m = constituent_removal_water_recovery.create(m, unit_process_type, unit_process_name)

    ### SET PARAMS HERE FOR UP ###
    # PARAMS =
    import financials
    getattr(m.fs, unit_process_name).get_costing(module=financials, unit_params=unit_params)

    if with_connection == True:

        if from_splitter == True:
            
            setattr(m.fs, splitter_tream, 
                    Arc(source=getattr(m.fs, link_from).outlet1, 
                                           destination=getattr(m.fs, link_to).inlet))            
            
        if from_splitter == False:
            m = connect_blocks(m = m, up1 = link_from, up2 = link_to, 
                               connection_type = connection_type, 
                               stream_name = stream_name)
    
    return m




def add_water_source(m = None, source_name = None, link_to = None, 
                     reference = None, water_type = None, case_study = None, flow = None): # in design
    
    import importfile
    
    df = importfile.feedwater(
        input_file="data/case_study_water_sources.csv",
        reference = reference, water_type = water_type, 
        case_study = case_study)
    
    #set the flow based on the case study if not specified. This should have already been set in case study .py
    if flow is None: flow = float(df.loc["flow"].value)
    
    train_constituent_list = generate_constituent_list.run()
    
    setattr(m.fs, source_name, Source(default={"property_package": m.fs.water}))
    
    getattr(m.fs, source_name).inlet.flow_vol.fix(flow)
        
    for constituent_name in train_constituent_list:
        
        if constituent_name in df.index:
            getattr(m.fs, source_name).inlet.conc_mass[:, constituent_name].fix(df.loc[constituent_name].value)        
        
        else:
            getattr(m.fs, source_name).inlet.conc_mass[:, constituent_name].fix(0)
        
        getattr(m.fs, source_name).removal_fraction[:, constituent_name].fix(0)
    
    getattr(m.fs, source_name).inlet.temperature.fix(300)
    getattr(m.fs, source_name).inlet.pressure.fix(2e5)
    
    
    # Set removal and recovery fractions -> CAN WE JUST SET OUTLETS AND THAT'S IT? OR CLEANER WITH THE SAME FORMAT?
    getattr(m.fs, source_name).water_recovery.fix(1.0)
    # Also set pressure drops - for now I will set these to zero
    getattr(m.fs, source_name).deltaP_outlet.fix(1e-4)
    getattr(m.fs, source_name).deltaP_waste.fix(1e-4)
    
    if link_to is not None: # TODO - potential for multiple streams
        connect_blocks(m = m, up1 = source_name, up2 = link_to, connection_type = None, 
                               stream_name = ("%s_stream" % source_name))
        
    return m


def add_splitter(m = None, split_name = None, with_connection = False, outlet_list = None, outlet_fractions = None,
                    link_to = None, link_from = None, stream_name = None, unfix = False): # in design
    
    
    setattr(m.fs, split_name, Separator1(default={
        "property_package": m.fs.water,
        "ideal_separation": False,
        "outlet_list": outlet_list}))
    
    if unfix == True:
        getattr(m.fs, split_name).split_fraction[0, key].unfix()    
    else:
        for key in outlet_fractions.keys():
            getattr(m.fs, split_name).split_fraction[0, key].fix(outlet_fractions[key])     
    
    return m

# TO DO MAKE THE FRACTION A DICTIONARY
def add_mixer(m = None, mixer_name = None, with_connection = False, inlet_list = None,
                    link_to = None, link_from = None, stream_name = None): # in design
    
    setattr(m.fs, mixer_name, Mixer1(default={
        "property_package": m.fs.water,
        "inlet_list": inlet_list}))
    
    return m





def main():
    print("importing something")
    # need to define anything here?


if __name__ == "__main__":
    main()