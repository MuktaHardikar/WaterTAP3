import numpy as np
import math
from pyomo.environ import units as pyunits
import pandas as pd
import requests
import urllib

def truck_costing(distance, state='TX', wacc = 0.05, plant_lifetime_yrs = 30,fuel_price_file_path = '/Users/mhardika/Documents/watertap3/WaterTAP3/watertap3/watertap3/data/fuel_costs.csv'):
    '''
    Reference: 
    Marufuzzaman, M., et al. (2015). "Truck versus pipeline transportation cost analysis of wastewater sludge." 
    Transportation Research Part A: Policy and Practice 74: 14-30.

    Gas/diesel costs: https://gasprices.aaa.com/state-gas-price-averages/

    Wastewater treatment plan owned truck cost function
    This function returns LCOW of transportation using trucks as a function distance and state (cost of fuel)
    
    '''

    if distance == 0:
        return 0
    distance = distance*pyunits.km
    distance = pyunits.convert(distance,to_units = pyunits.miles)
    truck_capacity = 30    #m3
    loading_unloading_rate = 0.8*60    #m3/h
    loading_unloading_time = truck_capacity/loading_unloading_rate  #h
    additional_loading_unloading_time = 20/60   #h
    days_operation = 365
    plant_utilization = days_operation/365

    total_loading_unloading_time = 2*loading_unloading_time + additional_loading_unloading_time

    # Fixed cost components

    cost_ownership = 26810   #$ --> The salvage value is already deducted so the capital recovery factor was not included for the truck fixed capital cost
    annual_sales_tax = 720   #$
    license_fees_taxes = 2086  #$
    management_overhead = 13020  #$
    insurance_cost = 8726  #$

    distance_basis = 100000 # miles
    annual_trips_basis = 500  
    volume_basis  = truck_capacity*annual_trips_basis  #m3

    total_fixed_capital_cost = (cost_ownership + annual_sales_tax + license_fees_taxes 
                                + management_overhead + insurance_cost)/volume_basis # $/m3
    
    capital_recovery_factor = (wacc * (1 + wacc) ** plant_lifetime_yrs) / (((1 + wacc) ** plant_lifetime_yrs) - 1)

    # Capital recovery factor in paper
    wacc_1 = 0.1
    plant_lifetime_yrs_1 = 30
    capital_recovery_factor_1 = (wacc_1 * (1 + wacc_1) ** plant_lifetime_yrs_1) / (((1 + wacc_1) ** plant_lifetime_yrs_1) - 1)

    total_fixed_capital_cost = total_fixed_capital_cost * capital_recovery_factor / capital_recovery_factor_1
    

    # Variable cost components

    # reading fuel costs as a function of state
    fuel_df = pd.read_csv(fuel_price_file_path)
    fuel_price = fuel_df.loc[fuel_df['state_code']==state]['diesel_cost'].values[0] #$/gal --> function of state
    mileage = 5.85 #gal/mil (average of 5.1 and 6.6)
    fuel_cost_mile = fuel_price/mileage #$/mile
    labor_cost = 0.82 #$/mile
    maintenance_repair_cost = 0.17 #$/mile
    tire_cost = 0.04  #$/mile

    total_variable_cost = (fuel_cost_mile + labor_cost + 
                           maintenance_repair_cost + tire_cost )/truck_capacity   #$/mile/m3

    return  (total_fixed_capital_cost + total_variable_cost * distance())


def pipe_costing(capacity, distance, elev_gain = 1e-5, wacc = 0.05, plant_lifetime_yrs = 30,electricity_rate = 0.06,pump_power = 20,pumping_velocity = 3):
    '''
    Reference: Marufuzzaman, M., et al. (2015). "Truck versus pipeline transportation cost analysis of wastewater sludge." 
    Transportation Research Part A: Policy and Practice 74: 14-30.

    This function return the LCOW of transport through pipes as a function of distance and volume
    '''

    if distance == 0:
        return 0
    if capacity == 0:
        return 0
    if elev_gain == 0:
        elev_gain = 1e-5
    
    # Inputs
    storage_capacity = capacity *pyunits.m**3/pyunits.day
    distance = distance *pyunits.km

    # Assumed velocity from ref (Pootakham and Kumar 2010) was 1.5, updated to 3 m/s
    pumping_velocity = pumping_velocity *pyunits.m/pyunits.s
    # Assumed pump power from ref (Pootakham and Kumar 2010)
    pump_power = pump_power *pyunits.hp
    
    days_operation = 350
    plant_utilization = days_operation/365

    capital_recovery_factor = (wacc * (1 + wacc) ** plant_lifetime_yrs) / (((1 + wacc) ** plant_lifetime_yrs) - 1)

    # Fixed capital cost - not a function of distance
    
    # Inlet pumping station - based on correlation in "Truck versus pipeline transportation cost analysis of wastewater sludge"
    # storage_tank = 1000000*((storage_capacity/9400)*0.65)
    
    tank_cost_basis = 1556056 #1e6  # $/tank
    tank_capacity_basis = 9400 #m3
    storage_tank = tank_cost_basis*((storage_capacity/tank_capacity_basis)**0.65)  # $

    # Valve
    pipe_csa = pyunits.convert(storage_capacity,to_units = pyunits.m**3/pyunits.s)/pumping_velocity
    pipe_diameter = pyunits.convert(2 * (pipe_csa/np.pi)**0.5 , to_units = pyunits.inch)

    if pipe_diameter() > 63:
        # print('Diameter reduced')
        pipe_diameter = 63 * pyunits.inch
        pipe_csa = pyunits.convert(np.pi/4*pipe_diameter**2, to_units = pyunits.m**2)
        pumping_velocity = pyunits.convert(storage_capacity,to_units = pyunits.m**3/pyunits.s)/pipe_csa
        # print(pumping_velocity())

    # print('pipe diameter (inch):',pipe_diameter())
    fitting_valve_cost_basis = 17496 #13220
    fitting_valve_cost = fitting_valve_cost_basis*(pyunits.convert(pipe_diameter, to_units = pyunits.ft))**1.05

    # Inlet pump
    inlet_pump_cost_basis = 1750 #1322
    inlet_pump_cost = inlet_pump_cost_basis*(pump_power**0.8)

    # Outlet pump - Assumed to have 5 HP pump power
    outlet_pump_cost = inlet_pump_cost_basis*(5**0.8)

    # Miscellaneous / construction costs --> Taken from reference "Truck versus pipeline transportation cost analysis of wastewater sludge"
    
    road_access_cost = 424/capital_recovery_factor*plant_utilization  # 320
    building_foundation_cost = 1469/capital_recovery_factor*plant_utilization  #1110

    # Total inlet and outlet station cost
    total_inlet_station_fixed_capital_cost = storage_tank + fitting_valve_cost + inlet_pump_cost + building_foundation_cost + road_access_cost 
    total_outlet_station_fixed_capital_cost = storage_tank + fitting_valve_cost + outlet_pump_cost + building_foundation_cost

    # Total fixed capital cost - $/m3
    total_fixed_capital_cost = ( total_inlet_station_fixed_capital_cost + total_outlet_station_fixed_capital_cost )

    total_fixed_capital_cost = total_fixed_capital_cost * capital_recovery_factor/plant_utilization #$/year

    # Variable -function of distance # $/m3

    # Booster station costs
    booster_pump_cost = inlet_pump_cost
    booster_pump_installation_cost = 0.1 * inlet_pump_cost

    # Number of booster pumps 
    friction_factor = 0.005 
    density = 1000*pyunits.kg/pyunits.m**3
    g = 9.8*pyunits.m/pyunits.s**2
    
    deltaP_grad =  friction_factor*density*(pumping_velocity**2)/(2*pyunits.convert(pipe_diameter,to_units = pyunits.m))*1e-5 #bar/m
    deltaP_elev_gain = density*g*elev_gain*pyunits.m/pyunits.convert(distance,pyunits.m) * 1e-5 #bar/m

    Pmax = 15 # maximum allowable pressure in the pipe
    Pmin = 2  # minimum pressure after which there's no flow

    lx = (Pmax-Pmin)/(deltaP_grad() + deltaP_elev_gain())
    N = pyunits.convert(distance,to_units = pyunits.m)/lx
    
    N = math.ceil(N())-1
    # print('N:',N,'lx:',lx)
    booster_power_line_cost = 1.75*((8400*N)+8400)  #1.32

    total_booster_station_cost = (N*(building_foundation_cost + booster_pump_cost() + booster_pump_installation_cost() + road_access_cost) + booster_power_line_cost)

    # Annual maintenance variable costs --> putting numbers from the paper
    # Pipe costs
    pipe_material_cost = 1200 /pyunits.ton   # updated to HDPE, PVC pipe cost
    pipe_thickness = 0.05*pipe_diameter() * pyunits.inches
    pipe_cost_basis = 37.3 #28.2
    pipe_cost = pipe_cost_basis*(pyunits.convert(pipe_diameter,to_units = pyunits.inches)-pipe_thickness)*pipe_thickness*pyunits.convert(distance,to_units=pyunits.mile)*pipe_material_cost

    construction_cost_basis = 41077 #31037.1
    construction_cost = construction_cost_basis * pyunits.convert(distance,to_units = pyunits.mile) * pyunits.convert(pipe_diameter,to_units = pyunits.inches)
    pipe_maintenance_cost = 0.5/100*pipe_cost
    # print(pipe_maintenance_cost())
    pump_maintenance_cost = 0.03*(total_fixed_capital_cost) #/capital_recovery_factor*plant_utilization
    # print(pump_maintenance_cost())
    man_hours = 8400  # hours per year for 100 miles
    
    # Labor cost is towards operating the pumps so removed the distance component
    labor_cost_basis = 38  #29.2
    labor_cost = labor_cost_basis*man_hours/capital_recovery_factor*plant_utilization/100/1.6*distance()

    # Following method in WT3 water_pumping_station
    electricity = pyunits.convert((N+1)*pump_power,to_units = pyunits.kW) 

    total_electricity_cost = electricity_rate * electricity *  days_operation * 24

    # Constant 17780 updated
    road_access_cost_variable_cost =  23532/capital_recovery_factor*plant_utilization
    
    # Total miscellaneous costs
    total_misc_variable_capital_cost = ( pipe_cost + construction_cost +  labor_cost  + road_access_cost_variable_cost )

    total_variable_capital_cost = (total_booster_station_cost + total_misc_variable_capital_cost)*capital_recovery_factor/plant_utilization #$/year

    # Total O&M costs
    total_onm_costs = (pipe_maintenance_cost + pump_maintenance_cost + total_electricity_cost)/plant_utilization

    # LCOW $/m3
    return (total_fixed_capital_cost() + total_variable_capital_cost() + total_onm_costs())/(storage_capacity()*365)


def elevation(lat,lon):
    url = r'https://epqs.nationalmap.gov/v1/json?'

    #location
    params = {
            'output': 'json',
            'x':lon ,
            'y': lat,
            'units': 'Meters'
    }
    result = requests.get((url + urllib.parse.urlencode(params)))
    try:
        elevation_start = float(result.json()['value'])
    except:
        elevation_start = 0
        print('Failure')

    return elevation_start


def elevation_gain(lat1,lon1,lat2,lon2):
    url = r'https://epqs.nationalmap.gov/v1/json?'

    # Start location
    params = {
            'output': 'json',
            'x':lon1 ,
            'y': lat1,
            'units': 'Meters'
    }
    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation_start = float(result.json()['value'])
    

    # End location
    params = {
            'output': 'json',
            'x':lon2 ,
            'y': lat2,
            'units': 'Meters'
    }

    result = requests.get((url + urllib.parse.urlencode(params)))
    elevation_end = float(result.json()['value'])

    if (elevation_end-elevation_start)<=0:
        elev_gain = 0
    else:
        elev_gain = elevation_end-elevation_start 

    return elev_gain

