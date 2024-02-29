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


def pipe_costing(flow_in, distance, elev_gain = 1e-5, wacc = 0.05, plant_lifetime_yrs = 30,electricity_rate = 0.06,
                 pump_power = 24,pumping_velocity = 3, hour_storage = 6,cost_break_down = False):
    '''
    Reference: Marufuzzaman, M., et al. (2015). "Truck versus pipeline transportation cost analysis of wastewater sludge." 
    Transportation Research Part A: Policy and Practice 74: 14-30.

    This function return the LCOW of transport through pipes as a function of distance and volumetric flow in
    Units:
    flow_in: m3/day
    distance: km
    elev_gain: m
    electricity_rate: #/kWh
    pump_power: hp
    pumping_velocity: m/s

    '''
    if distance == 0:
        return 0
    if flow_in == 0:
        return 0
    if elev_gain == 0:
        elev_gain = 1e-5
    
    # Inputs
    
    flow_in = flow_in*pyunits.m**3/pyunits.day
    distance = distance *pyunits.km
    elev_gain = elev_gain*pyunits.m
    electricity_rate = electricity_rate*pyunits.kWh
    # Assumed velocity from ref (Pootakham and Kumar 2010) was 1.5, updated to 3 m/s
    pumping_velocity = pumping_velocity*pyunits.m/pyunits.s
    # Assumed pump power from ref (Pootakham and Kumar 2010)
    pump_power = pump_power *pyunits.hp
    
    days_operation = 350
    plant_utilization = days_operation/365

    capital_recovery_factor = (wacc * (1 + wacc) ** plant_lifetime_yrs) / (((1 + wacc) ** plant_lifetime_yrs) - 1)


    # Fixed Capital Costs

    # Updating storage capital cost to be from WT3
    a = 0.00344
    b = 0.72093
    storage_duration = hour_storage*pyunits.h
    storage_capacity = storage_duration*pyunits.convert(flow_in,to_units = pyunits.m**3/pyunits.h)
    # Storage capital cost in $
    storage_tank_capital_cost = 1e6*a*storage_capacity**b

    # Fitting valve capital cost in $
    pipe_csa = pyunits.convert(flow_in,to_units = pyunits.m**3/pyunits.s)/pumping_velocity
    pipe_diameter = pyunits.convert(2 * (pipe_csa/np.pi)**0.5 , to_units = pyunits.inch)

    # Maximum physical pipe diameter
    if pipe_diameter() > 63:
        # print('Diameter reduced')
        pipe_diameter = 63 * pyunits.inch
        pipe_csa = pyunits.convert(np.pi/4*pipe_diameter**2, to_units = pyunits.m**2)
        pumping_velocity = pyunits.convert(flow_in,to_units = pyunits.m**3/pyunits.s)/pipe_csa
        # print(pumping_velocity())

    # print('pipe diameter (inch):',pipe_diameter())
   
    fitting_valve_cost_basis = 17496 #13220
    fitting_valve_capital_cost = fitting_valve_cost_basis*(pyunits.convert(pipe_diameter, to_units = pyunits.ft))**1.05

    # Inlet pump capital cost in $
    inlet_pump_cost_basis = 1750 #1322
    inlet_pump_capital_cost = inlet_pump_cost_basis*(pump_power**0.8)

    # Outlet pump in $- Assumed to have 5 HP pump power
    outlet_pump_cost_capital_cost = inlet_pump_cost_basis*(5**0.8)

    # Miscellaneous / construction costs --> Taken from reference "Truck versus pipeline transportation cost analysis of wastewater sludge"
    # 320 $/year corrected to 424 $/year adjusted to $ by multiplying with plant_utilization/capital recovery factor
    road_access_capital_cost =  6250.059540792913
    # 1110 $/year corrected to 1469 $/year adjusted to $ by multiplying with plant_utilization/capital recovery factor
    building_foundation_capital_cost = 21654.097795813184

    # Total inlet and outlet station cost
    total_inlet_station_fixed_capital_cost = storage_tank_capital_cost + fitting_valve_capital_cost + inlet_pump_capital_cost + building_foundation_capital_cost + road_access_capital_cost 
    total_outlet_station_fixed_capital_cost = storage_tank_capital_cost + fitting_valve_capital_cost + outlet_pump_cost_capital_cost + building_foundation_capital_cost

    # Total fixed capital cost - $
    total_fixed_capital_cost = (total_inlet_station_fixed_capital_cost + total_outlet_station_fixed_capital_cost)



    # Variable Capital Costs

    # Booster station costs in $
    booster_pump_capital_cost = inlet_pump_capital_cost
    booster_pump_installation_capital_cost = 0.1 * inlet_pump_capital_cost

    # Number of booster pumps 
    friction_factor = 0.005 
    density = 1000*pyunits.kg/pyunits.m**3
    g = 9.8*pyunits.m/pyunits.s**2
    
    # Pressure drop (deltaP_grad) because of friction losses and pressure drop (deltaP_elev_gain) because of elevation gain
    deltaP_grad =  friction_factor*density*(pumping_velocity**2)/(2*pyunits.convert(pipe_diameter,to_units = pyunits.m))*1e-5  #bar/m
    deltaP_elev_gain = density*g*elev_gain/pyunits.convert(distance,pyunits.m) * 1e-5  #bar/m

    Pmax = 15 # maximum allowable pressure in the pipe in bar
    Pmin = 0  # minimum pressure after which there's no flow in bar

    # Maximum distance between booster pumps with the calculated pressure drop and the feasible pressure range
    lx = (Pmax-Pmin)/(deltaP_grad() + deltaP_elev_gain())
    
    # Number of booster pumps
    n_booster_pumps = pyunits.convert(distance,to_units = pyunits.m)/lx
    
    # Round up the number of booster pumps
    n_booster_pumps = math.ceil(n_booster_pumps())-1
    # print('N:',N,'lx:',lx)

    # Total variable capital cost of setting up electrical power lines in $
    booster_power_line_variable_capital_cost = 1.75*((8400*n_booster_pumps)+8400)  #1.32

    total_booster_station_cost = (n_booster_pumps*(building_foundation_capital_cost + booster_pump_capital_cost() + 
                                                   booster_pump_installation_capital_cost() + road_access_capital_cost) + booster_power_line_variable_capital_cost)

    # Pipe cost in $
    pipe_material_cost = 1200 /pyunits.ton   # updated to HDPE, PVC pipe cost
    pipe_thickness = pyunits.convert(0.05*pipe_diameter,to_units = pyunits.inches)
    pipe_cost_basis = 37.3 #28.2
    pipe_cost = pipe_cost_basis*(pyunits.convert(pipe_diameter,to_units = pyunits.inches)-pipe_thickness)*pipe_thickness*pyunits.convert(distance,to_units=pyunits.mile)*pipe_material_cost

    # Pipe Construction cost in $
    construction_cost_basis = 41077   #31037.1
    pipe_construction_cost = construction_cost_basis * pyunits.convert(distance,to_units = pyunits.mile) * pyunits.convert(pipe_diameter,to_units = pyunits.inches)

    # 17780 $/year corrected to 23532 $/year adjusted to $ by multiplying with plant_utilization/capital recovery factor
    pipe_road_access_cost_variable_cost =  346878.3
    

    # Annual Maintenance Costs $/year

    # Maintenance Costs in $/year
    pipe_maintenance_cost = 0.5/100*pipe_cost
    pump_maintenance_cost = 0.03*(total_inlet_station_fixed_capital_cost + total_outlet_station_fixed_capital_cost + n_booster_pumps*booster_pump_capital_cost) # Check units

    man_hours = 8400   # hours per year for 100 miles (Assumes 50 weeks * 7 days * 24 h)
    
    # Labor cost is towards operating the pumps so removed the distance component
    labor_cost_basis = 38  #29.2 in $/h
    labor_cost = labor_cost_basis*man_hours # Normalize to pipe distance

    # Following method in WT3 water_pumping_station
    electricity = pyunits.convert((n_booster_pumps+1)*pump_power,to_units = pyunits.kW) 
    # Electrcity cost in $ 
    total_electricity_cost = electricity_rate * electricity * days_operation * 24

    
    # Total miscellaneous costs ($/year)
    total_pipe_variable_capital_cost = (pipe_cost + pipe_construction_cost + pipe_road_access_cost_variable_cost)

    total_variable_capital_cost = (total_booster_station_cost + total_pipe_variable_capital_cost)*capital_recovery_factor/plant_utilization #$/year

    total_fixed_capital_cost = total_fixed_capital_cost * capital_recovery_factor/plant_utilization #$/year

    # Total O&M costs
    total_onm_costs = (pipe_maintenance_cost + pump_maintenance_cost +  labor_cost  + total_electricity_cost)/plant_utilization

    if cost_break_down == True:
        breakdown = {'storage_capacity': storage_capacity(),
                    'storage_tank_capital_cost': storage_tank_capital_cost()*capital_recovery_factor/plant_utilization,
                    'fitting_valve_capital_cost': fitting_valve_capital_cost()*capital_recovery_factor/plant_utilization,
                    'inlet_pump_capital_cost': inlet_pump_capital_cost()*capital_recovery_factor/plant_utilization,
                    'outlet_pump_cost_capital_cost':outlet_pump_cost_capital_cost*capital_recovery_factor/plant_utilization,
                    'road_access_capital_cost': road_access_capital_cost*capital_recovery_factor/plant_utilization,
                    'building_foundation_capital_cost':building_foundation_capital_cost*capital_recovery_factor/plant_utilization,
                    'booster_pump_capital_cost': booster_pump_capital_cost()*capital_recovery_factor/plant_utilization,
                    'booster_pump_installation_capital_cost':booster_pump_installation_capital_cost()*capital_recovery_factor/plant_utilization,
                    'n_booster_pumps': n_booster_pumps,
                    'booster_power_line_variable_capital_cost':booster_power_line_variable_capital_cost*capital_recovery_factor/plant_utilization,
                    'pipe_diameter':pipe_diameter(),
                    'pipe_cost': pipe_cost()*capital_recovery_factor/plant_utilization,
                    'pipe_construction_cost': pipe_construction_cost()*capital_recovery_factor/plant_utilization,
                    'pipe_maintenance_cost':pipe_maintenance_cost()*capital_recovery_factor/plant_utilization,
                    'pump_maintenance_cost': pump_maintenance_cost()*capital_recovery_factor/plant_utilization,
                    'labor_cost':labor_cost*capital_recovery_factor/plant_utilization,
                    'total_electricity_cost': total_electricity_cost(),
                    'pipe_road_access_cost_variable_cost': pipe_road_access_cost_variable_cost,
                    'total_inlet_station_fixed_capital_cost': total_inlet_station_fixed_capital_cost()*capital_recovery_factor/plant_utilization,
                    'total_outlet_station_fixed_capital_cost':total_outlet_station_fixed_capital_cost()*capital_recovery_factor/plant_utilization,
                    'total_booster_station_cost':total_booster_station_cost*capital_recovery_factor/plant_utilization,
                    'total_pipe_variable_capital_cost':total_pipe_variable_capital_cost()*capital_recovery_factor/plant_utilization,
                    'total_onm_costs': total_onm_costs(),
                    'capital_recovery_factor':capital_recovery_factor,
                    'plant_utilization':plant_utilization
                         }
        return breakdown

    # LCOW $/m3
    # return (total_fixed_capital_cost() + total_variable_capital_cost() + total_onm_costs())/(storage_capacity()*365)
    return (total_fixed_capital_cost() + total_variable_capital_cost() + total_onm_costs())/(flow_in()*365)




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

