from . import importfile, module_import
from .constituent_removal_water_recovery import create
from .mixer_example import Mixer
from .source_example import Source
from .split_test2 import Separator

__all__ = ['add_unit_process',
           'add_water_source',
           'add_splitter',
           'add_mixer']


def add_unit_process(m=None, unit_process_name=None, unit_process_type=None):

    up_module = m.fs.unit_module = module_import.get_module(unit_process_type)

    unit_params = m.fs.pfd_dict[unit_process_name]['Parameter']

    if 'basic' in unit_process_type:
        setattr(m.fs, unit_process_name, up_module.UnitProcess(default={'property_package': m.fs.water}))
        basic_unit_name = unit_params['unit_process_name']
        m = create(m, basic_unit_name, unit_process_name)

    else:
        setattr(m.fs, unit_process_name, up_module.UnitProcess(default={'property_package': m.fs.water}))
        m = create(m, unit_process_type, unit_process_name)


    getattr(m.fs, unit_process_name).get_costing(unit_params=unit_params)
    unit = getattr(m.fs, unit_process_name)
    unit.unit_name = unit_process_name

    return m


def add_water_source(m=None, source_name=None, link_to=None,
                     reference=None, water_type=None, case_study=None, flow=None):

    df = importfile.feedwater(
            input_file='data/case_study_water_sources.csv',
            reference=reference, water_type=water_type,
            case_study=case_study)

    setattr(m.fs, source_name, Source(default={'property_package': m.fs.water}))
    getattr(m.fs, source_name).set_source()
    getattr(m.fs, source_name).flow_vol_in.fix(flow)

    train_constituent_list = list(getattr(m.fs, source_name).config.property_package.component_list)

    for constituent_name in train_constituent_list:
        if constituent_name in df.index:
            getattr(m.fs, source_name).conc_mass_in[:, constituent_name].fix(df.loc[constituent_name].value)
        else:
            getattr(m.fs, source_name).conc_mass_in[:, constituent_name].fix(0)

    getattr(m.fs, source_name).pressure_in.fix(1)
    return m


def add_splitter(m=None, split_name=None, with_connection=False, outlet_list=None, outlet_fractions=None,
                 link_to=None, link_from=None, stream_name=None, unfix=False):

    setattr(m.fs, split_name, Separator(default={
            'property_package': m.fs.water,
            'ideal_separation': False,
            'outlet_list': outlet_list
            }))

    if unfix == True:
        getattr(m.fs, split_name).split_fraction[0, key].unfix()
    else:
        for key in outlet_fractions.keys():
            getattr(m.fs, split_name).split_fraction[0, key].fix(outlet_fractions[key])
    return m


# TO DO MAKE THE FRACTION A DICTIONARY
def add_mixer(m=None, mixer_name=None, with_connection=False, inlet_list=None,
              link_to=None, link_from=None, stream_name=None):

    setattr(m.fs, mixer_name, Mixer(default={
            'property_package': m.fs.water,
            'inlet_list': inlet_list
            }))
    return m
