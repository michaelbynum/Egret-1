#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains the declarations for the modeling components
typically used for buses (including loads and shunts)
"""
import pyomo.environ as pe
from pyomo.core.base.block import _BlockData
from pyomo.core.base.set import _SetData
from egret.data.model_data import ModelData
import egret.model_library.decl as decl
from egret.model_library.defn import FlowType, CoordinateType, ApproximationType
from math import tan,  radians
from typing import Optional
import math


def declare_set_bus_set(
        m: _BlockData,
        md: ModelData,
):
    buses = list(md.data['elements']['bus'].keys())
    m.bus_set = pe.Set(initialize=buses)


def declare_expression_p_balance_slack_expr(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
        rule: Optional[str] = 'default'
):
    if rule not in {'default', None}:
        raise ValueError("rule should be either 'default' or None")

    m.p_balance_slack_expr = pe.Expression(index_set)

    if rule == 'default':
        for b in index_set:
            m.p_balance_slack_expr[b] = 0


def declare_expression_q_balance_slack_expr(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
        rule: Optional[str] = 'default'
):
    if rule not in {'default', None}:
        raise ValueError("rule should be either 'default' or None")

    m.q_balance_slack_expr = pe.Expression(index_set)

    if rule == 'default':
        for b in index_set:
            m.q_balance_slack_expr[b] = 0


def declare_var_p_over_generation(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    model.p_over_generation = pe.Var(index_set, initialize=0, bounds=(0, None))

    ubs = dict()
    for b in index_set:
        ubs[b] = 0

    for gname, gen in md.data['elements']['generator'].items():
        bus_name = gen['bus']
        if bus_name not in index_set:
            continue
        if gen['p_max'] > 0:
            ubs[bus_name] += gen['p_max']

    for lname, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'p_load' in load:
            if load['p_load'] < 0:
                ubs[bus_name] -= load['p_load']

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] != 'fixed':
                continue
            if shunt['gs'] >= 0:
                continue
            bus_name = shunt['bus']
            if bus_name not in index_set:
                continue
            ubs[bus_name] -= shunt['gs'] * md.data['elements']['bus'][bus_name]['v_max']**2

    for b in index_set:
        model.p_over_generation[b].setub(ubs[b])


def declare_var_q_over_generation(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    model.q_over_generation = pe.Var(index_set, initialize=0, bounds=(0, None))

    ubs = dict()
    for b in index_set:
        ubs[b] = 0

    for gname, gen in md.data['elements']['generator'].items():
        bus_name = gen['bus']
        if bus_name not in index_set:
            continue
        if gen['q_max'] > 0:
            ubs[bus_name] += gen['q_max']

    for lname, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'q_load' in load:
            if load['q_load'] < 0:
                ubs[bus_name] -= load['q_load']

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] != 'fixed':
                continue
            if shunt['bs'] <= 0:
                continue
            bus_name = shunt['bus']
            if bus_name not in index_set:
                continue
            ubs[bus_name] += shunt['bs'] * md.data['elements']['bus'][bus_name]['v_max']**2

    for b in index_set:
        model.q_over_generation[b].setub(ubs[b])


def declare_var_p_load_shed(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    model.p_load_shed = pe.Var(index_set, initialize=0, bounds=(0, None))

    ubs = dict()
    for b in index_set:
        ubs[b] = 0

    for gname, gen in md.data['elements']['generator'].items():
        bus_name = gen['bus']
        if bus_name not in index_set:
            continue
        if gen['p_min'] < 0:
            ubs[bus_name] -= gen['p_min']

    for lname, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'p_load' in load:
            if load['p_load'] > 0:
                ubs[bus_name] += load['p_load']

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] != 'fixed':
                continue
            if shunt['gs'] <= 0:
                continue
            bus_name = shunt['bus']
            if bus_name not in index_set:
                continue
            ubs[bus_name] += shunt['gs'] * md.data['elements']['bus'][bus_name]['v_max']**2

    for b in index_set:
        model.p_load_shed[b].setub(ubs[b])


def declare_var_q_load_shed(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    model.q_load_shed = pe.Var(index_set, initialize=0, bounds=(0, None))

    ubs = dict()
    for b in index_set:
        ubs[b] = 0

    for gname, gen in md.data['elements']['generator'].items():
        bus_name = gen['bus']
        if bus_name not in index_set:
            continue
        if gen['q_min'] < 0:
            ubs[bus_name] -= gen['q_min']

    for lname, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'q_load' in load:
            if load['q_load'] > 0:
                ubs[bus_name] += load['q_load']

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] != 'fixed':
                continue
            if shunt['bs'] >= 0:
                continue
            bus_name = shunt['bus']
            if bus_name not in index_set:
                continue
            ubs[bus_name] -= shunt['gs'] * md.data['elements']['bus'][bus_name]['v_max']**2

    for b in index_set:
        model.q_load_shed[b].setub(ubs[b])


def declare_var_vr(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create variable for the real component of the voltage at a bus
    """
    model.vr = pe.Var(index_set)

    for b in index_set:
        bus = md.data['elements']['bus'][b]
        model.vr[b].value = bus['vm'] * math.cos(radians(bus['va']))
        if add_bounds:
            model.vr[b].setlb(-bus['v_max'])
            model.vr[b].setub(bus['v_max'])


def declare_var_vj(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create variable for the imaginary component of the voltage at a bus
    """
    model.vj = pe.Var(index_set)

    for b in index_set:
        bus = md.data['elements']['bus'][b]
        model.vj[b].value = bus['vm'] * math.sin(radians(bus['va']))
        if add_bounds:
            model.vj[b].setlb(-bus['v_max'])
            model.vj[b].setub(bus['v_max'])


def declare_var_vm(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create variable for the voltage magnitude of the voltage at a bus
    """
    model.vm = pe.Var(index_set, initialize=1)

    if add_bounds:
        for b in index_set:
            bus = md.data['elements']['bus'][b]
            model.vm[b].setlb(bus['v_min'])
            model.vm[b].setub(bus['v_max'])


def declare_var_va(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create variable for the phase angle of the voltage at a bus
    """
    model.va = pe.Var(index_set)

    for b in index_set:
        bus = md.data['elements']['bus'][b]
        model.va[b].value = radians(bus['va'])
        if add_bounds:
            model.va[b].setlb(-math.pi)
            model.va[b].setub(math.pi)


def declare_expr_vmsq(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create an expression for the voltage magnitude squared at a bus
    """
    m = model
    expr_set = decl.declare_set('_expr_vmsq', model, index_set)
    m.vmsq = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for bus in expr_set:
            m.vmsq[bus] = m.vr[bus] ** 2 + m.vj[bus] ** 2
    elif coordinate_type == CoordinateType.POLAR:
        for bus in expr_set:
            m.vmsq[bus] = m.vm[bus] ** 2


def declare_var_vmsq(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create auxiliary variable for the voltage magnitude squared at a bus
    """
    model.vmsq = pe.Var(index_set, initialize=1)

    if add_bounds:
        for bname in index_set:
            bus = md.data['elements']['bus'][bname]
            v_min = bus['v_min']
            v_max = bus['v_max']
            model.vmsq[bname].setlb(v_min**2)
            model.vmsq[bname].setub(v_max**2)


def declare_eq_vmsq(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        coordinate_type=CoordinateType.POLAR
):
    """
    Create a constraint relating vmsq to the voltages
    """
    m = model
    m.eq_vmsq = pe.Constraint(index_set)

    if coordinate_type == CoordinateType.POLAR:
        for bus in index_set:
            m.eq_vmsq[bus] = m.vmsq[bus] == m.vm[bus] ** 2
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus in index_set:
            m.eq_vmsq[bus] = m.vmsq[bus] == m.vr[bus]**2 + m.vj[bus]**2
    else:
        raise ValueError('unexpected coordinate_type: {0}'.format(str(coordinate_type)))


def declare_var_ir_aggregation_at_bus(model, index_set, **kwargs):
    """
    Create a variable for the aggregated real current at a bus
    """
    decl.declare_var('ir_aggregation_at_bus', model=model, index_set=index_set, **kwargs)


def declare_var_ij_aggregation_at_bus(model, index_set, **kwargs):
    """
    Create a variable for the aggregated imaginary current at a bus
    """
    decl.declare_var('ij_aggregation_at_bus', model=model, index_set=index_set, **kwargs)


def declare_var_pl(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        fix: bool = True,
):
    """
    Create variable for the real power load at a bus
    """
    model.pl = pe.Var(index_set)

    for bname in index_set:
        model.pl[bname].value = 0

    for load_name, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'p_load' in load:
            model.pl[bus_name].value += load['p_load']

    if fix:
        model.pl.fix()


def declare_var_ql(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        fix: bool = True,
):
    """
    Create variable for the real power load at a bus
    """
    model.ql = pe.Var(index_set)

    for bname in index_set:
        model.ql[bname].value = 0

    for load_name, load in md.data['elements']['load'].items():
        bus_name = load['bus']
        if bus_name in index_set and 'q_load' in load:
            model.ql[bus_name].value += load['q_load']

    if fix:
        model.ql.fix()


def declare_var_p_nw(model, index_set, **kwargs):
    """
    Create variable for the reactive power load at a bus
    """
    decl.declare_var('p_nw', model=model, index_set=index_set, **kwargs)


def declare_expr_shunt_power_at_bus(model, index_set, shunt_attrs,
                                    coordinate_type=CoordinateType.POLAR):
    """
    Create the expression for the shunt power at the bus
    """
    m = model
    expr_set = decl.declare_set('_expr_shunt_at_bus_set', model, index_set)

    m.shunt_p = pe.Expression(expr_set, initialize=0.0)
    m.shunt_q = pe.Expression(expr_set, initialize=0.0)

    if coordinate_type == CoordinateType.POLAR:
        for bus_name in expr_set:
            if bus_name in shunt_attrs['bus']:
                vmsq = m.vm[bus_name]**2
                m.shunt_p[bus_name] = shunt_attrs['gs'][bus_name]*vmsq
                m.shunt_q[bus_name] = -shunt_attrs['bs'][bus_name]*vmsq
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus_name in expr_set:
            if bus_name in shunt_attrs['bus']:
                vmsq = m.vr[bus_name]**2 + m.vj[bus_name]**2
                m.shunt_p[bus_name] = shunt_attrs['gs'][bus_name]*vmsq
                m.shunt_q[bus_name] = -shunt_attrs['bs'][bus_name]*vmsq

def _get_dc_dicts(dc_inlet_branches_by_bus, dc_outlet_branches_by_bus, con_set):
    if dc_inlet_branches_by_bus is None:
        assert dc_outlet_branches_by_bus is None
        dc_inlet_branches_by_bus = {bn:() for bn in con_set}
    if dc_outlet_branches_by_bus is None:
        dc_outlet_branches_by_bus = dc_inlet_branches_by_bus
    return dc_inlet_branches_by_bus, dc_outlet_branches_by_bus

def declare_expr_p_net_withdraw_at_bus(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts,
                                       dc_inlet_branches_by_bus=None, dc_outlet_branches_by_bus=None):
    """
    Create a named pyomo expression for bus net withdraw
    """
    m = model
    decl.declare_expr('p_nw', model, index_set)

    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(dc_inlet_branches_by_bus,
                                                                        dc_outlet_branches_by_bus,
                                                                        index_set)

    for b in index_set:
        m.p_nw[b] = ( bus_gs_fixed_shunts[b] 
                    + ( m.pl[b] if bus_p_loads[b] != 0.0 else 0.0 )
                    - sum( m.pg[g] for g in gens_by_bus[b] ) 
                    + sum(m.dcpf[branch_name] for branch_name in dc_outlet_branches_by_bus[b])
                    - sum(m.dcpf[branch_name] for branch_name in dc_inlet_branches_by_bus[b])
                    )
        
def declare_eq_p_net_withdraw_at_bus(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts,
                                     dc_inlet_branches_by_bus=None, dc_outlet_branches_by_bus=None):
    """
    Create a named pyomo expression for bus net withdraw
    """
    m = model
    con_set = decl.declare_set('_con_eq_p_net_withdraw_at_bus', model, index_set)

    dc_inlet_branches_by_bus, dc_outlet_branches_by_bus = _get_dc_dicts(dc_inlet_branches_by_bus,
                                                                        dc_outlet_branches_by_bus,
                                                                        index_set)

    m.eq_p_net_withdraw_at_bus = pe.Constraint(con_set)

    for b in index_set:
        m.eq_p_net_withdraw_at_bus[b] = m.p_nw[b] == ( bus_gs_fixed_shunts[b] 
                                                    + ( m.pl[b] if bus_p_loads[b] != 0.0 else 0.0 )
                                                    - sum( m.pg[g] for g in gens_by_bus[b] )
                                                    + sum(m.dcpf[branch_name] for branch_name
                                                           in dc_outlet_branches_by_bus[b])
                                                    - sum(m.dcpf[branch_name] for branch_name
                                                           in dc_inlet_branches_by_bus[b])
                                                    )
                    
def declare_eq_ref_bus_nonzero(model, ref_angle, ref_bus):
    """
    Create an equality constraint to enforce tan(\theta) = vj/vr at  the reference bus
    """
    m = model
    m.eq_ref_bus_nonzero = pe.Constraint(expr = tan(radians(ref_angle)) * m.vr[ref_bus] == m.vj[ref_bus])

def declare_eq_i_aggregation_at_bus(model, index_set,
                                    bus_bs_fixed_shunts, bus_gs_fixed_shunts,
                                    inlet_branches_by_bus, outlet_branches_by_bus):
    """
    Create the equality constraints for the aggregated real and imaginary
    currents at the bus
    """
    m = model
    con_set = decl.declare_set('_con_eq_i_aggregation_at_bus_set', model, index_set)

    m.eq_ir_aggregation_at_bus = pe.Constraint(con_set)
    m.eq_ij_aggregation_at_bus = pe.Constraint(con_set)

    for bus_name in con_set:
        ir_expr = sum([m.ifr[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        ir_expr += sum([m.itr[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])
        ij_expr = sum([m.ifj[branch_name] for branch_name in outlet_branches_by_bus[bus_name]])
        ij_expr += sum([m.itj[branch_name] for branch_name in inlet_branches_by_bus[bus_name]])

        if bus_bs_fixed_shunts[bus_name] != 0.0:
            ir_expr -= bus_bs_fixed_shunts[bus_name] * m.vj[bus_name]
            ij_expr += bus_bs_fixed_shunts[bus_name] * m.vr[bus_name]
        if bus_gs_fixed_shunts[bus_name] != 0.0:
            ir_expr += bus_gs_fixed_shunts[bus_name] * m.vr[bus_name]
            ij_expr += bus_gs_fixed_shunts[bus_name] * m.vj[bus_name]

        ir_expr -= m.ir_aggregation_at_bus[bus_name]
        ij_expr -= m.ij_aggregation_at_bus[bus_name]

        m.eq_ir_aggregation_at_bus[bus_name] = ir_expr == 0
        m.eq_ij_aggregation_at_bus[bus_name] = ij_expr == 0


def declare_eq_p_balance_ed(model, index_set, bus_p_loads, gens_by_bus, bus_gs_fixed_shunts, **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    p_expr = sum(m.pg[gen_name] for bus_name in index_set for gen_name in gens_by_bus[bus_name])
    p_expr -= sum(m.pl[bus_name] for bus_name in index_set if bus_p_loads[bus_name] is not None)
    p_expr -= sum(bus_gs_fixed_shunts[bus_name] for bus_name in index_set if bus_gs_fixed_shunts[bus_name] != 0.0)

    relaxed_balance = False

    if rhs_kwargs:
        for idx,val in rhs_kwargs.items():
            if idx == 'include_feasibility_load_shed':
                p_expr += eval("m." + val)
            if idx == 'include_feasibility_over_generation':
                p_expr -= eval("m." + val)
            if idx == 'include_losses':
                p_expr -= sum(m.pfl[branch_name] for branch_name in val)
            if idx == 'relax_balance':
                relaxed_balance = True

    if relaxed_balance:
        m.eq_p_balance = pe.Constraint(expr = p_expr >= 0.0)
    else:
        m.eq_p_balance = pe.Constraint(expr = p_expr == 0.0)

def declare_eq_p_balance_dc_approx(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        approximation_type: ApproximationType = ApproximationType.BTHETA,
):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    assert approximation_type in {ApproximationType.BTHETA,
                                  ApproximationType.BTHETA_LOSSES}
    m = model

    exprs = dict()
    for bus_name in index_set:
        exprs[bus_name] = 0

    for branch_name in m.branch_set:
        branch = md.data['elements']['branch'][branch_name]
        exprs[branch['from_bus']] -= m.pf[branch_name]
        exprs[branch['to_bus']] += m.pf[branch_name]
    if approximation_type == ApproximationType.BTHETA_LOSSES:
        for branch_name in m.branch_set:
            branch = md.data['elements']['branch'][branch_name]
            exprs[branch['from_bus']] -= 0.5 * m.pfl[branch_name]
            exprs[branch['to_bus']] -= 0.5 * m.pfl[branch_name]

    for dc_bname in m.dc_branch_set:
        dc_branch = md.data['elements']['dc_branch'][dc_bname]
        exprs[dc_branch['from_bus']] -= m.dcpf[dc_bname]
        exprs[dc_branch['to_bus']] += m.dcpf[dc_bname]

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] == 'fixed':
                if shunt['bus'] in index_set:
                    exprs[shunt['bus']] -= shunt['gs']

    for bus_name in index_set:
        exprs[bus_name] -= m.pl[bus_name]

    for bus_name in index_set:
        exprs[bus_name] += m.p_balance_slack_expr[bus_name]

    for gen_name in m.gen_set:
        gen = md.data['elements']['generator'][gen_name]
        exprs[gen['bus']] += m.pg[gen_name] * m.gen_in_service_expr[gen_name]

    m.eq_p_balance = pe.Constraint(index_set)

    for bus_name in index_set:
        m.eq_p_balance[bus_name] = exprs[bus_name] == 0


def declare_eq_p_balance(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    exprs = dict()
    for bus_name in index_set:
        exprs[bus_name] = 0

    for branch_name in m.branch_set:
        branch = md.data['elements']['branch'][branch_name]
        exprs[branch['from_bus']] -= m.pf[branch_name]
        exprs[branch['to_bus']] -= m.pt[branch_name]

    for gen_name in m.gen_set:
        gen = md.data['elements']['generator'][gen_name]
        exprs[gen['bus']] += m.pg[gen_name] * m.gen_in_service_expr[gen_name]

    for bus_name in index_set:
        exprs[bus_name] -= m.pl[bus_name]

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] == 'fixed':
                if shunt['bus'] in index_set:
                    exprs[shunt['bus']] -= shunt['gs'] * m.vmsq[shunt['bus']]

    for bus_name in index_set:
        exprs[bus_name] += m.p_balance_slack_expr[bus_name]

    m.eq_p_balance = pe.Constraint(index_set)

    for bus_name in index_set:
        m.eq_p_balance[bus_name] = exprs[bus_name] == 0


def declare_eq_q_balance(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the equality constraints for the reactive power balance
    at a bus using the variables for reactive power flows.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model

    exprs = dict()
    for bus_name in index_set:
        exprs[bus_name] = 0

    for branch_name, branch in md.data['elements']['branch'].items():
        exprs[branch['from_bus']] -= m.qf[branch_name]
        exprs[branch['to_bus']] -= m.qt[branch_name]

    for gen_name, gen in md.data['elements']['generator'].items():
        exprs[gen['bus']] += m.qg[gen_name] * m.gen_in_service_expr[gen_name]

    for bus_name in index_set:
        exprs[bus_name] -= m.ql[bus_name]

    if 'shunt' in md.data['elements']:
        for shunt_name, shunt in md.data['elements']['shunt'].items():
            if shunt['shunt_type'] == 'fixed':
                exprs[shunt['bus']] += shunt['bs'] * m.vmsq[shunt['bus']]

    for bus_name in index_set:
        exprs[bus_name] += m.q_balance_slack_expr[bus_name]

    m.eq_q_balance = pe.Constraint(index_set)

    for bus_name in index_set:
        m.eq_q_balance[bus_name] = exprs[bus_name] == 0


def declare_eq_p_balance_with_i_aggregation(model, index_set,
                                            bus_p_loads,
                                            gens_by_bus,
                                            **rhs_kwargs):
    """
    Create the equality constraints for the real power balance
    at a bus using the variables for real power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_p_balance', model, index_set)

    m.eq_p_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        p_expr = -m.vr[bus_name] * m.ir_aggregation_at_bus[bus_name] + \
                 -m.vj[bus_name] * m.ij_aggregation_at_bus[bus_name]

        if bus_p_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            p_expr -= m.pl[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    p_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    p_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            p_expr += m.pg[gen_name]

        m.eq_p_balance[bus_name] = \
            p_expr == 0.0


def declare_eq_q_balance_with_i_aggregation(model, index_set,
                                            bus_q_loads,
                                            gens_by_bus,
                                            **rhs_kwargs):
    """
    Create the equality constraints for the reactive power balance
    at a bus using the variables for reactive power flows, respectively.

    NOTE: Equation build orientates constants to the RHS in order to compute the correct dual variable sign
    """
    m = model
    con_set = decl.declare_set('_con_eq_q_balance', model, index_set)

    m.eq_q_balance = pe.Constraint(con_set)

    for bus_name in con_set:
        q_expr = m.vr[bus_name] * m.ij_aggregation_at_bus[bus_name] + \
                 -m.vj[bus_name] * m.ir_aggregation_at_bus[bus_name]

        if bus_q_loads[bus_name] != 0.0: # only applies to fixed loads, otherwise may cause an error
            q_expr -= m.ql[bus_name]

        if rhs_kwargs:
            for idx, val in rhs_kwargs.items():
                if idx == 'include_feasibility_load_shed':
                    q_expr += eval("m." + val)[bus_name]
                if idx == 'include_feasibility_over_generation':
                    q_expr -= eval("m." + val)[bus_name]

        for gen_name in gens_by_bus[bus_name]:
            q_expr += m.qg[gen_name]

        m.eq_q_balance[bus_name] = \
            q_expr == 0.0


def declare_ineq_vm_bus_lbub(model, index_set, buses, coordinate_type=CoordinateType.POLAR):
    """
    Create the inequalities for the voltage magnitudes from the
    voltage variables
    """
    m = model
    con_set = decl.declare_set('_con_ineq_vm_bus_lbub',
                               model=model, index_set=index_set)

    m.ineq_vm_bus_lb = pe.Constraint(con_set)
    m.ineq_vm_bus_ub = pe.Constraint(con_set)

    if coordinate_type == CoordinateType.POLAR:
        for bus_name in con_set:
            m.ineq_vm_bus_lb[bus_name] = \
                buses[bus_name]['v_min'] <= m.vm[bus_name]
            m.ineq_vm_bus_ub[bus_name] = \
                m.vm[bus_name] <= buses[bus_name]['v_max']
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for bus_name in con_set:
            m.ineq_vm_bus_lb[bus_name] = \
                buses[bus_name]['v_min']**2 <= m.vr[bus_name]**2 + m.vj[bus_name]**2
            m.ineq_vm_bus_ub[bus_name] = \
                m.vr[bus_name]**2 + m.vj[bus_name]**2 <= buses[bus_name]['v_max']**2
