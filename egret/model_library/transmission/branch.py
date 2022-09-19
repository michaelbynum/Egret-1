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
typically used for transmission lines
"""
import math
import pyomo.environ as pe
import egret.model_library.transmission.tx_calc as tx_calc
import egret.model_library.decl as decl
from egret.model_library.defn import FlowType, CoordinateType, ApproximationType, RelaxationType
from egret.data.data_utils import zip_items
from egret.data.model_data import ModelData
from pyomo.core.util import quicksum
from pyomo.core.expr.numeric_expr import LinearExpression
from collections import OrderedDict
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core.base.block import _BlockData
from pyomo.core.base.set import _SetData
from typing import Mapping, Optional, Sequence
from .tx_utils import get_unique_bus_pairs, get_bus_to_branch_map
from egret.data.networkx_utils import get_networkx_graph
import warnings
import networkx
import logging
from math import pi
from typing import List, Tuple, AbstractSet
from pyomo.contrib.fbbt import interval
from pyomo.common.collections.orderedset import OrderedSet
from pyomo.core.base.var import IndexedVar
try:
    import coramin
    coramin_available = True
except ImportError:
    coramin_available = False


logger = logging.getLogger(__name__)


def declare_set_unique_bus_pairs(
        m: _BlockData,
        md: ModelData,
):
    unique_bus_pairs = get_unique_bus_pairs(md)
    for fb, tb in unique_bus_pairs:
        assert (tb, fb) not in unique_bus_pairs
    m.unique_bus_pairs = pe.Set(initialize=unique_bus_pairs)


def declare_set_cycle_basis_bus_pairs(
        m: _BlockData,
        md: ModelData,
) -> List[List]:
    graph = get_networkx_graph(md)
    ref_bus = md.data['system']['reference_bus']
    cycle_basis = networkx.algorithms.cycle_basis(graph, root=ref_bus)

    cycle_basis_bus_pairs = OrderedSet()
    for cycle in cycle_basis:
        for ndx in range(len(cycle) - 1):
            b1 = cycle[ndx]
            b2 = cycle[ndx + 1]
            assert ((b1, b2) in m.unique_bus_pairs) != ((b2, b1) in m.unique_bus_pairs)
            if (b1, b2) in m.unique_bus_pairs:
                cycle_basis_bus_pairs.add((b1, b2))
            else:
                cycle_basis_bus_pairs.add((b2, b1))
        b1 = cycle[-1]
        b2 = cycle[0]
        assert ((b1, b2) in m.unique_bus_pairs) != ((b2, b1) in m.unique_bus_pairs)
        if (b1, b2) in m.unique_bus_pairs:
            cycle_basis_bus_pairs.add((b1, b2))
        else:
            cycle_basis_bus_pairs.add((b2, b1))

    m.cycle_basis_bus_pairs = pe.Set(initialize=cycle_basis_bus_pairs)

    return cycle_basis


def declare_set_branch_set(
        m: _BlockData,
        md: ModelData,
):
    branches = list(md.data['elements']['branch'].keys())
    m.branch_set = pe.Set(initialize=branches)


def declare_expression_branch_in_service_expr(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
        rule: Optional[str] = 'default'
):
    if rule not in {'default', None}:
        raise ValueError("rule should be either 'default' or None")

    m.branch_in_service_expr = pe.Expression(index_set)

    if rule == 'default':
        for b in index_set:
            branch = md.data['elements']['branch'][b]
            if branch['in_service']:
                m.branch_in_service_expr[b] = 1
            else:
                m.branch_in_service_expr[b] = 0


def declare_var_dva(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create variable or the angle difference between interconnected bus pairs
    """
    model.dva = pe.Var(index_set, initialize=0)

    if add_bounds:
        for bname, branch in md.data['elements']['branch'].items():
            from_bus = branch['from_bus']
            to_bus = branch['to_bus']
            if (from_bus, to_bus) in index_set:
                angle_min = max(-math.pi/2, math.radians(branch['angle_diff_min']))
                angle_max = min(math.pi/2, math.radians(branch['angle_diff_max']))
                if model.dva[from_bus, to_bus].has_lb():
                    angle_min = max(angle_min, model.dva[from_bus, to_bus].lb)
                if model.dva[from_bus, to_bus].has_ub():
                    angle_max = min(angle_max, model.dva[from_bus, to_bus].ub)
                model.dva[from_bus, to_bus].setlb(angle_min)
                model.dva[from_bus, to_bus].setub(angle_max)


def declare_var_pfl(model, index_set, **kwargs):
    """
    Create variable for the real part of the power loss in the transmission
    line
    """
    decl.declare_var('pfl', model=model, index_set=index_set, **kwargs)


def declare_var_pf(
        model: _BlockData,
        md: ModelData,
        index_set:_SetData,
        add_bounds: bool = True
):
    """
    Create variable for the real part of the power flow in the "from"
    end of the transmission line
    """
    model.pf = pe.Var(index_set)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            smax = branch['rating_long_term']
            if smax is not None:
                model.pf[bname].setlb(-smax)
                model.pf[bname].setub(smax)


def declare_var_pt(
        model: _BlockData,
        md: ModelData,
        index_set:_SetData,
        add_bounds: bool = True
):
    """
    Create variable for the real part of the power flow in the "to"
    end of the transmission line
    """
    model.pt = pe.Var(index_set)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            smax = branch['rating_long_term']
            if smax is not None:
                model.pt[bname].setlb(-smax)
                model.pt[bname].setub(smax)


def declare_var_qf(
        model: _BlockData,
        md: ModelData,
        index_set:_SetData,
        add_bounds: bool = True
):
    """
    Create variable for the imaginary part of the power flow in the "from"
    end of the transmission line
    """
    model.qf = pe.Var(index_set)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            smax = branch['rating_long_term']
            if smax is not None:
                model.qf[bname].setlb(-smax)
                model.qf[bname].setub(smax)


def declare_var_qt(
        model: _BlockData,
        md: ModelData,
        index_set:_SetData,
        add_bounds: bool = True
):
    """
    Create variable for the imaginary part of the power flow in the "to"
    end of the transmission line
    """
    model.qt = pe.Var(index_set)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            smax = branch['rating_long_term']
            if smax is not None:
                model.qt[bname].setlb(-smax)
                model.qt[bname].setub(smax)


def declare_expr_pf(model, index_set, **kwargs):
    """
    Create expression for the real part of the power flow in the "from"
    end of the transmission line
    """
    decl.declare_expr('pf', model=model, index_set=index_set, **kwargs)


def declare_var_pf_slack_pos(model, index_set, **kwargs):
    """
    Create the positive slack variable for the real part of power flow
    in the "from" end of the transmission line
    """
    decl.declare_var('pf_slack_pos', model=model, index_set=index_set, **kwargs)


def declare_var_pf_slack_neg(model, index_set, **kwargs):
    """
    Create the negative slack variable for the real part of power flow
    in the "from" end of the transmission line
    """
    decl.declare_var('pf_slack_neg', model=model, index_set=index_set, **kwargs)


def declare_var_pfi(model, index_set, **kwargs):
    """
    Create variable for the real part of the power flow through an interface
    """
    decl.declare_var('pfi', model=model, index_set=index_set, **kwargs)


def declare_expr_pfi(model, index_set, **kwargs):
    """
    Create expression for the real part of the power flow through an interface
    """
    decl.declare_expr('pfi', model=model, index_set=index_set, **kwargs)


def declare_var_pfi_slack_pos(model, index_set, **kwargs):
    """
    Create the positive slack variable for the real part of power flow
    through an interface
    """
    decl.declare_var('pfi_slack_pos', model=model, index_set=index_set, **kwargs)


def declare_var_pfi_slack_neg(model, index_set, **kwargs):
    """
    Create the negative slack variable for the real part of power flow
    through an interface
    """
    decl.declare_var('pfi_slack_neg', model=model, index_set=index_set, **kwargs)


def declare_var_dcpf(model, index_set, **kwargs):
    """
    Create the variable for the real power flow through a HVDC line
    """
    decl.declare_var('dcpf', model=model, index_set=index_set, **kwargs)


def declare_var_ifr(model, index_set, **kwargs):
    """
    Create variable for the real part of the current flow in the "from"
    end of the transmission line
    """
    decl.declare_var('ifr', model=model, index_set=index_set, **kwargs)


def declare_var_ifj(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the current flow in the "from"
    end of the transmission line
    """
    decl.declare_var('ifj', model=model, index_set=index_set, **kwargs)


def declare_var_itr(model, index_set, **kwargs):
    """
    Create variable for the real part of the current flow in the "to"
    end of the transmission line
    """
    decl.declare_var('itr', model=model, index_set=index_set, **kwargs)


def declare_var_itj(model, index_set, **kwargs):
    """
    Create variable for the imaginary part of the current flow in the "to"
    end of the transmission line
    """
    decl.declare_var('itj', model=model, index_set=index_set, **kwargs)


def declare_eq_branch_dva(model, index_set, branches):
    """
    Create the equality constraints for the angle difference
    in the branch

    dva = va[from_bus] - va[to_bus] + transformer_phase_shift
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_dva_set", model, index_set)

    m.eq_dva_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        shift = 0.0
        if branch['branch_type'] == 'transformer':
            shift = math.radians(branch['transformer_phase_shift'])

        m.eq_dva_branch[branch_name] = \
            m.dva[branch_name] == \
            m.va[from_bus] - m.va[to_bus] + shift


def declare_eq_delta_va(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    """
    Create the equality constraints for the angle difference
    in the branch

    dva = va[from_bus] - va[to-bus]
    """
    m = model
    m.eq_delta_va = pe.Constraint(index_set)

    for from_bus, to_bus in index_set:
        m.eq_delta_va[(from_bus, to_bus)] = m.dva[(from_bus, to_bus)] == m.va[from_bus] - m.va[to_bus]


def declare_expr_c(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create expression for the nonlinear, nonconvex term based on cosine
    of the phase angle difference (polar) or bilinear voltages (rectangular)
    """
    m = model
    expr_set = decl.declare_set('_expr_c', model, index_set)
    m.c = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for from_bus, to_bus in expr_set:
            m.c[(from_bus,to_bus)] = m.vr[from_bus]*m.vr[to_bus] + m.vj[from_bus]*m.vj[to_bus]
    elif coordinate_type == CoordinateType.POLAR:
        for from_bus, to_bus in expr_set:
            m.c[(from_bus,to_bus)] = m.vm[from_bus]*m.vm[to_bus]*pe.cos(m.va[from_bus]-m.va[to_bus])


def declare_expr_s(model, index_set, coordinate_type=CoordinateType.POLAR):
    """
    Create expression for the nonlinear, nonconvex term based on cosine
    of the phase angle difference (polar) or bilinear voltages (rectangular)
    """
    m = model
    expr_set = decl.declare_set('_expr_s', model, index_set)
    m.s = pe.Expression(expr_set)

    if coordinate_type == CoordinateType.RECTANGULAR:
        for from_bus, to_bus in expr_set:
            m.s[(from_bus,to_bus)] = m.vj[from_bus]*m.vr[to_bus] - m.vr[from_bus]*m.vj[to_bus]
    elif coordinate_type == CoordinateType.POLAR:
        for from_bus, to_bus in expr_set:
            m.s[(from_bus,to_bus)] = m.vm[from_bus]*m.vm[to_bus]*pe.sin(m.va[from_bus]-m.va[to_bus])


def declare_var_c(
    model: _BlockData,
    md: ModelData,
    index_set: _SetData,
    add_bounds=True,
):
    """
    Create an auxiliary variable for vf * vt * cos(theta_f - theta_t)
    """

    model.c = pe.Var(index_set, initialize=1)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            from_bus_name = branch['from_bus']
            to_bus_name = branch['to_bus']
            from_bus = md.data['elements']['bus'][from_bus_name]
            to_bus = md.data['elements']['bus'][to_bus_name]
            theta_bounds = (branch['angle_diff_min']*pi/180, branch['angle_diff_max']*pi/180)
            vf_bounds = (from_bus['v_min'], from_bus['v_max'])
            vt_bounds = (to_bus['v_min'], to_bus['v_max'])
            c_bounds = interval.cos(*theta_bounds)
            c_bounds = interval.mul(*vf_bounds, *c_bounds)
            c_lb, c_ub = interval.mul(*vt_bounds, *c_bounds)
            model.c[bname].setlb(c_lb)
            model.c[bname].setub(c_ub)


def declare_var_s(
    model: _BlockData,
    md: ModelData,
    index_set: _SetData,
    add_bounds=True,
):
    """
    Create an auxiliary variable for vf * vt * sin(theta_f - theta_t)
    """

    model.s = pe.Var(index_set, initialize=0)

    if add_bounds:
        for bname in index_set:
            branch = md.data['elements']['branch'][bname]
            from_bus_name = branch['from_bus']
            to_bus_name = branch['to_bus']
            from_bus = md.data['elements']['bus'][from_bus_name]
            to_bus = md.data['elements']['bus'][to_bus_name]
            theta_bounds = (branch['angle_diff_min']*pi/180, branch['angle_diff_max']*pi/180)
            vf_bounds = (from_bus['v_min'], from_bus['v_max'])
            vt_bounds = (to_bus['v_min'], to_bus['v_max'])
            s_bounds = interval.sin(*theta_bounds)
            s_bounds = interval.mul(*vf_bounds, *s_bounds)
            s_lb, s_ub = interval.mul(*vt_bounds, *s_bounds)
            model.s[bname].setlb(s_lb)
            model.s[bname].setub(s_ub)


def declare_duplicate_c_cons(
        model: _BlockData,
        md: ModelData,
        branch_set: _SetData,
):
    model.duplicate_c_cons = pe.ConstraintList()

    seen_bus_pairs = dict()

    for b in branch_set:
        branch = md.data['elements']['branch'][b]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        if (from_bus, to_bus) in seen_bus_pairs:
            other_c = seen_bus_pairs[from_bus, to_bus]
            model.duplicate_c_cons.add(model.c[b] == other_c)
        elif (to_bus, from_bus) in seen_bus_pairs:
            other_c = seen_bus_pairs[to_bus, from_bus]
            model.duplicate_c_cons.add(model.c[b] == other_c)
        else:
            seen_bus_pairs[from_bus, to_bus] = model.c[b]


def declare_duplicate_s_cons(
        model: _BlockData,
        md: ModelData,
        branch_set: _SetData,
):
    model.duplicate_s_cons = pe.ConstraintList()

    seen_bus_pairs = dict()

    for b in branch_set:
        branch = md.data['elements']['branch'][b]
        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        if (from_bus, to_bus) in seen_bus_pairs:
            other_s = seen_bus_pairs[from_bus, to_bus]
            model.duplicate_s_cons.add(model.s[b] == other_s)
        elif (to_bus, from_bus) in seen_bus_pairs:
            other_s = seen_bus_pairs[to_bus, from_bus]
            model.duplicate_s_cons.add(model.s[b] == -other_s)
        else:
            seen_bus_pairs[from_bus, to_bus] = model.s[b]


def declare_eq_c(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        coordinate_type=CoordinateType.POLAR
):
    """
    Create a constraint relating c to the voltages
    """
    m = model
    m.eq_c = pe.Constraint(index_set)

    bus_to_branch_map = get_bus_to_branch_map(md)

    seen_bus_pairs = set()

    for from_bus, to_bus in index_set:
        if (from_bus, to_bus) in seen_bus_pairs:
            continue
        if (to_bus, from_bus) in seen_bus_pairs:
            continue
        seen_bus_pairs.add((from_bus, to_bus))
        branch_name = bus_to_branch_map[from_bus, to_bus]
        if coordinate_type == CoordinateType.POLAR:
            m.eq_c[(from_bus, to_bus)] = (
                m.c[branch_name] == (
                    m.vm[from_bus] * m.vm[to_bus] * pe.cos(m.dva[(from_bus, to_bus)])
                )
            )
        elif coordinate_type == CoordinateType.RECTANGULAR:
            m.eq_c[(from_bus, to_bus)] = (
                m.c[branch_name] == (
                    m.vr[from_bus] * m.vr[to_bus] + m.vj[from_bus] * m.vj[to_bus]
                )
            )
        else:
            raise ValueError('unexpected coordinate_type: {0}'.format(str(coordinate_type)))


def declare_eq_s(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        coordinate_type: CoordinateType = CoordinateType.POLAR
):
    """
    Create a constraint relating s to the voltages
    """
    m = model
    m.eq_s = pe.Constraint(index_set)

    bus_to_branch_map = get_bus_to_branch_map(md)

    seen_bus_pairs = set()

    for from_bus, to_bus in index_set:
        if (from_bus, to_bus) in seen_bus_pairs:
            continue
        if (to_bus, from_bus) in seen_bus_pairs:
            continue
        seen_bus_pairs.add((from_bus, to_bus))
        branch_name = bus_to_branch_map[from_bus, to_bus]
        if coordinate_type == CoordinateType.POLAR:
            m.eq_s[from_bus, to_bus] = (
                m.s[branch_name] == (
                    m.vm[from_bus] * m.vm[to_bus] * pe.sin(m.dva[(from_bus, to_bus)])
                )
            )
        elif coordinate_type == CoordinateType.RECTANGULAR:
            m.eq_s[from_bus, to_bus] = (
                m.s[branch_name] == (
                    m.vj[from_bus] * m.vr[to_bus] - m.vr[from_bus] * m.vj[to_bus]
                )
            )
        else:
            raise ValueError('unexpected coordinate_type: {0}'.format(str(coordinate_type)))


def declare_eq_dva_arctan(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m = model
    m.eq_dva_arctan = pe.Constraint(index_set)

    bus_to_branch_map = get_bus_to_branch_map(md)

    for from_bus, to_bus in index_set:
        svar = m.s[bus_to_branch_map[from_bus, to_bus]]
        cvar = m.c[bus_to_branch_map[from_bus, to_bus]]
        expr = m.dva[from_bus, to_bus] == pe.atan(svar / cvar)
        m.eq_dva_arctan[from_bus, to_bus] = expr


def declare_eq_dva_cycle_sum(
        model: _BlockData,
        md: ModelData,
        cycle_basis: List[List],
        valid_bus_pairs: _SetData
):
    m = model
    m.dva_cycle_sum_set = pe.Set(initialize=list(range(len(cycle_basis))))
    m.eq_dva_cycle_sum = pe.Constraint(m.dva_cycle_sum_set)

    for con_ndx, cycle in enumerate(cycle_basis):
        expr = 0
        for ndx in range(len(cycle) - 1):
            b1 = cycle[ndx]
            b2 = cycle[ndx + 1]
            assert ((b1, b2) in valid_bus_pairs) != ((b2, b1) in valid_bus_pairs)  # exclusive or
            if (b1, b2) in valid_bus_pairs:
                expr += m.dva[b1, b2]
            else:
                expr -= m.dva[b2, b1]
        b1 = cycle[-1]
        b2 = cycle[0]
        assert ((b1, b2) in valid_bus_pairs) != ((b2, b1) in valid_bus_pairs)
        if (b1, b2) in valid_bus_pairs:
            expr += m.dva[b1, b2]
        else:
            expr -= m.dva[b2, b1]
        m.eq_dva_cycle_sum[con_ndx] = expr == 0


def declare_eq_branch_current(model, index_set, branches, coordinate_type=CoordinateType.RECTANGULAR):
    """
    Create the equality constraints for the real and imaginary current
    in the branch
    """
    assert(coordinate_type != CoordinateType.POLAR
           and "Branch current in polar coordinates not implemented.")

    m = model
    con_set = decl.declare_set("_con_eq_branch_current_set", model, index_set)

    m.eq_ifr_branch = pe.Constraint(con_set)
    m.eq_ifj_branch = pe.Constraint(con_set)
    m.eq_itr_branch = pe.Constraint(con_set)
    m.eq_itj_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        g = tx_calc.calculate_conductance(branch)
        b = tx_calc.calculate_susceptance(branch)
        bc = branch['charging_susceptance']
        tau = 1.0
        shift = 0.0

        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        g11 = g / tau**2
        g12 = (g * math.cos(shift) - b * math.sin(shift)) / tau
        g21 = (g * math.cos(shift) + b * math.sin(shift)) / tau
        g22 = g

        b11 = (b + bc / 2) / tau**2
        b12 = (b * math.cos(shift) + g*math.sin(shift)) / tau
        b21 = (b * math.cos(shift) - g*math.sin(shift)) / tau
        b22 = b + bc / 2

        m.eq_ifr_branch[branch_name] = \
            m.ifr[branch_name] == \
            g11 * m.vr[from_bus] - g12 * m.vr[to_bus] - (b11 * m.vj[from_bus] - b12 * m.vj[to_bus])

        m.eq_ifj_branch[branch_name] = \
            m.ifj[branch_name] == \
            g11 * m.vj[from_bus] - g12 * m.vj[to_bus] + (b11 * m.vr[from_bus] - b12 * m.vr[to_bus])

        m.eq_itr_branch[branch_name] = \
            m.itr[branch_name] == \
            -(g21 * m.vr[from_bus] - g22 * m.vr[to_bus] - (b21 * m.vj[from_bus] - b22 * m.vj[to_bus]))

        m.eq_itj_branch[branch_name] = \
            m.itj[branch_name] == \
            -(g21 * m.vj[from_bus] - g22 * m.vj[to_bus] + (b21 * m.vr[from_bus] - b22 * m.vr[to_bus]))


def _get_branch_params(branch: Mapping):
    g = tx_calc.calculate_conductance(branch)
    b = tx_calc.calculate_susceptance(branch)
    bc = branch['charging_susceptance']
    tau = 1.0
    shift = 0.0

    if branch['branch_type'] == 'transformer':
        tau = branch['transformer_tap_ratio']
        shift = math.radians(branch['transformer_phase_shift'])

    g11 = g / tau ** 2
    g12 = g * math.cos(shift) / tau
    g21 = g * math.sin(shift) / tau
    g22 = g

    b11 = (b + bc / 2) / tau ** 2
    b12 = b * math.cos(shift) / tau
    b21 = b * math.sin(shift) / tau
    b22 = b + bc / 2

    return g11, g12, g21, g22, b11, b12, b21, b22


def declare_eq_pf_branch(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.eq_pf_branch = pe.Constraint(index_set)
    for bname in index_set:
        branch = md.data['elements']['branch'][bname]

        from_bus = branch['from_bus']

        g11, g12, g21, g22, b11, b12, b21, b22 = _get_branch_params(branch)

        coefs = [g11, b21 - g12, -g21 - b12]
        vlist = [
            m.vmsq[from_bus],
            m.c[bname],
            m.s[bname],
        ]
        expr = LinearExpression(
            constant=0,
            linear_coefs=coefs,
            linear_vars=vlist,
        )
        rhs = expr * m.branch_in_service_expr[bname]
        m.pf[bname].value = pe.value(rhs)
        m.eq_pf_branch[bname] = m.pf[bname] == rhs


def declare_eq_pt_branch(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.eq_pt_branch = pe.Constraint(index_set)
    for bname in index_set:
        branch = md.data['elements']['branch'][bname]

        to_bus = branch['to_bus']

        g11, g12, g21, g22, b11, b12, b21, b22 = _get_branch_params(branch)

        coefs = [g22, -g12 - b21, b12 - g21]
        vlist = [
            m.vmsq[to_bus],
            m.c[bname],
            m.s[bname],
        ]
        expr = LinearExpression(
            constant=0,
            linear_coefs=coefs,
            linear_vars=vlist,
        )
        rhs = expr * m.branch_in_service_expr[bname]
        m.pt[bname].value = pe.value(rhs)
        m.eq_pt_branch[bname] = m.pt[bname] == rhs


def declare_eq_qf_branch(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.eq_qf_branch = pe.Constraint(index_set)
    for bname in index_set:
        branch = md.data['elements']['branch'][bname]

        from_bus = branch['from_bus']

        g11, g12, g21, g22, b11, b12, b21, b22 = _get_branch_params(branch)

        coefs = [-b11, b12 + g21, b21 - g12]
        vlist = [
            m.vmsq[from_bus],
            m.c[bname],
            m.s[bname],
        ]
        expr = LinearExpression(
            constant=0,
            linear_coefs=coefs,
            linear_vars=vlist,
        )
        rhs = expr * m.branch_in_service_expr[bname]
        m.qf[bname].value = pe.value(rhs)
        m.eq_qf_branch[bname] = m.qf[bname] == rhs


def declare_eq_qt_branch(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.eq_qt_branch = pe.Constraint(index_set)
    for bname in index_set:
        branch = md.data['elements']['branch'][bname]

        to_bus = branch['to_bus']

        g11, g12, g21, g22, b11, b12, b21, b22 = _get_branch_params(branch)

        coefs = [-b22, b12 - g21, b21 + g12]
        vlist = [
            m.vmsq[to_bus],
            m.c[bname],
            m.s[bname],
        ]
        expr = LinearExpression(
            constant=0,
            linear_coefs=coefs,
            linear_vars=vlist,
        )
        rhs = expr * m.branch_in_service_expr[bname]
        m.qt[bname].value = pe.value(rhs)
        m.eq_qt_branch[bname] = m.qt[bname] == rhs


def declare_eq_branch_power(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the equality constraints for the real and reactive power
    in the branch
    """
    declare_eq_pf_branch(model, md, index_set)
    declare_eq_pt_branch(model, md, index_set)
    declare_eq_qf_branch(model, md, index_set)
    declare_eq_qt_branch(model, md, index_set)


def declare_ineq_soc(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        use_outer_approximation=False
):
    """
    create the constraint for the second order cone
    """
    m = model

    bus_to_branch_map = get_bus_to_branch_map(md)

    if not use_outer_approximation:
        m.ineq_soc = pe.Constraint(index_set)
        for from_bus, to_bus in index_set:
            branch_name = bus_to_branch_map[from_bus, to_bus]
            m.ineq_soc[(from_bus, to_bus)] = m.c[branch_name] ** 2 + m.s[branch_name] ** 2 <= m.vmsq[
                from_bus] * m.vmsq[to_bus]
    else:
        if not coramin_available:
            raise ImportError('Cannot create SOC relaxation with outer approximation unless coramin is available.')
        """
        in order to use outer approximation, we have to reformulate 

        c**2 + s**2 <= vmsq[from_bus] * vmsq[to_bus]

        to

        (c**2 + s**2 + z1**2) ** 0.5 <= z2
        z1 = 0.5 * (vmsq[from_bus] - vmsq[to_bus])
        z2 = 0.5 * (vmsq[from_bus] + vmsq[to_bus]) 
        """
        m._z1 = pe.Var(index_set)
        m._z2 = pe.Var(index_set)
        m._eq_z1 = pe.Constraint(index_set)
        m._eq_z2 = pe.Constraint(index_set)
        m.ineq_soc_OA = coramin.relaxations.MultivariateRelaxation(index_set)
        for from_bus, to_bus in index_set:
            m._eq_z1[from_bus, to_bus] = m._z1[from_bus, to_bus] == 0.5 * (m.vmsq[from_bus] - m.vmsq[to_bus])
            m._eq_z2[from_bus, to_bus] = m._z2[from_bus, to_bus] == 0.5 * (m.vmsq[from_bus] + m.vmsq[to_bus])
            fbbt(m._eq_z1[from_bus, to_bus])
            fbbt(m._eq_z2[from_bus, to_bus])
            branch_name = bus_to_branch_map[from_bus, to_bus]
            m.ineq_soc_OA[from_bus, to_bus].build(aux_var=m._z2[from_bus, to_bus],
                                                  shape=coramin.utils.FunctionShape.CONVEX,
                                                  f_x_expr=(m.c[branch_name] ** 2 +
                                                            m.s[branch_name] ** 2 +
                                                            m._z1[from_bus, to_bus] ** 2) ** 0.5)


def declare_ineq_soc_ub(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    create the constraint for the second order cone
    """
    m = model
    m.ineq_soc_ub = pe.Constraint(index_set)
    bus_to_branch_map = get_bus_to_branch_map(md)
    for from_bus, to_bus in index_set:
        branch_name = bus_to_branch_map[from_bus, to_bus]
        m.ineq_soc_ub[(from_bus, to_bus)] = (m.c[branch_name] ** 2 + m.s[branch_name] ** 2 >=
                                             m.vmsq[from_bus] * m.vmsq[to_bus])


def declare_eq_branch_power_btheta_approx(model, index_set, branches, approximation_type=ApproximationType.BTHETA):
    """
    Create the equality constraints for power (from BTHETA approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_power_btheta_approx_set", model, index_set)


    m.eq_pf_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        from_bus = branch['from_bus']
        to_bus = branch['to_bus']

        tau = 1.0
        shift = 0.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
            shift = math.radians(branch['transformer_phase_shift'])

        if approximation_type == ApproximationType.BTHETA:
            x = branch['reactance']
            b = -1/(tau*x)
        elif approximation_type == ApproximationType.BTHETA_LOSSES:
            b = tx_calc.calculate_susceptance(branch)/tau

        m.eq_pf_branch[branch_name] = \
            m.pf[branch_name] == \
            b * (m.va[from_bus] - m.va[to_bus] + shift)


def declare_eq_branch_loss_btheta_approx(model, index_set, branches, relaxation_type = RelaxationType.NONE):
    """
    Create the equality constraints for losses (from BTHETA approximation)
    in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_btheta_approx_set", model, index_set)

    m.eq_pfl_branch = pe.Constraint(con_set)
    for branch_name in con_set:
        branch = branches[branch_name]

        tau = 1.0
        if branch['branch_type'] == 'transformer':
            tau = branch['transformer_tap_ratio']
        g = tx_calc.calculate_conductance(branch)/tau

        if relaxation_type == RelaxationType.NONE:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] == \
                g * (m.dva[branch_name])**2
        elif relaxation_type == RelaxationType.SOC:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] >= \
                g * (m.dva[branch_name])**2


def declare_eq_interface_power_btheta_approx(model, index_set, interfaces):
    """
    Create the equality constraints for interface real power flow
    """
    m = model
    con_set = decl.declare_set("_con_eq_interface_power_btheta_approx", model, index_set)

    m.eq_pf_interface = pe.Constraint(con_set)
    for interface_name in con_set:
        interface = interfaces[interface_name]

        expr = 0.
        for line, orientation in zip(interface['lines'], interface['line_orientation']):
            ### the later case could happen
            ### if the line is out of service
            if orientation == 0 or line not in m.pf:
                continue
            elif orientation == 1:
                expr += m.pf[line]
            elif orientation == -1:
                expr -= m.pf[line]
            else:
                raise Exception("line_orientation must be in [-1,0,1], found "\
                        "line_orientation {} for line {} in interface {}".format(
                            orientation, line, interface_name))
        m.eq_pf_interface[interface_name] = m.pfi[interface_name] == expr


def get_power_flow_expr_ptdf_approx(model, branch_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix
    """

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_branch_const(branch_name)

    max_coef = PTDF.get_branch_ptdf_abs_max(branch_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef) 
    ## NOTE: It would be easy to hold on to the 'ptdf' dictionary here,
    ##       if we wanted to
    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_branch_ptdf_iterator(branch_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    if isinstance(m_p_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr


def declare_eq_branch_power_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create the equality constraints or expressions for power (from PTDF 
    approximation) in the branch
    """

    m = model

    con_set = decl.declare_set("_con_eq_branch_power_ptdf_approx_set", model, index_set)

    pf_is_var = isinstance(m.pf, pe.Var)

    if pf_is_var:
        m.eq_pf_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pf, pe.Expression):
            raise Exception("Unrecognized type for m.pf", m.pf.pprint())

    for branch_name in con_set:
        expr = \
            get_power_flow_expr_ptdf_approx(m, branch_name, PTDF, rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        if pf_is_var:
            m.eq_pf_branch[branch_name] = \
                m.pf[branch_name] == expr
        else:
            m.pf[branch_name] = expr


def get_branch_loss_expr_ptdf_approx(model, branch_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None): 
    """
    Create a pyomo power flow loss expression from PTDF matrix
    """
    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_branch_losses_phase_shift(branch_name)
    const += PTDF.get_branch_ldf_c(branch_name)
    const += PTDF.get_branch_phi_losses_adj(branch_name)

    max_coef = PTDF.get_branch_ldf_abs_max(branch_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef) 
    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_branch_ldf_iterator(branch_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    if isinstance(m_p_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr


def declare_eq_branch_loss_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create the equality constraints or expressions for losses (from PTDF 
    approximation) in the branch
    """
    m = model

    con_set = decl.declare_set("_con_eq_branch_loss_ptdf_approx_set", model, index_set)
    pfl_is_var = isinstance(m.pfl, pe.Var)
    if pfl_is_var:
        m.eq_pfl_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pfl, pe.Expression):
            raise Exception("Unrecognized type for m.pfl", m.pfl.pprint())

    for branch_name in con_set:
        expr = \
            get_branch_loss_expr_ptdf_approx(m, branch_name, PTDF, rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        if pfl_is_var:
            m.eq_pfl_branch[branch_name] = \
                m.pfl[branch_name] == expr
        else:
            m.pfl[branch_name] = expr


def get_contingency_power_flow_expr_ptdf_approx(model, contingency_name, branch_name, PTDF,
                                                rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix for a contingency
    """

    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_contingency_branch_const(contingency_name, branch_name)

    max_coef = PTDF.get_contingency_branch_ptdf_abs_max(contingency_name, branch_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)
    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_contingency_branch_ptdf_iterator(contingency_name, branch_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    if isinstance(m_p_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr


def declare_eq_contingency_branch_power_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create the equality constraints or expressions for power (from PTDF
    approximation) in the branch under contingency
    """

    m = model

    con_set = decl.declare_set("_con_eq_contingency_branch_power_ptdf_approx_set", model, index_set)

    pfc_is_var = isinstance(m.pfc, pe.Var)

    if pfc_is_var:
        m.eq_pfc_branch = pe.Constraint(con_set)
    else:
        if not isinstance(m.pf, pe.Expression):
            raise Exception("Unrecognized type for m.pf", m.pf.pprint())

    for contingency_name, branch_name in con_set:
        expr = \
            get_contingency_power_flow_expr_ptdf_approx(m, contingency_name, branch_name, PTDF, rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        if pfc_is_var:
            m.eq_pfc_branch[branch_name] = \
                m.pfc[branch_name] == expr
        else:
            m.pfc[branch_name] = expr


def get_power_flow_interface_expr_ptdf(model, interface_name, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create a pyomo power flow expression from PTDF matrix for an interface
    """
    if rel_ptdf_tol is None:
        rel_ptdf_tol = 0.
    if abs_ptdf_tol is None:
        abs_ptdf_tol = 0.

    const = PTDF.get_interface_const(interface_name)
    max_coef = PTDF.get_interface_ptdf_abs_max(interface_name)

    ptdf_tol = max(abs_ptdf_tol, rel_ptdf_tol*max_coef)

    m_p_nw = model.p_nw
    ## if model.p_nw is Var, we can use LinearExpression
    ## to build these dense constraints much faster
    coef_list = []
    var_list = []
    for bus_name, coef in PTDF.get_interface_ptdf_iterator(interface_name):
        if abs(coef) >= ptdf_tol:
            coef_list.append(coef)
            var_list.append(m_p_nw[bus_name])

    if isinstance(m_p_nw, pe.Var):
        expr = LinearExpression(linear_vars=var_list, linear_coefs=coef_list, constant=const)
    else:
        expr = quicksum( (coef*var for coef, var in zip(coef_list, var_list)), start=const, linear=True)

    return expr


def declare_eq_interface_power_ptdf_approx(model, index_set, PTDF, rel_ptdf_tol=None, abs_ptdf_tol=None):
    """
    Create equality constraints or expressions for power (from PTDF
    approximation) across the interface
    """

    m = model
    con_set = decl.declare_set("_con_eq_interface_power_ptdf_approx_set", model, index_set)

    pfi_is_var = isinstance(m.pfi, pe.Var)

    if pfi_is_var:
        m.eq_pf_interface = pe.Constraint(con_set)
    else:
        if not isinstance(m.pfi, pe.Expression):
            raise Exception("Unrecognized type for m.pfi", m.pfi.pprint())

    for interface_name in con_set:
        expr = \
            get_power_flow_interface_expr_ptdf(m, interface_name, PTDF,
                    rel_ptdf_tol=rel_ptdf_tol, abs_ptdf_tol=abs_ptdf_tol)

        if pfi_is_var:
            m.eq_pf_interface[interface_name] = \
                    m.pfi[interface_name] == expr
        else:
            m.pfi[interface_name] = expr


def declare_ineq_sf_branch_thermal_limit(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m = model
    m.ineq_sf_branch_thermal_limit = pe.Constraint(index_set)

    for b in index_set:
        branch = md.data['elements']['branch'][b]
        smax = branch['rating_long_term']
        if smax is None:
            continue
        m.ineq_sf_branch_thermal_limit[b] = m.pf[b]**2 + m.qf[b]**2 <= smax**2


def declare_ineq_st_branch_thermal_limit(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m = model
    m.ineq_st_branch_thermal_limit = pe.Constraint(index_set)

    for b in index_set:
        branch = md.data['elements']['branch'][b]
        smax = branch['rating_long_term']
        if smax is None:
            continue
        m.ineq_st_branch_thermal_limit[b] = m.pt[b]**2 + m.qt[b]**2 <= smax**2


def declare_ineq_s_branch_thermal_limit(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    declare_ineq_sf_branch_thermal_limit(model, md, index_set)
    declare_ineq_st_branch_thermal_limit(model, md, index_set)


def declare_ineq_s_branch_current_limit(model, index_set,
                                        branches, s_thermal_limits,
                                        flow_type=FlowType.POWER):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_s_branch_thermal_limit',
                               model=model, index_set=index_set)

    m.ineq_sf_branch_current_limit = pe.Constraint(con_set)
    m.ineq_st_branch_current_limit = pe.Constraint(con_set)

    assert flow_type == FlowType.CURRENT

    for branch_name in con_set:
        if s_thermal_limits[branch_name] is None:
            continue

        from_bus = branches[branch_name]['from_bus']
        to_bus = branches[branch_name]['to_bus']
        m.ineq_sf_branch_current_limit[branch_name] = \
            (m.vr[from_bus] ** 2 + m.vj[from_bus] ** 2) * (m.ifr[branch_name] ** 2 + m.ifj[branch_name] ** 2) \
            <= s_thermal_limits[branch_name] ** 2
        m.ineq_st_branch_current_limit[branch_name] = \
            (m.vr[to_bus] ** 2 + m.vj[to_bus] ** 2) * (m.itr[branch_name] ** 2 + m.itj[branch_name] ** 2) \
            <= s_thermal_limits[branch_name] ** 2


def declare_ineq_p_branch_thermal_lbub(model, index_set,
                                        branches, p_thermal_limits,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create the inequality constraints for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_p_branch_thermal_lbub',
                               model=model, index_set=index_set)

    # flag for if slacks are on the model
    if slacks:
        if not hasattr(model, 'pf_slack_pos'):
            raise Exception('No positive slack branch variables on model, but slacks=True')
        if not hasattr(model, 'pf_slack_neg'):
            raise Exception('No negative slack branch variables on model, but slacks=True')
        if slack_cost_expr is None:
            raise Exception('No cost expression for slacks, but slacks=True')

    m.ineq_pf_branch_thermal_lb = pe.Constraint(con_set)
    m.ineq_pf_branch_thermal_ub = pe.Constraint(con_set)

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for branch_name in con_set:
            if p_thermal_limits[branch_name] is None:
                continue

            if slacks and branch_name in m.pf_slack_neg.index_set():
                assert branch_name in m.pf_slack_pos.index_set()
                neg_slack = m.pf_slack_neg[branch_name]
                pos_slack = m.pf_slack_pos[branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model.TimePeriodLengthHours*uc_model.BranchLimitPenalty[branch_name] *
                                         (neg_slack + pos_slack) )
                assert len(m.pf_slack_pos) == len(m.pf_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            if neg_slack is not None:
                pf_bn = m.pf[branch_name]
                if hasattr(pf_bn, 'expr') and isinstance(pf_bn.expr, LinearExpression):
                    ## create a copy
                    old_expr = pf_bn.expr
                    expr = LinearExpression(constant=old_expr.constant,
                                            linear_vars = old_expr.linear_vars[:] + [neg_slack],
                                            linear_coefs = old_expr.linear_coefs[:] + [1],
                                            )

                else:
                    expr = m.pf[branch_name] + neg_slack
                m.ineq_pf_branch_thermal_lb[branch_name] = \
                    (-p_thermal_limits[branch_name], expr, None)
            else:
                m.ineq_pf_branch_thermal_lb[branch_name] = \
                    (-p_thermal_limits[branch_name], m.pf[branch_name], None)

            if pos_slack is not None:
                pf_bn = m.pf[branch_name]
                if hasattr(pf_bn, 'expr') and isinstance(pf_bn.expr, LinearExpression):
                    ## create a copy
                    old_expr = pf_bn.expr
                    expr = LinearExpression(constant=old_expr.constant,
                                            linear_vars = old_expr.linear_vars[:] + [pos_slack],
                                            linear_coefs = old_expr.linear_coefs[:] + [-1],
                                            )
                else:
                    expr = m.pf[branch_name] - pos_slack
                m.ineq_pf_branch_thermal_ub[branch_name] = \
                    (None, expr, p_thermal_limits[branch_name])
            else:
                m.ineq_pf_branch_thermal_ub[branch_name] = \
                    (None, m.pf[branch_name], p_thermal_limits[branch_name])

def generate_thermal_bounds(pf, llimit, ulimit, neg_slack=None, pos_slack=None):
    """
    Create a constraint for thermal limits on a line given the power flow
    expression or variable pf, a lower limit llimit, a uppder limit ulimit,
    and the negative slack variable, neg_slack, (None if not needed) and
    positive slack variable, pos_slack, (None if not needed) added to this 
    constraint.
    """
    if hasattr(pf, 'expr') and isinstance(pf.expr, LinearExpression):
        ## if necessary, copy again, so that m.pf[bn] **is** the flow
        add_vars = list()
        add_coefs = list()
        if neg_slack is not None:
            add_vars.append(neg_slack)
            add_coefs.append(1)
        if pos_slack is not None:
            add_vars.append(pos_slack)
            add_coefs.append(-1)
        if add_vars:
            ## create a copy
            old_expr = pf.expr
            expr = LinearExpression(constant = old_expr.constant,
                                    linear_vars = old_expr.linear_vars[:] + add_vars,
                                    linear_coefs = old_expr.linear_coefs[:] + add_coefs,
                                   )
        else:
            expr = pf
    else:
        expr = pf
        if neg_slack is not None:
            expr += neg_slack
        if pos_slack is not None:
            expr -= pos_slack
    return (llimit, expr, ulimit)

def declare_ineq_p_branch_thermal_bounds(model, index_set,
                                        branches, p_thermal_limits,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_p_branch_thermal_bounds',
                               model=model, index_set=index_set)
    # flag for if slacks are on the model
    if slacks:
        if not hasattr(model, 'pf_slack_pos'):
            raise Exception('No positive slack branch variables on model, but slacks=True')
        if not hasattr(model, 'pf_slack_neg'):
            raise Exception('No negative slack branch variables on model, but slacks=True')
        if slack_cost_expr is None:
            raise Exception('No cost expression for slacks, but slacks=True')

    m.ineq_pf_branch_thermal_bounds = pe.Constraint(con_set)

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for branch_name in con_set:
            limit = p_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and branch_name in m.pf_slack_neg.index_set():
                assert branch_name in m.pf_slack_pos.index_set()
                neg_slack = m.pf_slack_neg[branch_name]
                pos_slack = m.pf_slack_pos[branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model.TimePeriodLengthHours*uc_model.BranchLimitPenalty[branch_name] *
                                    (neg_slack + pos_slack) )
                assert len(m.pf_slack_pos) == len(m.pf_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            m.ineq_pf_branch_thermal_bounds[branch_name] = \
                    generate_thermal_bounds(m.pf[branch_name], -limit, limit, neg_slack, pos_slack)

def declare_ineq_p_contingency_branch_thermal_bounds(model, index_set,
                                                     pc_thermal_limits,
                                                     approximation_type=ApproximationType.PTDF,
                                                     slacks=False, slack_cost_expr=None):
    """
    Create an inequality constraint for the branch thermal limits
    based on the power variables or expressions.
    """
    m = model
    # flag for if slacks are on the model
    if slacks:
        if not hasattr(model, 'pfc_slack_pos'):
            raise Exception('No positive slack branch variables on model, but slacks=True')
        if not hasattr(model, 'pfc_slack_neg'):
            raise Exception('No negative slack branch variables on model, but slacks=True')
        if slack_cost_expr is None:
            raise Exception('No cost expression for slacks, but slacks=True')

    m.ineq_pf_contingency_branch_thermal_bounds = pe.Constraint(index_set)

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for (contingency_name, branch_name) in con_set:
            limit = pc_thermal_limits[branch_name]
            if limit is None:
                continue

            if slacks and (contingency_name, branch_name) in m.pfc_slack_neg.index_set():
                assert (contingency_name, branch_name) in m.pfc_slack_pos.index_set()
                neg_slack = m.pfc_slack_neg[contingency_name, branch_name]
                pos_slack = m.pfc_slack_pos[contingency_name, branch_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model.TimePeriodLengthHours
                                         * uc_model.SystemContingencyLimitPenalty
                                         * (neg_slack + pos_slack) )
                assert len(m.pfc_slack_pos) == len(m.pfc_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            m.ineq_pf_contingency_branch_thermal_bounds[contingency_name, branch_name] = \
                    generate_thermal_bounds(m.pfc[contingency_name, branch_name], -limit, limit, neg_slack, pos_slack)


def declare_ineq_angle_diff_branch_lb_c_s(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the inequality constraints for the angle difference
    bounds between interconnected buses.
    """
    m = model

    m.ineq_angle_diff_branch_lb = clb = pe.Constraint(index_set)

    for bname in index_set:
        branch = md.data['elements']['branch'][bname]
        angle_min = math.radians(branch['angle_diff_min'])

        if angle_min <= -math.pi/2:
            continue
        if angle_min < math.radians(-89):
            msg = 'angle difference limits larger than 89 degrees will introduce large coefficients'
            logger.warning(msg)
            warnings.warn(msg)
        clb[bname] = (
                math.tan(angle_min) * m.c[bname] <=
                m.s[bname] * m.branch_in_service_expr[bname]
        )


def declare_ineq_angle_diff_branch_ub_c_s(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the inequality constraints for the angle difference
    bounds between interconnected buses.
    """
    m = model

    m.ineq_angle_diff_branch_ub = cub = pe.Constraint(index_set)

    for bname in index_set:
        branch = md.data['elements']['branch'][bname]
        angle_max = math.radians(branch['angle_diff_max'])

        if angle_max >= math.pi/2:
            continue
        if angle_max > math.radians(89):
            msg = 'angle difference limits larger than 89 degrees will introduce large coefficients'
            logger.warning(msg)
            warnings.warn(msg)
        cub[bname] = (
                m.s[bname] * m.branch_in_service_expr[bname] <=
                math.tan(angle_max) * m.c[bname]
        )


def declare_ineq_angle_diff_branch_lbub_c_s(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    """
    Create the inequality constraints for the angle difference
    bounds between interconnected buses.
    """
    declare_ineq_angle_diff_branch_lb_c_s(model, md, index_set)
    declare_ineq_angle_diff_branch_ub_c_s(model, md, index_set)


def declare_ineq_angle_diff_branch_lbub(model, index_set, branches, coordinate_type=CoordinateType.POLAR):
    """
    Create the inequality constraints for the angle difference
    bounds between interconnected buses.
    """
    m = model
    con_set = decl.declare_set('_con_ineq_angle_diff_branch_lbub',
                               model=model, index_set=index_set)

    m.ineq_angle_diff_branch_lb = pe.Constraint(con_set)
    m.ineq_angle_diff_branch_ub = pe.Constraint(con_set)

    if coordinate_type == CoordinateType.POLAR:
        for branch_name in con_set:
            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']

            m.ineq_angle_diff_branch_lb[branch_name] = \
                math.radians(branches[branch_name]['angle_diff_min']) <= m.va[from_bus] - m.va[to_bus]
            m.ineq_angle_diff_branch_ub[branch_name] = \
                m.va[from_bus] - m.va[to_bus] <= math.radians(branches[branch_name]['angle_diff_max'])
    elif coordinate_type == CoordinateType.RECTANGULAR:
        for branch_name in con_set:
            from_bus = branches[branch_name]['from_bus']
            to_bus = branches[branch_name]['to_bus']

            if branches[branch_name]['angle_diff_min'] > -90:
                if branches[branch_name]['angle_diff_min'] < -89:
                    msg = 'angle difference limits larger than 89 will introduce large coefficients'
                    logger.warning(msg)
                    warnings.warn(msg)
                m.ineq_angle_diff_branch_lb[branch_name] = (math.tan(math.radians(branches[branch_name]['angle_diff_min'])) *
                                                            (m.vr[from_bus] * m.vr[to_bus] + m.vj[from_bus] * m.vj[to_bus]) <=
                                                            m.vj[from_bus] * m.vr[to_bus] - m.vr[from_bus] * m.vj[to_bus])
            if branches[branch_name]['angle_diff_max'] < 90:
                if branches[branch_name]['angle_diff_min'] > 89:
                    msg = 'angle difference limits larger than 89 will introduce large coefficients'
                    logger.warning(msg)
                    warnings.warn(msg)
                m.ineq_angle_diff_branch_ub[branch_name] = (m.vj[from_bus] * m.vr[to_bus] - m.vr[from_bus] * m.vj[to_bus] <=
                                                            math.tan(math.radians(branches[branch_name]['angle_diff_max'])) *
                                                            (m.vr[from_bus] * m.vr[to_bus] + m.vj[from_bus] * m.vj[to_bus]))


def declare_ineq_p_interface_bounds(model, index_set, interfaces,
                                        approximation_type=ApproximationType.BTHETA,
                                        slacks=False, slack_cost_expr=None):
    """
    Create the inequality constraints for the interface limits
    based on the power variables or expressions.

    p_interface_limits should be (lower, upper) tuple
    """
    m = model
    con_set = decl.declare_set('_con_ineq_p_interface_bounds',
                               model=model, index_set=index_set)

    m.ineq_pf_interface_bounds = pe.Constraint(con_set)

    # flag for if slacks are on the model
    if slacks:
        if not hasattr(model, 'pfi_slack_pos'):
            raise Exception('No positive slack interface variables on model, but slacks=True')
        if not hasattr(model, 'pfi_slack_neg'):
            raise Exception('No negative slack interface variables on model, but slacks=True')
        if slack_cost_expr is None:
            raise Exception('No cost expression for slacks, but slacks=True')

    if approximation_type == ApproximationType.BTHETA or \
            approximation_type == ApproximationType.PTDF:
        for interface_name in con_set:
            interface = interfaces[interface_name]
            if interface['minimum_limit'] is None and \
                    interface['maximum_limit'] is None:
                continue

            if slacks and interface_name in m.pfi_slack_neg.index_set():
                assert interface_name in m.pfi_slack_pos.index_set()
                neg_slack = m.pfi_slack_neg[interface_name]
                pos_slack = m.pfi_slack_pos[interface_name]
                uc_model = slack_cost_expr.parent_block()
                slack_cost_expr.expr += (uc_model.TimePeriodLengthHours*uc_model.InterfaceLimitPenalty[interface_name] *
                                    (neg_slack + pos_slack) )
                assert len(m.pfi_slack_pos) == len(m.pfi_slack_neg)
            else:
                neg_slack = None
                pos_slack = None

            m.ineq_pf_interface_bounds[interface_name] = \
                generate_thermal_bounds(m.pfi[interface_name], interface['minimum_limit'], interface['maximum_limit'],
                                        neg_slack, pos_slack)
