#  ___________________________________________________________________________
#
#  EGRET: Electrical Grid Research and Engineering Tools
#  Copyright 2019 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

"""
This module contains the modeling components used when modeling generators
in transmission models
"""
import pyomo.environ as pe
import egret.model_library.decl as decl
from egret.model_library.transmission import tx_utils
from pyomo.core.expr.numeric_expr import LinearExpression
from pyomo.core.base.block import _BlockData
from pyomo.core.base.set import _SetData
from egret.data.model_data import ModelData
from typing import Optional, Union
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import IndexedConstraint
from pyomo.core.expr.numvalue import NumericValue


def declare_set_gen_set(
        m: _BlockData,
        md: ModelData,
):
    gens = list(md.data['elements']['generator'].keys())
    m.gen_set = pe.Set(initialize=gens)


def declare_pw_p_cost_gen_set(
        m: _BlockData,
        md: ModelData,
):
    pw_p_cost_gens = list()
    for gen_name, gen in md.data['elements']['generator'].items():
        if 'p_cost' in gen and gen['p_cost']['cost_curve_type'] == 'piecewise':
            pw_p_cost_gens.append(gen_name)

    m.pw_p_cost_gen_set = pe.Set(initialize=pw_p_cost_gens)


def declare_poly_p_cost_gen_set(
        m: _BlockData,
        md: ModelData,
):
    poly_p_cost_gens = list()
    for gen_name, gen in md.data['elements']['generator'].items():
        if 'p_cost' in gen and gen['p_cost']['cost_curve_type'] == 'polynomial':
            poly_p_cost_gens.append(gen_name)

    m.poly_p_cost_gen_set = pe.Set(initialize=poly_p_cost_gens)


def declare_pw_q_cost_gen_set(
        m: _BlockData,
        md: ModelData,
):
    pw_q_cost_gens = list()
    for gen_name, gen in md.data['elements']['generator'].items():
        if 'q_cost' in gen and gen['q_cost']['cost_curve_type'] == 'piecewise':
            pw_q_cost_gens.append(gen_name)

    m.pw_q_cost_gen_set = pe.Set(initialize=pw_q_cost_gens)


def declare_poly_q_cost_gen_set(
        m: _BlockData,
        md: ModelData,
):
    poly_q_cost_gens = list()
    for gen_name, gen in md.data['elements']['generator'].items():
        if 'q_cost' in gen and gen['q_cost']['cost_curve_type'] == 'polynomial':
            poly_q_cost_gens.append(gen_name)

    m.poly_q_cost_gen_set = pe.Set(initialize=poly_q_cost_gens)


def declare_expression_gen_in_service_expr(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
        rule: Optional[str] = 'default'
):
    if rule not in {'default', None}:
        raise ValueError("rule should be either 'default' or None")

    m.gen_in_service_expr = pe.Expression(index_set)

    if rule == 'default':
        for g in index_set:
            gen = md.data['elements']['generator'][g]
            if gen['in_service']:
                m.gen_in_service_expr[g] = 1
            else:
                m.gen_in_service_expr[g] = 0


def declare_var_pg(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create auxiliary variable for the voltage magnitude squared at a bus
    """
    model.pg = pe.Var(index_set)

    for gname in index_set:
        gen = md.data['elements']['generator'][gname]
        pmin = gen['p_min']
        pmax = gen['p_max']
        if add_bounds:
            model.pg[gname].setlb(pmin)
            model.pg[gname].setub(pmax)
        model.pg[gname].value = 0.5 * (pmin + pmax)


def declare_var_qg(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        add_bounds: bool = True,
):
    """
    Create auxiliary variable for the voltage magnitude squared at a bus
    """
    model.qg = pe.Var(index_set)

    for gname in index_set:
        gen = md.data['elements']['generator'][gname]
        qmin = gen['q_min']
        qmax = gen['q_max']
        if add_bounds:
            model.qg[gname].setlb(qmin)
            model.qg[gname].setub(qmax)
        model.qg[gname].value = 0.5 * (qmin + qmax)


def declare_ineq_pg_lb(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.ineq_pg_lb = pe.Constraint(index_set)

    for g in index_set:
        pmin = md.data['elements']['generator'][g]['p_min']
        m.ineq_pg_lb[g] = pmin * m.gen_in_service_expr[g] <= m.pg[g]


def declare_ineq_pg_ub(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.ineq_pg_ub = pe.Constraint(index_set)

    for g in index_set:
        pmax = md.data['elements']['generator'][g]['p_max']
        m.ineq_pg_ub[g] = pmax * m.gen_in_service_expr[g] >= m.pg[g]


def declare_ineq_qg_lb(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.ineq_qg_lb = pe.Constraint(index_set)

    for g in index_set:
        qmin = md.data['elements']['generator'][g]['q_min']
        m.ineq_qg_lb[g] = qmin * m.gen_in_service_expr[g] <= m.qg[g]


def declare_ineq_qg_ub(
        m: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m.ineq_qg_ub = pe.Constraint(index_set)

    for g in index_set:
        qmax = md.data['elements']['generator'][g]['q_max']
        m.ineq_qg_ub[g] = qmax * m.gen_in_service_expr[g] >= m.qg[g]


def pw_gen_generator(index_set, costs):
    for gen_name in index_set:
        if gen_name not in costs:
            continue
        curve = costs[gen_name]
        assert curve['cost_curve_type'] in {'piecewise', 'polynomial'}
        if curve['cost_curve_type'] != 'piecewise':
            continue
        yield gen_name


def declare_var_delta_pg(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
):
    m = model

    m.delta_pg_set = pe.Set(dimen=2)
    m.delta_pg = pe.Var(m.delta_pg_set)
    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        p_min = gen['p_min']
        p_max = gen['p_max']
        curve = gen['p_cost']
        cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                curve_type='cost_curve',
                                                                p_min=p_min,
                                                                p_max=p_max,
                                                                gen_name=gen_name)
        for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
            m.delta_pg_set.add((gen_name, ndx))
            m.delta_pg[gen_name, ndx].setlb(0)
            m.delta_pg[gen_name, ndx].setub(o2 - o1)


def declare_var_delta_qg(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model

    m.delta_qg_set = pe.Set(dimen=2)
    m.delta_qg = pe.Var(m.delta_qg_set)
    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        q_min = gen['q_min']
        q_max = gen['q_max']
        curve = gen['q_cost']
        cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                curve_type='cost_curve',
                                                                p_min=q_min,
                                                                p_max=q_max,
                                                                gen_name=gen_name)
        for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
            m.delta_qg_set.add((gen_name, ndx))
            m.delta_qg[gen_name, ndx].setlb(0)
            m.delta_qg[gen_name, ndx].setub(o2 - o1)


def declare_pg_delta_pg_con(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model

    m.pg_delta_pg_con = pe.Constraint(index_set)
    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        p_min = gen['p_min']
        p_max = gen['p_max']
        curve = gen['p_cost']
        cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                curve_type='cost_curve',
                                                                p_min=p_min,
                                                                p_max=p_max,
                                                                gen_name=gen_name)
        lin_coefs = []
        lin_vars = []
        for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
            lin_coefs.append(1)
            lin_vars.append(m.delta_pg[gen_name, ndx])
        expr = LinearExpression(constant=cleaned_values[0][0], linear_coefs=lin_coefs, linear_vars=lin_vars)
        m.pg_delta_pg_con[gen_name] = m.pg[gen_name] == expr * m.gen_in_service_expr[gen_name]


def declare_qg_delta_qg_con(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model

    m.qg_delta_qg_con = pe.Constraint(index_set)
    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        q_min = gen['q_min']
        q_max = gen['q_max']
        curve = gen['q_cost']
        cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                curve_type='cost_curve',
                                                                p_min=q_min,
                                                                p_max=q_max,
                                                                gen_name=gen_name)
        lin_coefs = []
        lin_vars = []
        for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
            lin_coefs.append(1)
            lin_vars.append(m.delta_qg[gen_name, ndx])
        expr = LinearExpression(constant=cleaned_values[0][0], linear_coefs=lin_coefs, linear_vars=lin_vars)
        m.qg_delta_qg_con[gen_name] = m.qg[gen_name] == expr * m.gen_in_service_expr[gen_name]


def declare_var_pg_cost(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model
    m.pg_cost = pe.Var(index_set)


def declare_var_qg_cost(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model
    m.qg_cost = pe.Var(index_set)


def _pw_cost_helper(
        md: ModelData,
        p_or_q: str,
        cost_var: _GeneralVarData,
        gen_var: _GeneralVarData,
        pw_cost_set: _SetData,
        gen_name: str,
        indexed_pw_cost_con: IndexedConstraint,
        in_service_expr: Union[float, int, NumericValue],
):
    gen = md.data['elements']['generator'][gen_name]
    cleaned_values = tx_utils.validate_and_clean_cost_curve(gen[f'{p_or_q}_cost'],
                                                            curve_type='cost_curve',
                                                            p_min=gen[f'{p_or_q}_min'],
                                                            p_max=gen[f'{p_or_q}_max'],
                                                            gen_name=gen_name)
    if len(cleaned_values) > 1:
        for ndx, ((pt1, cost1), (pt2, cost2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
            slope = (cost2 - cost1) / (pt2 - pt1)
            intercept = cost2 - slope * pt2
            pw_cost_set.add((gen_name, ndx))
            indexed_pw_cost_con[gen_name, ndx] = (None, (slope * gen_var + intercept) * in_service_expr - cost_var * in_service_expr, 0)
    else:
        intercept = cleaned_values[0][1]
        pw_cost_set.add((gen_name, 0))
        indexed_pw_cost_con[gen_name, 0] = cost_var == intercept * in_service_expr


def declare_piecewise_pg_cost_cons(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model

    m.pg_piecewise_cost_set = pe.Set(dimen=2)
    m.pg_piecewise_cost_cons = IndexedConstraint(m.pg_piecewise_cost_set)

    for gen_name in index_set:
        _pw_cost_helper(md=md,
                        p_or_q='p',
                        cost_var=m.pg_cost[gen_name],
                        gen_var=m.pg[gen_name],
                        pw_cost_set=m.pg_piecewise_cost_set,
                        gen_name=gen_name,
                        indexed_pw_cost_con=m.pg_piecewise_cost_cons,
                        in_service_expr=m.gen_in_service_expr[gen_name])


def declare_piecewise_qg_cost_cons(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData
):
    m = model

    m.qg_piecewise_cost_set = pe.Set(dimen=2)
    m.qg_piecewise_cost_cons = IndexedConstraint(m.qg_piecewise_cost_set)

    for gen_name in index_set:
        _pw_cost_helper(md=md,
                        p_or_q='q',
                        cost_var=m.qg_cost[gen_name],
                        gen_var=m.qg[gen_name],
                        pw_cost_set=m.qg_piecewise_cost_set,
                        gen_name=gen_name,
                        indexed_pw_cost_con=m.qg_piecewise_cost_cons,
                        in_service_expr=m.gen_in_service_expr[gen_name])


def declare_expression_pg_operating_cost(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        pw_formulation: str = 'delta'
):
    """
    Create the Expression objects to represent the operating costs
    for the real power of each of the generators.
    """
    m = model
    m.pg_operating_cost = pe.Expression(index_set)

    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        if 'p_cost' in gen:
            if gen['p_cost']['cost_curve_type'] == 'polynomial':
                m.pg_operating_cost[gen_name] = sum(v*m.gen_in_service_expr[gen_name]*m.pg[gen_name]**i for i, v in gen['p_cost']['values'].items())
            elif gen['p_cost']['cost_curve_type'] == 'piecewise':
                if pw_formulation == 'delta':
                    p_min = gen['p_min']
                    p_max = gen['p_max']
                    curve = gen['p_cost']
                    cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                            curve_type='cost_curve',
                                                                            p_min=p_min,
                                                                            p_max=p_max,
                                                                            gen_name=gen_name)
                    expr = cleaned_values[0][1]
                    if len(cleaned_values) > 1:
                        for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
                            slope = (c2 - c1) / (o2 - o1)
                            expr += slope * m.delta_pg[gen_name, ndx]
                    m.pg_operating_cost[gen_name] = expr * m.gen_in_service_expr[gen_name]
                else:
                    m.pg_operating_cost[gen_name] = m.pg_cost[gen_name] * m.gen_in_service_expr[gen_name]
            else:
                raise ValueError(f"Unrecognized cost_curve_type: {gen['p_cost']['cost_curve_type']}")
        else:
            m.pg_operating_cost[gen_name] = 0


def declare_expression_qg_operating_cost(
        model: _BlockData,
        md: ModelData,
        index_set: _SetData,
        pw_formulation: str = 'delta'
):
    """
    Create the Expression objects to represent the operating costs
    for the reactive power of each of the generators.
    """
    m = model
    m.qg_operating_cost = pe.Expression(index_set)

    for gen_name in index_set:
        gen = md.data['elements']['generator'][gen_name]
        if 'q_cost' in gen:
            if gen['q_cost']['cost_curve_type'] == 'polynomial':
                m.qg_operating_cost[gen_name] = sum(v*m.gen_in_service_expr[gen_name]*m.qg[gen_name]**i for i, v in gen['q_cost']['values'].items())
            elif gen['q_cost']['cost_curve_type'] == 'piecewise':
                if pw_formulation == 'delta':
                    q_min = gen['q_min']
                    q_max = gen['q_max']
                    curve = gen['q_cost']
                    cleaned_values = tx_utils.validate_and_clean_cost_curve(curve=curve,
                                                                            curve_type='cost_curve',
                                                                            p_min=q_min,
                                                                            p_max=q_max,
                                                                            gen_name=gen_name)
                    expr = cleaned_values[0][1]
                    for ndx, ((o1, c1), (o2, c2)) in enumerate(zip(cleaned_values, cleaned_values[1:])):
                        slope = (c2 - c1) / (o2 - o1)
                        expr += slope * m.delta_pg[gen_name, ndx]
                    m.qg_operating_cost[gen_name] = expr * m.gen_in_service_expr[gen_name]
                else:
                    m.qg_operating_cost[gen_name] = m.qg_cost[gen_name] * m.gen_in_service_expr[gen_name]
            else:
                raise ValueError(f"Unrecognized cost_curve_type: {gen['q_cost']['cost_curve_type']}")
        else:
            m.qg_operating_cost[gen_name] = 0
