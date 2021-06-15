import coramin
from egret.models.ac_relaxations import create_atan_relaxation
import pyomo.environ as pe
from galini.galini import Galini
from galini.branch_and_bound.algorithm import BranchAndBoundAlgorithm
from pyomo.common.config import ConfigDict, ConfigValue, NonNegativeFloat, PositiveFloat, NonNegativeInt
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from egret.data.model_data import ModelData
import itertools
from galini.branch_and_bound.node import NodeSolution
from pyomo.contrib.fbbt.fbbt import fbbt
from galini.branch_and_bound.strategy import BranchingStrategy
from galini.branch_and_bound.branching import BranchingPoint
from pyomo.core.expr.visitor import identify_variables
import math
import logging
from galini.branch_and_cut.branching import compute_branching_decision
from galini.solvers.solution import load_solution_from_model
from galini.branch_and_bound.selection import BestLowerBoundSelectionStrategy
from galini.branch_and_cut.node_storage import _NodeStorageBase
from galini.pyomo import safe_setlb, safe_setub
from galini.branch_and_bound.branching import branch_at_point
from mpi4py import MPI
from pyomo.common.collections.component_set import ComponentSet
import warnings


comm: MPI.Comm = MPI.COMM_WORLD
rank = comm.Get_rank()
logger = logging.getLogger('gacopf')


class GlobalACOPFConfig(ConfigDict):
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super(GlobalACOPFConfig, self).__init__(description=description,
                                                doc=doc,
                                                implicit=implicit,
                                                implicit_domain=implicit_domain,
                                                visibility=visibility)

        self.timelimit = self.declare('timelimit', ConfigValue(default=3600, domain=NonNegativeFloat))
        self.lp_solver = self.declare('lp_solver', ConfigValue())
        self.nlp_solver = self.declare('nlp_solver', ConfigValue())
        self.absolute_gap = self.declare('absolute_gap', ConfigValue(default=1e-4, domain=PositiveFloat))
        self.relative_gap = self.declare('relative_gap', ConfigValue(default=1e-3, domain=PositiveFloat))
        self.obbt_solver = self.declare('obbt_solver', ConfigValue())
        self.log_level = self.declare('log_level', ConfigValue(default=25, domain=NonNegativeInt))


def _get_galini(config: GlobalACOPFConfig):
    galini = _ACOPFGalini()
    galini.get_configuration_group('galini').set('timelimit', config.timelimit)
    galini.get_configuration_group('branch_and_cut')['bab'].set('absolute_gap', config.absolute_gap)
    galini.get_configuration_group('branch_and_cut')['bab'].set('relative_gap', config.relative_gap)
    if rank == 0:
        galini.get_configuration_group('logging').set('level', config.log_level)
    else:
        galini.get_configuration_group('logging').set('level', logging.WARNING)
        galini.get_configuration_group('logging').set('stdout', False)
    galini._log_manager.apply_config(galini.get_configuration_group('logging'))
    galini.acopf_config = config
    return galini


class _ACOPFGalini(Galini):
    def get_algorithm(self, name):
        return _ACOPFBranchAndBound


class ACOPFBranchingStrategy(BranchingStrategy):
    def branch(self, node, tree):
        m: coramin.domain_reduction.dbt.TreeBlockData = node.storage.model()
        bounds = node.storage.model_bounds
        coupling_vars_to_branch_on = list()
        for stage in range(m.num_stages()):
            for block in m.stage_blocks(stage, active=True):
                if not block.is_leaf():
                    for con in block.linking_constraints.values():
                        v = next(identify_variables(con.body))
                        lb, ub = bounds[v]
                        if lb is None:
                            lb = -math.inf
                        if ub is None:
                            ub = math.inf
                        if ub - lb > 0.01:
                            weight = (ub - lb) / (stage + 1)**3
                            coupling_vars_to_branch_on.append((v, weight))
        if len(coupling_vars_to_branch_on) > 0:
            coupling_vars_to_branch_on.sort(key=lambda x: x[1])
            var_to_branch_on, weight = coupling_vars_to_branch_on[-1]
            lb, ub = bounds[var_to_branch_on]
            if lb is None or ub is None:
                logger.warning(f'branching on unbounded variable: {str(var_to_branch_on)}; LB: {lb}; UB: {ub}')
            if lb is None and ub is None:
                bp = BranchingPoint(variable=var_to_branch_on, points=[-1000, 1000])
            elif lb is None:
                bp = BranchingPoint(variable=var_to_branch_on, points=[ub - 1000])
            elif ub is None:
                bp = BranchingPoint(variable=var_to_branch_on, points=[lb + 1000])
            else:
                bp = BranchingPoint(variable=var_to_branch_on, points=[0.5*(ub + lb)])
        else:
            bp = BranchingPoint(variable=node.storage.branching_decision.variable,
                                points=[node.storage.branching_decision.point])
        return bp


class ACOPFNodeStorageBase(_NodeStorageBase):
    def update_bounds(self, bounds):
        for v, bnds in bounds.items():
            self._bounds[v] = bnds

    def branch_at_point(self, branching_point, mc):
        assert self.branching_point is None
        self.branching_point = branching_point
        children_bounds = branch_at_point(self.model(), self._bounds, branching_point, mc)
        return [ACOPFNodeStorage(self.root, self, bounds, branching_point.variable) for bounds in children_bounds]

    @property
    def relaxation_data(self):
        raise NotImplementedError

    def recompute_model_relaxation_bounds(self):
        rel = self.root._linear_model
        for v, (lb, ub) in self.model_bounds.items():
            linear_var = self.root.model_to_relaxation_var_map[v]
            if linear_var in self.root.unfixed_vars:
                linear_var.unfix()
            safe_setlb(linear_var, lb)
            safe_setub(linear_var, ub)
            if not linear_var.is_fixed() and lb is not None and ub is not None and math.isclose(lb, ub, rel_tol=1e-5, abs_tol=1e-6):
                linear_var.fix(lb)

        for b in coramin.relaxations.relaxation_data_objects(rel, active=True, descend_into=True):
            if not self.is_root:
                b.pop_oa_points(key=self.parent)
                b.push_oa_points(key=self.parent)
            b.rebuild()
            b.push_oa_points(key=self)


class ACOPFNodeStorage(ACOPFNodeStorageBase):
    def __init__(self, root, parent, bounds, branching_variable):
        super().__init__(root, parent, bounds)
        self.branching_variable = branching_variable

    @property
    def is_root(self):
        return False

    @property
    def model_to_relaxation_var_map(self):
        return self.root.model_to_relaxation_var_map

    @property
    def relaxation_to_model_var_map(self):
        return self.root.relaxation_to_model_var_map


class ACOPFRootNodeStorage(ACOPFNodeStorageBase):
    def __init__(self, model):
        bounds = pe.ComponentMap((v, v.bounds) for v in coramin.relaxations.nonrelaxation_component_data_objects(model, pe.Var, active=True, descend_into=True))
        super().__init__(root=self, parent=None, bounds=bounds)
        self._model = model
        self._linear_model = self._model.clone()
        self.model_to_relaxation_var_map = pe.ComponentMap()
        self.relaxation_to_model_var_map = pe.ComponentMap()
        self.unfixed_vars = ComponentSet()
        for v in coramin.relaxations.nonrelaxation_component_data_objects(model, pe.Var, active=True, descend_into=True):
            rel_var = self._linear_model.find_component(v)
            assert rel_var is not None
            self.model_to_relaxation_var_map[v] = rel_var
            self.relaxation_to_model_var_map[rel_var] = v
            if not rel_var.is_fixed():
                self.unfixed_vars.add(rel_var)

        for b in coramin.relaxations.relaxation_data_objects(self._model, descend_into=True, active=True):
            b.rebuild(build_nonlinear_constraint=True)

        self._linear_model.galini_nonlinear_relaxations = list()
        for b in coramin.relaxations.relaxation_data_objects(self._linear_model, descend_into=True, active=True):
            self._linear_model.galini_nonlinear_relaxations.append(b)

    @property
    def is_root(self):
        return True


class _ACOPFBranchAndBound(BranchAndBoundAlgorithm):
    name = 'branch_and_cut'

    def __init__(self, galini: _ACOPFGalini):
        super().__init__(galini)
        self._acopf_config: GlobalACOPFConfig = galini.acopf_config
        self._branching_strategy = ACOPFBranchingStrategy()
        self._selection_strategy = BestLowerBoundSelectionStrategy(self)

    @property
    def branching_strategy(self):
        return self._branching_strategy

    @property
    def node_selection_strategy(self):
        return self._selection_strategy

    @property
    def bab_config(self):
        return self.galini.get_configuration_group('branch_and_cut').bab

    def _ub_solve(self, model):
        opt = self._acopf_config.nlp_solver
        res = opt.solve(model, load_solutions=False)
        if pe.check_optimal_termination(res):
            model.solutions.load_from(res)
            feasible_objective = pe.value(coramin.utils.get_objective(model))
            sol = load_solution_from_model(results=res, model=model)
        else:
            feasible_objective = None
            sol = None
        return feasible_objective, sol

    def _lb_solve(self, model):
        opt = self._acopf_config.lp_solver
        opt.set_instance(model)
        res = opt.solve(save_results=False, load_solutions=False)
        if res.solver.termination_condition == pe.TerminationCondition.optimal:
            bound = res.problem.lower_bound
            opt.load_vars()
            sol = load_solution_from_model(results=res, model=model)
            sol.objective = bound
        elif res.solver.termination_condition == pe.TerminationCondition.infeasible:
            bound = math.inf
            sol = None
        else:
            res = opt.solve(tee=True, save_results=False, load_solutions=False)
            raise RuntimeError(f'Lower bounding problem did not converge; termination condition: {res.solver.termination_condition}')
        return bound, sol

    def find_initial_solution(self, model, tree, node):
        ub, sol = self._ub_solve(model)
        return NodeSolution(None, sol)

    def _solve_problem_at_node(self, tree, node, is_root):
        self.logger.debug(f'solving problem at node: {node.coordinate}')
        self.logger.debug('getting nlp')
        nlp = node.storage.model()
        self.logger.debug('starting fbbt')
        new_bounds = fbbt(nlp, max_iter=2, feasibility_tol=1e-6)
        self.logger.debug('updating bounds')
        node.storage.update_bounds(bounds=new_bounds)
        self.logger.debug('solving upper bounding problem')
        ub, nlp_sol = self._ub_solve(nlp)
        self.logger.debug('getting relaxation')
        relaxation = node.storage.model_relaxation()
        self.logger.debug('initial solve of relaxation')
        lb, rel_sol = self._lb_solve(relaxation)
        self.logger.debug(f'node LB: {lb};    node UB: {ub}')
        if lb < tree.upper_bound - self.galini.mc.epsilon:
            self.logger.debug('starting DBT')
            dbt_info = coramin.domain_reduction.perform_dbt(relaxation=relaxation,
                                                            solver=self._acopf_config.obbt_solver,
                                                            time_limit=self.galini.timelimit.seconds_left(),
                                                            objective_bound=tree.upper_bound,
                                                            parallel=True,
                                                            feasibility_tol=1e-8,
                                                            safety_tol=1e-4,
                                                            with_progress_bar=False)
            self.logger.debug(str(dbt_info))
            self.logger.debug('updating bounds')
            new_bounds = pe.ComponentMap()
            for v in coramin.relaxations.nonrelaxation_component_data_objects(relaxation, pe.Var, active=True, descend_into=True):
                nlp_v = node.storage.relaxation_to_model_var_map[v]
                new_bounds[nlp_v] = (v.lb, v.ub)
            node.storage.update_bounds(bounds=new_bounds)
            self.logger.debug('getting updated relaxation')
            relaxation = node.storage.model_relaxation()
            self.logger.debug('resolving relaxation')
            lb, rel_sol = self._lb_solve(relaxation)
            self.logger.debug(f'node LB: {lb};    node UB: {ub}')
        else:
            self.logger.debug('lower bound is large enough; bounds tightening is not needed.')
        if math.isfinite(lb):
            weights = {'sum': self.bab_config['branching_weight_sum'],
                       'max': self.bab_config['branching_weight_max'],
                       'min': self.bab_config['branching_weight_min']}
            self.logger.debug('computing branching decision')
            bd = compute_branching_decision(model=nlp,
                                            linear_model=relaxation,
                                            root_bounds=node.tree.root.storage.model_bounds,
                                            mip_solution=rel_sol,
                                            weights=weights,
                                            lambda_=self.bab_config['branching_weight_lambda'],
                                            mc=self.galini.mc)
        else:
            bd = None
        node.storage.branching_decision = bd
        return NodeSolution(rel_sol, nlp_sol)

    def solve_problem_at_root(self, tree, node):
        return self._solve_problem_at_node(tree, node, True)

    def solve_problem_at_node(self, tree, node):
        return self._solve_problem_at_node(tree, node, False)

    def init_node_storage(self, model):
        return ACOPFRootNodeStorage(model=model)


def _multivariate_oa(m):
    for b in coramin.relaxations.relaxation_data_objects(m, descend_into=True, active=True):
        if isinstance(b, coramin.relaxations.MultivariateRelaxationData):
            b.clear_oa_points()
            for bnd_combination in itertools.product(*[itertools.product(['L', 'U'], [v]) for v in b.get_rhs_vars()]):
                bnd_dict = pe.ComponentMap()
                for lower_or_upper, v in bnd_combination:
                    if lower_or_upper == 'L':
                        if v.has_lb():
                            bnd_dict[v] = v.lb
                        else:
                            bnd_dict[v] = -1
                    else:
                        assert lower_or_upper == 'U'
                        if v.has_ub():
                            bnd_dict[v] = v.ub
                        else:
                            bnd_dict[v] = 1
                b.add_oa_point(var_values=bnd_dict)


def solve_global_acopf(md: ModelData, config: GlobalACOPFConfig):
    if not isinstance(config.lp_solver, PersistentSolver):
        raise ValueError('Only persistent solvers should be used')
    logger.info('creating atan relaxation')
    m, scaled_md = create_atan_relaxation(md, use_linear_relaxation=False)
    logger.info('decomposing')
    m, component_map, termination_reason = coramin.domain_reduction.decompose_model(m, max_leaf_nnz=1000)
    all_nonlinear_vars = ComponentSet()
    arctan_count = 0
    for b in coramin.relaxations.relaxation_data_objects(m, descend_into=True, active=True):
        all_nonlinear_vars.update(b.get_rhs_vars())
        if isinstance(b, coramin.relaxations.PWArctanRelaxationData):
            b.use_linear_relaxation = True
            arctan_count += 1
    logger.info(f'arctan count: {arctan_count}')
    logger.info(f'# nonlinear vars: {len(all_nonlinear_vars)}')
    # _multivariate_oa(m)
    galini = _get_galini(config)
    logger.info('solving')
    sol = galini.solve(m, clone_model=False)
    return sol
