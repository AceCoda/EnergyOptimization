from pathlib import Path

import pandas as pd
import pyomo.environ as am

# =========================================================
# 1. DATA AND CONFIGURATION
# =========================================================

YEARS = list(range(1, 11))
SEASONS = ["S1", "S2"]
LOAD_ZONES = ["Day", "Night"]

DAYS_PER_SEASON = {"S1": 219, "S2": 146}
DISCOUNT_RATE = 0.07
DEMAND_GROWTH = 0.001

# Carbon policy
ELEC_EMISSION_FACTOR = 0.570 / 1000.0
GAS_EMISSION_FACTOR = 0.220 / 1000.0
CARBON_TAX_PER_TON = 0
TOTAL_EMISSION_CAP = 5_505_555.0

# Toggle for the 20% efficiency reduction feature
ENABLE_EFFICIENCY_TECH = False

# Grid data
GRID_CARRIERS = ["Elec", "Gas"]
SUB_COST = {"Elec": 2000, "Gas": 3000}
GRID_PRICE = {"Elec": 0.09, "Gas": 0.03}

# Technology data
TECHS = ["PV", "CHP", "Boiler", "SplitAC", "EWH"]
CAP_COST = {
    "PV": 1000,
    "CHP": 1500,
    "Boiler": 300,
    "SplitAC": 500,
    "EWH": 120,
}
MAINT_COST = {
    "PV": 15,
    "CHP": 150,
    "Boiler": 10,
    "SplitAC": 20,
    "EWH": 5,
}
AF = {
    "PV": 0.95,
    "CHP": 0.8,
    "Boiler": 0.95,
    "SplitAC": 1.0,
    "EWH": 0.95,
}
LIFESPAN = {
    "PV": 25,
    "CHP": 20,
    "Boiler": 20,
    "SplitAC": 15,
    "EWH": 12,
}
MIN_CAP = {"PV": 0, "CHP": 100, "Boiler": 10, "SplitAC": 20, "EWH": 10}
MAX_CAP = {"PV": 100, "CHP": 700, "Boiler": 300, "SplitAC": 1000, "EWH": 3000}

# Efficiencies
BOILER_EFF = 0.90
CHP_ELEC_EFF = 0.35
CHP_HEAT_EFF = 0.50
SPLITAC_COP = 3.2
EWH_EFF = 0.97

# CHP polygon region
PT_A = (100, 720)
PT_B = (100, 200)
PT_C = (300, 560)
PT_D = (300, 100)

M_TOP = (PT_C[1] - PT_A[1]) / (PT_C[0] - PT_A[0])
B_TOP = PT_A[1] - M_TOP * PT_A[0]
M_BOT = (PT_D[1] - PT_B[1]) / (PT_D[0] - PT_B[0])
B_BOT = PT_B[1] - M_BOT * PT_B[0]

# Demand data: (Electricity, Heat, Cooling, SolarCF)
DEMAND_DATA = {
    ("S1", "Day"): (180, 20, 160, 0.26),
    ("S1", "Night"): (120, 0, 60, 0),
    ("S2", "Day"): (180, 200, 0, 0.12),
    ("S2", "Night"): (120, 140, 0, 0),
}

BIG_M = 1e6
EPS = 0
OUTPUT_FILE = Path("energy_system_results_FULL.xlsx")


# =========================================================
# 2. MODEL CONSTRUCTION
# =========================================================

def build_model():
    model = am.ConcreteModel()

    model.T = am.Set(initialize=YEARS)
    model.W = am.Set(initialize=YEARS)
    model.S = am.Set(initialize=SEASONS)
    model.L = am.Set(initialize=LOAD_ZONES)
    model.Tech = am.Set(initialize=TECHS)
    model.Carrier = am.Set(initialize=GRID_CARRIERS)

    model.u_grid = am.Var(model.Carrier, domain=am.Binary)
    model.u_tech = am.Var(model.Tech, model.W, domain=am.Binary)
    model.y_tech = am.Var(model.Tech, model.W, domain=am.NonNegativeReals)
    model.x_tech = am.Var(model.Tech, model.T, model.S, model.L, domain=am.NonNegativeReals)
    model.p_chp = am.Var(model.T, model.S, model.L, domain=am.NonNegativeReals)
    model.q_chp = am.Var(model.T, model.S, model.L, domain=am.NonNegativeReals)
    model.u_chp_on = am.Var(model.T, model.S, model.L, domain=am.Binary)
    model.purchased = am.Var(model.Carrier, model.T, model.S, model.L, domain=am.NonNegativeReals)
    model.u_used = am.Var(model.Tech, model.T, domain=am.Binary)
    model.chp_gas = am.Var(model.T, model.S, model.L, domain=am.NonNegativeReals)
    model.active_cap_used = am.Var(model.Tech, model.T, domain=am.NonNegativeReals)

    model.obj = am.Objective(rule=objective_rule, sense=am.minimize)

    add_single_install_constraints(model)
    add_capacity_bounds(model)
    add_activity_limit_constraints(model)
    add_operation_if_installed_constraints(model)
    add_usage_activation_constraints(model)
    add_active_cap_constraints(model)
    add_electricity_balance_constraints(model)
    add_heat_balance_constraints(model)
    add_cooling_balance_constraints(model)
    add_gas_balance_constraints(model)
    add_chp_fuel_constraints(model)
    add_chp_constraints(model)
    add_grid_access_constraints(model)
    add_grid_subscription_constraints(model)
    add_emission_limit_constraint(model)

    return model


def objective_rule(m):
    subscription_cost = sum(m.u_grid[c] * SUB_COST[c] for c in m.Carrier)

    capital_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** w)
        * sum(m.y_tech[tech, w] * CAP_COST[tech] for tech in m.Tech)
        for w in m.W
    )

    maintenance_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** t)
        * sum(MAINT_COST[tech] * m.active_cap_used[tech, t] for tech in m.Tech)
        for t in m.T
    )

    fuel_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** t)
        * sum(
            DAYS_PER_SEASON[s]
            * sum(m.purchased[c, t, s, l] * GRID_PRICE[c] for c in m.Carrier for l in m.L)
            for s in m.S
        )
        for t in m.T
    )

    carbon_cost = sum(
        DAYS_PER_SEASON[s]
        * (
            m.purchased["Elec", t, s, l] * ELEC_EMISSION_FACTOR
            + m.purchased["Gas", t, s, l] * GAS_EMISSION_FACTOR
        )
        * CARBON_TAX_PER_TON
        for t in m.T for s in m.S for l in m.L
    )

    return subscription_cost + capital_cost + maintenance_cost + fuel_cost + carbon_cost


def add_single_install_constraints(model):
    def rule(m, tech):
        return sum(m.u_tech[tech, w] for w in m.W) <= 1

    model.c_single_install = am.Constraint(model.Tech, rule=rule)


def add_capacity_bounds(model):
    model.c_cap_bounds = am.ConstraintList()
    for tech in TECHS:
        for w in YEARS:
            model.c_cap_bounds.add(model.y_tech[tech, w] >= MIN_CAP[tech] * model.u_tech[tech, w])
            model.c_cap_bounds.add(model.y_tech[tech, w] <= MAX_CAP[tech] * model.u_tech[tech, w])


def active_capacity(model, tech, t):
    return sum(model.y_tech[tech, w] for w in YEARS if w <= t and t < w + LIFESPAN[tech])


def add_activity_limit_constraints(model):
    def rule(m, tech, t, s, l):
        active_cap = active_capacity(m, tech, t)
        if tech == "CHP":
            return am.Constraint.Skip
        if tech == "PV":
            return m.x_tech[tech, t, s, l] <= active_cap * AF[tech] * DEMAND_DATA[s, l][3]
        return m.x_tech[tech, t, s, l] <= active_cap * AF[tech]

    model.c_activity_limit = am.Constraint(model.Tech, model.T, model.S, model.L, rule=rule)


def add_operation_if_installed_constraints(model):
    model.c_operation_if_installed = am.ConstraintList()
    for tech in TECHS:
        if tech == "CHP":
            continue
        for t in YEARS:
            active_cap = active_capacity(model, tech, t)
            for s in SEASONS:
                for l in LOAD_ZONES:
                    model.c_operation_if_installed.add(model.x_tech[tech, t, s, l] <= active_cap)


def add_usage_activation_constraints(model):
    model.c_usage_activation = am.ConstraintList()
    for tech in TECHS:
        for t in YEARS:
            if tech == "CHP":
                total_use = sum(model.p_chp[t, s, l] + model.q_chp[t, s, l] for s in SEASONS for l in LOAD_ZONES)
            else:
                total_use = sum(model.x_tech[tech, t, s, l] for s in SEASONS for l in LOAD_ZONES)

            model.c_usage_activation.add(total_use <= BIG_M * model.u_used[tech, t])
            model.c_usage_activation.add(total_use >= EPS * model.u_used[tech, t])


def add_active_cap_constraints(model):
    model.c_active_cap_used = am.ConstraintList()
    for tech in TECHS:
        for t in YEARS:
            active_cap = active_capacity(model, tech, t)
            model.c_active_cap_used.add(model.active_cap_used[tech, t] <= active_cap)
            model.c_active_cap_used.add(model.active_cap_used[tech, t] <= MAX_CAP[tech] * model.u_used[tech, t])
            model.c_active_cap_used.add(model.active_cap_used[tech, t] >= active_cap - MAX_CAP[tech] * (1 - model.u_used[tech, t]))


def add_electricity_balance_constraints(model):
    def rule(m, t, s, l):
        supply = m.x_tech["PV", t, s, l] + m.p_chp[t, s, l] + m.purchased["Elec", t, s, l]
        demand = DEMAND_DATA[s, l][0] + m.x_tech["SplitAC", t, s, l] / SPLITAC_COP + m.x_tech["EWH", t, s, l] / EWH_EFF
        return supply >= demand

    model.c_elec_balance = am.Constraint(model.T, model.S, model.L, rule=rule)


def add_heat_balance_constraints(model):
    def rule(m, t, s, l):
        growth = (1 + DEMAND_GROWTH) ** (t - 1)
        red_factor = 1.0
        if ENABLE_EFFICIENCY_TECH and t >= 5:
            red_factor = 0.8 ** (t - 4)
        supply = m.q_chp[t, s, l] + m.x_tech["Boiler", t, s, l] + m.x_tech["EWH", t, s, l]
        demand = DEMAND_DATA[s, l][1] * growth * red_factor
        return supply >= demand

    model.c_heat_balance = am.Constraint(model.T, model.S, model.L, rule=rule)


def add_cooling_balance_constraints(model):
    def rule(m, t, s, l):
        growth = (1 + DEMAND_GROWTH) ** (t - 1)
        red_factor = 1.0
        if ENABLE_EFFICIENCY_TECH and t >= 5:
            red_factor = 0.8 ** (t - 4)
        return m.x_tech["SplitAC", t, s, l] >= DEMAND_DATA[s, l][2] * growth * red_factor

    model.c_cool_balance = am.Constraint(model.T, model.S, model.L, rule=rule)


def add_gas_balance_constraints(model):
    def rule(m, t, s, l):
        boiler_gas = m.x_tech["Boiler", t, s, l] / BOILER_EFF
        return m.purchased["Gas", t, s, l] == boiler_gas + m.chp_gas[t, s, l]

    model.c_gas_balance = am.Constraint(model.T, model.S, model.L, rule=rule)


def add_chp_fuel_constraints(model):
    def rule(m, t, s, l):
        return m.chp_gas[t, s, l] == m.p_chp[t, s, l] / CHP_ELEC_EFF + m.q_chp[t, s, l] / CHP_HEAT_EFF

    model.c_chp_fuel = am.Constraint(model.T, model.S, model.L, rule=rule)


def add_chp_constraints(model):
    model.c_chp = am.ConstraintList()
    for t in YEARS:
        for s in SEASONS:
            for l in LOAD_ZONES:
                active_chp = active_capacity(model, "CHP", t)
                model.c_chp.add(model.p_chp[t, s, l] <= BIG_M * model.u_chp_on[t, s, l])
                model.c_chp.add(model.q_chp[t, s, l] <= BIG_M * model.u_chp_on[t, s, l])
                model.c_chp.add(active_chp >= EPS * model.u_chp_on[t, s, l])
                model.c_chp.add(model.p_chp[t, s, l] >= 100 * model.u_chp_on[t, s, l])
                model.c_chp.add(model.p_chp[t, s, l] <= 300 * model.u_chp_on[t, s, l])
                model.c_chp.add(model.p_chp[t, s, l] + model.q_chp[t, s, l] <= active_chp * AF["CHP"])
                model.c_chp.add(model.q_chp[t, s, l] <= M_TOP * model.p_chp[t, s, l] + B_TOP)
                model.c_chp.add(model.q_chp[t, s, l] >= M_BOT * model.p_chp[t, s, l] + B_BOT - BIG_M * (1 - model.u_chp_on[t, s, l]))


def add_grid_access_constraints(model):
    def rule(m, c, t, s, l):
        return m.purchased[c, t, s, l] <= BIG_M * m.u_grid[c]

    model.c_grid_access = am.Constraint(model.Carrier, model.T, model.S, model.L, rule=rule)


def add_grid_subscription_constraints(model):
    model.c_grid_use = am.ConstraintList()
    for c in GRID_CARRIERS:
        total_purchase = sum(model.purchased[c, t, s, l] for t in YEARS for s in SEASONS for l in LOAD_ZONES)
        model.c_grid_use.add(total_purchase <= BIG_M * model.u_grid[c])
        model.c_grid_use.add(total_purchase >= EPS * model.u_grid[c])


def add_emission_limit_constraint(model):
    def rule(m):
        total_emissions = sum(
            DAYS_PER_SEASON[s]
            * (
                m.purchased["Elec", t, s, l] * ELEC_EMISSION_FACTOR
                + m.purchased["Gas", t, s, l] * GAS_EMISSION_FACTOR
            )
            for t in m.T for s in m.S for l in m.L
        )
        return total_emissions <= TOTAL_EMISSION_CAP

    model.c_emission_limit = am.Constraint(rule=rule)


# =========================================================
# 3. SOLVE AND REPORT
# =========================================================

def solve_model(model):
    solver = am.SolverFactory("highs")
    return solver.solve(model, tee=True)


def _zero_small_values(frame):
    return frame.map(lambda value: 0 if isinstance(value, (int, float)) and abs(value) < 1e-6 else value)


def summarize_results(model):
    subscription_cost = sum(am.value(model.u_grid[c]) * SUB_COST[c] for c in GRID_CARRIERS)

    capital_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** w)
        * sum(am.value(model.y_tech[tech, w]) * CAP_COST[tech] for tech in TECHS)
        for w in YEARS
    )

    maintenance_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** t)
        * sum(MAINT_COST[tech] * am.value(model.active_cap_used[tech, t]) for tech in TECHS)
        for t in YEARS
    )

    fuel_cost = sum(
        (1 / (1 + DISCOUNT_RATE) ** t)
        * sum(
            DAYS_PER_SEASON[s]
            * sum(am.value(model.purchased[c, t, s, l]) * GRID_PRICE[c] for c in GRID_CARRIERS for l in LOAD_ZONES)
            for s in SEASONS
        )
        for t in YEARS
    )

    carbon_cost = sum(
        DAYS_PER_SEASON[s]
        * (
            am.value(model.purchased["Elec", t, s, l]) * ELEC_EMISSION_FACTOR
            + am.value(model.purchased["Gas", t, s, l]) * GAS_EMISSION_FACTOR
        )
        * CARBON_TAX_PER_TON
        for t in YEARS for s in SEASONS for l in LOAD_ZONES
    )

    total_emissions = sum(
        DAYS_PER_SEASON[s]
        * (
            am.value(model.purchased["Elec", t, s, l]) * ELEC_EMISSION_FACTOR
            + am.value(model.purchased["Gas", t, s, l]) * GAS_EMISSION_FACTOR
        )
        for t in YEARS for s in SEASONS for l in LOAD_ZONES
    )

    objective_breakdown_df = pd.DataFrame([{
        "Subscription Cost": subscription_cost,
        "Capital Cost": capital_cost,
        "O&M Cost": maintenance_cost,
        "Fuel Cost": fuel_cost,
        "Carbon Tax Cost": carbon_cost,
        "Total Objective": am.value(model.obj),
        "Total Emissions (tons)": total_emissions,
    }])

    capacity_rows = []
    for tech in TECHS:
        for w in YEARS:
            val = am.value(model.y_tech[tech, w])
            if val is not None and val > 1e-6:
                capacity_rows.append({
                    "Technology": tech,
                    "Install Year": w,
                    "Installed Capacity": val,
                    "Lifetime": LIFESPAN[tech],
                })
    capacity_df = pd.DataFrame(capacity_rows)

    om_rows = []
    for t in YEARS:
        for tech in TECHS:
            active_cap = active_capacity(model, tech, t)
            used_cap = am.value(model.active_cap_used[tech, t])
            om_paid = MAINT_COST[tech] * used_cap
            discounted_om = om_paid * (1 / (1 + DISCOUNT_RATE) ** t)
            om_rows.append({
                "Year": t,
                "Technology": tech,
                "Installed Active Capacity": active_cap,
                "Used Capacity Paying O&M": used_cap,
                "O&M Cost": om_paid,
                "Discounted O&M Cost": discounted_om,
            })
    om_df = pd.DataFrame(om_rows)

    operation_rows = []
    for t in YEARS:
        growth = (1 + DEMAND_GROWTH) ** (t - 1)
        red_factor = 1.0
        if ENABLE_EFFICIENCY_TECH and t >= 5:
            red_factor = 0.8 ** (t - 4)

        for s in SEASONS:
            for l in LOAD_ZONES:
                elec_demand = DEMAND_DATA[s, l][0]
                heat_demand = DEMAND_DATA[s, l][1] * growth * red_factor
                cool_demand = DEMAND_DATA[s, l][2] * growth * red_factor

                pv_output = am.value(model.x_tech["PV", t, s, l])
                chp_elec = am.value(model.p_chp[t, s, l])
                chp_heat = am.value(model.q_chp[t, s, l])
                chp_gas = am.value(model.chp_gas[t, s, l])
                boiler_heat = am.value(model.x_tech["Boiler", t, s, l])
                boiler_gas = boiler_heat / BOILER_EFF
                split_cooling = am.value(model.x_tech["SplitAC", t, s, l])
                split_elec = split_cooling / SPLITAC_COP
                ewh_heat = am.value(model.x_tech["EWH", t, s, l])
                ewh_elec = ewh_heat / EWH_EFF
                elec_purchase = am.value(model.purchased["Elec", t, s, l])
                gas_purchase = am.value(model.purchased["Gas", t, s, l])

                operation_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Electricity Demand": elec_demand,
                    "Heat Demand": heat_demand,
                    "Cooling Demand": cool_demand,
                    "PV Electricity Output": pv_output,
                    "CHP Electricity Output": chp_elec,
                    "CHP Heat Output": chp_heat,
                    "CHP Gas Input": chp_gas,
                    "Boiler Heat Output": boiler_heat,
                    "Boiler Gas Input": boiler_gas,
                    "SplitAC Cooling Output": split_cooling,
                    "SplitAC Electricity Input": split_elec,
                    "EWH Heat Output": ewh_heat,
                    "EWH Electricity Input": ewh_elec,
                    "Electricity Purchase": elec_purchase,
                    "Gas Purchase": gas_purchase,
                    "CHP ON": am.value(model.u_chp_on[t, s, l]),
                })
    operations_df = pd.DataFrame(operation_rows)

    op_cost_rows = []
    for t in YEARS:
        discount_factor = 1 / (1 + DISCOUNT_RATE) ** t
        for s in SEASONS:
            days = DAYS_PER_SEASON[s]
            for l in LOAD_ZONES:
                chp_elec = am.value(model.p_chp[t, s, l])
                chp_heat = am.value(model.q_chp[t, s, l])
                chp_gas = am.value(model.chp_gas[t, s, l])
                chp_cost = chp_gas * GRID_PRICE["Gas"] * days * discount_factor
                op_cost_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Technology": "CHP",
                    "Electricity Output": chp_elec,
                    "Heat Output": chp_heat,
                    "Cooling Output": 0,
                    "Electricity Input": 0,
                    "Gas Input": chp_gas,
                    "Operating Cost": chp_cost,
                })

                boiler_heat = am.value(model.x_tech["Boiler", t, s, l])
                boiler_gas = boiler_heat / BOILER_EFF
                boiler_cost = boiler_gas * GRID_PRICE["Gas"] * days * discount_factor
                op_cost_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Technology": "Boiler",
                    "Electricity Output": 0,
                    "Heat Output": boiler_heat,
                    "Cooling Output": 0,
                    "Electricity Input": 0,
                    "Gas Input": boiler_gas,
                    "Operating Cost": boiler_cost,
                })

                split_cooling = am.value(model.x_tech["SplitAC", t, s, l])
                split_elec = split_cooling / SPLITAC_COP
                split_cost = split_elec * GRID_PRICE["Elec"] * days * discount_factor
                op_cost_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Technology": "SplitAC",
                    "Electricity Output": 0,
                    "Heat Output": 0,
                    "Cooling Output": split_cooling,
                    "Electricity Input": split_elec,
                    "Gas Input": 0,
                    "Operating Cost": split_cost,
                })

                ewh_heat = am.value(model.x_tech["EWH", t, s, l])
                ewh_elec = ewh_heat / EWH_EFF
                ewh_cost = ewh_elec * GRID_PRICE["Elec"] * days * discount_factor
                op_cost_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Technology": "EWH",
                    "Electricity Output": 0,
                    "Heat Output": ewh_heat,
                    "Cooling Output": 0,
                    "Electricity Input": 0,
                    "Gas Input": 0,
                    "Operating Cost": ewh_cost,
                })

                pv_output = am.value(model.x_tech["PV", t, s, l])
                op_cost_rows.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Technology": "PV",
                    "Electricity Output": pv_output,
                    "Heat Output": 0,
                    "Cooling Output": 0,
                    "Electricity Input": 0,
                    "Gas Input": 0,
                    "Operating Cost": 0,
                })
    op_cost_df = pd.DataFrame(op_cost_rows)

    grid_rows = []
    for c in GRID_CARRIERS:
        total_purchase = sum(am.value(model.purchased[c, t, s, l]) for t in YEARS for s in SEASONS for l in LOAD_ZONES)
        total_cost = sum(
            (1 / (1 + DISCOUNT_RATE) ** t)
            * DAYS_PER_SEASON[s]
            * am.value(model.purchased[c, t, s, l])
            * GRID_PRICE[c]
            for t in YEARS for s in SEASONS for l in LOAD_ZONES
        )
        grid_rows.append({
            "Carrier": c,
            "Subscribed": am.value(model.u_grid[c]),
            "Subscription Cost": am.value(model.u_grid[c]) * SUB_COST[c],
            "Total Purchased": total_purchase,
            "Total Purchase Cost": total_cost,
        })
    grid_df = pd.DataFrame(grid_rows)

    emissions_data = []
    for t in YEARS:
        for s in SEASONS:
            for l in LOAD_ZONES:
                elec_purchased = am.value(model.purchased["Elec", t, s, l])
                gas_purchased = am.value(model.purchased["Gas", t, s, l])
                elec_emissions_per_day = elec_purchased * ELEC_EMISSION_FACTOR
                gas_emissions_per_day = gas_purchased * GAS_EMISSION_FACTOR
                total_period_emissions = (elec_emissions_per_day + gas_emissions_per_day) * DAYS_PER_SEASON[s]
                emissions_data.append({
                    "Year": t,
                    "Season": s,
                    "Load Zone": l,
                    "Electricity Purchased (kWh)": elec_purchased,
                    "Gas Purchased (kWh)": gas_purchased,
                    "Electricity Emissions (tons CO2/day)": elec_emissions_per_day,
                    "Gas Emissions (tons CO2/day)": gas_emissions_per_day,
                    "Total Period Emissions (tons CO2)": total_period_emissions,
                })
    detailed_emissions_df = pd.DataFrame(emissions_data)

    return {
        "objective_breakdown": _zero_small_values(objective_breakdown_df),
        "capacity": _zero_small_values(capacity_df),
        "om": _zero_small_values(om_df),
        "operations": _zero_small_values(operations_df),
        "op_cost": _zero_small_values(op_cost_df),
        "grid": _zero_small_values(grid_df),
        "emissions": _zero_small_values(detailed_emissions_df),
    }


def export_results(results):
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        results["objective_breakdown"].to_excel(writer, sheet_name="Objective Breakdown", index=False)
        results["capacity"].to_excel(writer, sheet_name="Installed Capacity", index=False)
        results["om"].to_excel(writer, sheet_name="Yearly O&M Costs", index=False)
        results["operations"].to_excel(writer, sheet_name="Detailed Operations", index=False)
        results["op_cost"].to_excel(writer, sheet_name="Operational Costs", index=False)
        results["grid"].to_excel(writer, sheet_name="Grid Subscription", index=False)
        results["emissions"].to_excel(writer, sheet_name="Detailed Emissions", index=False)


def main():
    model = build_model()
    solve_model(model)
    results = summarize_results(model)
    export_results(results)

    print("\n========================================")
    print("OPTIMIZATION COMPLETED")
    print("========================================")
    print(f"\nObjective Value = {am.value(model.obj):,.2f}")
    print(f"\nResults exported to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
