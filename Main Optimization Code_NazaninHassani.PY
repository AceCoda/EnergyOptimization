import pyomo.environ as am
import pandas as pd

# =========================================================
# 1. DATA INITIALIZATION
# =========================================================

years = list(range(1, 11)) 

seasons = ['S1', 'S2']
load_zones = ['Day', 'Night']

days_per_season = {
    'S1': 219,
    'S2': 146
}

discount_rate = 0.07
demand_growth = 0.001

# Carbon Policy Parameters
ELEC_EMISSION_FACTOR = 0.570 / 1000.0  # tons/kWh
GAS_EMISSION_FACTOR = 0.220 / 1000.0   # tons/kWh
CARBON_TAX_PER_TON = 0                 # $/ton (Change to 83 as asked in the HW)
TOTAL_EMISSION_CAP = 5505555.0           # target emission limit (tons)


#  New Parameter for 20% Efficiency Technology
enable_efficiency_tech = False  # Toggle this to True for 20% annual reduction in Heat and Cooling


# =========================================================
# GRID DATA
# =========================================================

grid_carriers = ['Elec', 'Gas']

sub_cost = {
    'Elec': 2000,
    'Gas': 3000
}

grid_price = {
    'Elec': 0.09,
    'Gas': 0.03
}

# =========================================================
# TECHNOLOGY DATA
# =========================================================

techs = ['PV', 'CHP', 'Boiler', 'SplitAC', 'EWH']

cap_cost = {
    'PV': 1000,
    'CHP': 1500,
    'Boiler': 300,
    'SplitAC': 500,
    'EWH': 120
}

maint_cost = {
    'PV': 15,
    'CHP': 150,
    'Boiler': 10,
    'SplitAC': 20,
    'EWH': 5
}

af = {
    'PV': 0.95,
    'CHP': 0.8,
    'Boiler': 0.95,
    'SplitAC': 1.0,
    'EWH': 0.95
}

lifespan = {
    'PV': 25,
    'CHP': 20,
    'Boiler': 20,
    'SplitAC': 15,
    'EWH': 12
}

min_cap = {
    'PV': 0,
    'CHP': 100,
    'Boiler': 10,
    'SplitAC': 20,
    'EWH': 10
}

max_cap = {
    'PV': 100,
    'CHP': 700,
    'Boiler': 300,
    'SplitAC': 1000,
    'EWH': 3000
}

# =========================================================
# EFFICIENCIES
# =========================================================

boiler_eff = 0.90

# FIXED CHP EFFICIENCIES
chp_elec_eff = 0.35
chp_heat_eff = 0.50

splitac_cop = 3.2
ewh_eff = 0.97

# =========================================================
# CHP POLYGON REGION
# =========================================================

pt_a = (100, 720)
pt_b = (100, 200)

pt_c = (300, 560)
pt_d = (300, 100)

m_top = (pt_c[1] - pt_a[1]) / (pt_c[0] - pt_a[0])
b_top = pt_a[1] - m_top * pt_a[0]

m_bot = (pt_d[1] - pt_b[1]) / (pt_d[0] - pt_b[0])
b_bot = pt_b[1] - m_bot * pt_b[0]

# =========================================================
# DEMAND DATA
# =========================================================

# Format:
# (Electricity, Heat, Cooling, SolarCF)

demand_data = {
    ('S1', 'Day'):   (180, 20, 160, 0.26),
    ('S1', 'Night'): (120, 0, 60, 0),
    ('S2', 'Day'):   (180, 200, 0, 0.12),
    ('S2', 'Night'): (120, 140, 0, 0)
}

# =========================================================
# 2. MODEL
# =========================================================

model = am.ConcreteModel()

# =========================================================
# SETS
# =========================================================

model.T = am.Set(initialize=years)
model.W = am.Set(initialize=years)

model.S = am.Set(initialize=seasons)
model.L = am.Set(initialize=load_zones)

model.Tech = am.Set(initialize=techs)
model.Carrier = am.Set(initialize=grid_carriers)

# =========================================================
# VARIABLES
# =========================================================

model.u_grid = am.Var(
    model.Carrier,
    domain=am.Binary
)

model.u_tech = am.Var(
    model.Tech,
    model.W,
    domain=am.Binary
)

model.y_tech = am.Var(
    model.Tech,
    model.W,
    domain=am.NonNegativeReals
)

model.x_tech = am.Var(
    model.Tech,
    model.T,
    model.S,
    model.L,
    domain=am.NonNegativeReals
)

model.p_chp = am.Var(
    model.T,
    model.S,
    model.L,
    domain=am.NonNegativeReals
)

model.q_chp = am.Var(
    model.T,
    model.S,
    model.L,
    domain=am.NonNegativeReals
)

model.u_chp_on = am.Var(
    model.T,
    model.S,
    model.L,
    domain=am.Binary
)

model.purchased = am.Var(
    model.Carrier,
    model.T,
    model.S,
    model.L,
    domain=am.NonNegativeReals
)

model.u_used = am.Var(
    model.Tech,
    model.T,
    domain=am.Binary
)

model.chp_gas = am.Var(
    model.T,
    model.S,
    model.L,
    domain=am.NonNegativeReals
)

# =========================================================
# LINEARIZED O&M VARIABLE
# =========================================================

model.active_cap_used = am.Var(
    model.Tech,
    model.T,
    domain=am.NonNegativeReals
)

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def obj_rule(m):

    subs = sum(
        m.u_grid[c] * sub_cost[c]
        for c in m.Carrier
    )

    cap_exp = sum(

        (1 / (1 + discount_rate) ** w)

        *

        sum(
            m.y_tech[tech, w] * cap_cost[tech]
            for tech in m.Tech
        )

        for w in m.W
    )

    maint = sum(

        (1 / (1 + discount_rate) ** t)

        *

        sum(

            maint_cost[tech]
            *
            m.active_cap_used[tech, t]

            for tech in m.Tech
        )

        for t in m.T
    )

    fuel = sum(

        (1 / (1 + discount_rate) ** t)

        *

        sum(

            days_per_season[s]

            *

            sum(

                m.purchased[c, t, s, l]
                *
                grid_price[c]

                for c in m.Carrier
                for l in m.L
            )

            for s in m.S
        )

        for t in m.T
    )

    # --- Carbon Tax Cost Calculation ---
    carbon_cost = sum(
        days_per_season[s] * (
            m.purchased['Elec', t, s, l] * ELEC_EMISSION_FACTOR +
            m.purchased['Gas', t, s, l] * GAS_EMISSION_FACTOR
        ) * CARBON_TAX_PER_TON
        for t in m.T for s in m.S for l in m.L
    )

    return subs + cap_exp + maint + fuel + carbon_cost


model.obj = am.Objective(
    rule=obj_rule,
    sense=am.minimize
)

# =========================================================
# CONSTANTS
# =========================================================

BIG_M = 1e6
EPS = 0

# =========================================================
# SINGLE INSTALLATION
# =========================================================

def single_install_rule(m, tech):

    return sum(
        m.u_tech[tech, w]
        for w in m.W
    ) <= 1


model.c_single_install = am.Constraint(
    model.Tech,
    rule=single_install_rule
)

# =========================================================
# CAPACITY BOUNDS
# =========================================================

model.c_cap_bounds = am.ConstraintList()

for tech in techs:

    for w in years:

        model.c_cap_bounds.add(
            model.y_tech[tech, w]
            >=
            min_cap[tech]
            * model.u_tech[tech, w]
        )

        model.c_cap_bounds.add(
            model.y_tech[tech, w]
            <=
            max_cap[tech]
            * model.u_tech[tech, w]
        )

# =========================================================
# ACTIVITY LIMITS
# =========================================================

def activity_limit_rule(m, tech, t, s, l):

    active_cap = sum(
        m.y_tech[tech, w]
        for w in m.W
        if w <= t and t < w + lifespan[tech]
    )

    if tech == 'CHP':

        return am.Constraint.Skip

    elif tech == 'PV':

        return (
            m.x_tech[tech, t, s, l]
            <=
            active_cap
            *
            af[tech]
            *
            demand_data[s, l][3]
        )

    else:

        return (
            m.x_tech[tech, t, s, l]
            <=
            active_cap
            *
            af[tech]
        )


model.c_activity_limit = am.Constraint(
    model.Tech,
    model.T,
    model.S,
    model.L,
    rule=activity_limit_rule
)

# =========================================================
# FORCE OPERATION ONLY IF INSTALLED
# =========================================================

model.c_operation_if_installed = am.ConstraintList()

for tech in techs:

    if tech == 'CHP':
        continue

    for t in years:

        active_cap = sum(
            model.y_tech[tech, w]
            for w in years
            if w <= t and t < w + lifespan[tech]
        )

        for s in seasons:

            for l in load_zones:

                model.c_operation_if_installed.add(
                    model.x_tech[tech, t, s, l]
                    <=
                    active_cap
                )

# =========================================================
# TECHNOLOGY USAGE ACTIVATION
# =========================================================

model.c_usage_activation = am.ConstraintList()

for tech in techs:

    for t in years:

        if tech == 'CHP':

            total_use = sum(
                model.p_chp[t, s, l]
                +
                model.q_chp[t, s, l]
                for s in seasons
                for l in load_zones
            )

        else:

            total_use = sum(
                model.x_tech[tech, t, s, l]
                for s in seasons
                for l in load_zones
            )

        model.c_usage_activation.add(
            total_use
            <=
            BIG_M * model.u_used[tech, t]
        )

        model.c_usage_activation.add(
            total_use
            >=
            EPS * model.u_used[tech, t]
        )

# =========================================================
# LINEARIZED O&M CONSTRAINTS
# =========================================================

model.c_active_cap_used = am.ConstraintList()

for tech in techs:

    for t in years:

        active_cap = sum(
            model.y_tech[tech, w]
            for w in years
            if w <= t and t < w + lifespan[tech]
        )

        model.c_active_cap_used.add(
            model.active_cap_used[tech, t]
            <=
            active_cap
        )

        model.c_active_cap_used.add(
            model.active_cap_used[tech, t]
            <=
            max_cap[tech]
            * model.u_used[tech, t]
        )

        model.c_active_cap_used.add(
            model.active_cap_used[tech, t]
            >=
            active_cap
            -
            max_cap[tech]
            *
            (
                1 - model.u_used[tech, t]
            )
        )

# =========================================================
# ELECTRICITY BALANCE
# =========================================================

def elec_balance_rule(m, t, s, l):

    growth = 1

    supply = (

        m.x_tech['PV', t, s, l]

        +

        m.p_chp[t, s, l]

        +

        m.purchased['Elec', t, s, l]
    )

    demand = (

        demand_data[s, l][0]

        +

        (
            m.x_tech['SplitAC', t, s, l]
            / splitac_cop
        )

        +

        (
            m.x_tech['EWH', t, s, l]
            / ewh_eff
        )
    )

    return supply >= demand


model.c_elec_balance = am.Constraint(
    model.T,
    model.S,
    model.L,
    rule=elec_balance_rule
)

# =========================================================
# HEAT BALANCE
# =========================================================

def heat_balance_rule(m, t, s, l):

    growth = (1 + demand_growth) ** (t - 1)

    # --- Efficiency Reduction Factor ---
    red_factor = 1.0
    if enable_efficiency_tech and t >= 5:
        red_factor = 0.8 ** (t - 4)
    # -----------------------------------

    supply = (

        m.q_chp[t, s, l]

        +

        m.x_tech['Boiler', t, s, l]

        +

        m.x_tech['EWH', t, s, l]
    )

    demand = demand_data[s, l][1] * growth * red_factor

    return supply >= demand


model.c_heat_balance = am.Constraint(
    model.T,
    model.S,
    model.L,
    rule=heat_balance_rule
)

# =========================================================
# COOLING BALANCE
# =========================================================

def cool_balance_rule(m, t, s, l):

    growth = (1 + demand_growth) ** (t - 1)

    # --- Efficiency Reduction Factor ---
    red_factor = 1.0
    if enable_efficiency_tech and t >= 5:
        red_factor = 0.8 ** (t - 4)
    # -----------------------------------

    return (
        m.x_tech['SplitAC', t, s, l]
        >=
        demand_data[s, l][2] * growth * red_factor
    )


model.c_cool_balance = am.Constraint(
    model.T,
    model.S,
    model.L,
    rule=cool_balance_rule
)

# =========================================================
# GAS BALANCE
# =========================================================

def gas_balance_rule(m, t, s, l):

    boiler_gas = (
        m.x_tech['Boiler', t, s, l]
        / boiler_eff
    )

    return (

        m.purchased['Gas', t, s, l]

        ==

        boiler_gas

        +

        m.chp_gas[t, s, l]
    )


model.c_gas_balance = am.Constraint(
    model.T,
    model.S,
    model.L,
    rule=gas_balance_rule
)

# =========================================================
# CHP FUEL CONSUMPTION
# =========================================================

def chp_fuel_rule(m, t, s, l):

    return (
        m.chp_gas[t, s, l]
        ==
        (
            m.p_chp[t, s, l] / chp_elec_eff
        )
        +
        (
            m.q_chp[t, s, l] / chp_heat_eff
        )
    )

model.c_chp_fuel = am.Constraint(
    model.T,
    model.S,
    model.L,
    rule=chp_fuel_rule
)

# =========================================================
# CHP CONSTRAINTS
# =========================================================

model.c_chp = am.ConstraintList()

for t in years:

    for s in seasons:

        for l in load_zones:

            active_chp = sum(
                model.y_tech['CHP', w]
                for w in years
                if w <= t and t < w + lifespan['CHP']
            )

            model.c_chp.add(
                model.p_chp[t, s, l]
                <=
                BIG_M * model.u_chp_on[t, s, l]
            )

            model.c_chp.add(
                model.q_chp[t, s, l]
                <=
                BIG_M * model.u_chp_on[t, s, l]
            )

            model.c_chp.add(
                active_chp
                >=
                EPS * model.u_chp_on[t, s, l]
            )

            model.c_chp.add(
                model.p_chp[t, s, l]
                >=
                100 * model.u_chp_on[t, s, l]
            )

            model.c_chp.add(
                model.p_chp[t, s, l]
                <=
                300 * model.u_chp_on[t, s, l]
            )

            model.c_chp.add(
                model.p_chp[t, s, l]
                +
                model.q_chp[t, s, l]
                <=
                active_chp * af['CHP']
            )

            model.c_chp.add(
                model.q_chp[t, s, l]
                <=
                (
                    m_top
                    *
                    model.p_chp[t, s, l]
                    +
                    b_top
                )
            )

            model.c_chp.add(
                model.q_chp[t, s, l]
                >=
                (
                    m_bot
                    *
                    model.p_chp[t, s, l]
                    +
                    b_bot
                    -
                    BIG_M
                    *
                    (
                        1 - model.u_chp_on[t, s, l]
                    )
                )
            )

# =========================================================
# GRID ACCESS
# =========================================================

def grid_access_rule(m, c, t, s, l):

    return (
        m.purchased[c, t, s, l]
        <=
        BIG_M * m.u_grid[c]
    )


model.c_grid_access = am.Constraint(
    model.Carrier,
    model.T,
    model.S,
    model.L,
    rule=grid_access_rule
)

# =========================================================
# FORCE GRID SUBSCRIPTION ONLY IF USED
# =========================================================

model.c_grid_use = am.ConstraintList()

for c in grid_carriers:

    total_purchase = sum(
        model.purchased[c, t, s, l]
        for t in years
        for s in seasons
        for l in load_zones
    )

    model.c_grid_use.add(
        total_purchase
        <=
        BIG_M * model.u_grid[c]
    )

    model.c_grid_use.add(
        total_purchase
        >=
        EPS * model.u_grid[c]
    )

# =========================================================
# EMISSION LIMIT CONSTRAINT
# =========================================================

def emission_limit_rule(m):
    total_emissions = sum(
        days_per_season[s] * (
            m.purchased['Elec', t, s, l] * ELEC_EMISSION_FACTOR +
            m.purchased['Gas', t, s, l] * GAS_EMISSION_FACTOR
        )
        for t in m.T for s in m.S for l in m.L
    )
    return total_emissions <= TOTAL_EMISSION_CAP

model.c_emission_limit = am.Constraint(
    rule=emission_limit_rule
)

# =========================================================
# SOLVER
# =========================================================

solver = am.SolverFactory('highs')

results = solver.solve(
    model,
    tee=True
)

# =========================================================
# RESULTS
# =========================================================

print("\n========================================")
print("OPTIMIZATION COMPLETED")
print("========================================")

print(f"\nObjective Value = {am.value(model.obj):,.2f}")


subs_cost_val = sum(
    am.value(model.u_grid[c]) * sub_cost[c]
    for c in grid_carriers
)

cap_cost_val = sum(

    (1 / (1 + discount_rate) ** w)

    *

    sum(
        am.value(model.y_tech[tech, w]) * cap_cost[tech]
        for tech in techs
    )

    for w in years
)

maint_cost_val = sum(

    (1 / (1 + discount_rate) ** t)

    *

    sum(

        maint_cost[tech]
        *
        am.value(model.active_cap_used[tech, t])

        for tech in techs
    )

    for t in years
)

fuel_cost_val = sum(

    (1 / (1 + discount_rate) ** t)

    *

    sum(

        days_per_season[s]

        *

        sum(

            am.value(model.purchased[c, t, s, l])
            *
            grid_price[c]

            for c in grid_carriers
            for l in load_zones
        )

        for s in seasons
    )

    for t in years
)

# --- Carbon Cost and Total Emissions for Export ---
carbon_cost_val = sum(
    days_per_season[s] * (
        am.value(model.purchased['Elec', t, s, l]) * ELEC_EMISSION_FACTOR +
        am.value(model.purchased['Gas', t, s, l]) * GAS_EMISSION_FACTOR
    ) * CARBON_TAX_PER_TON
    for t in years for s in seasons for l in load_zones
)

total_emissions_val = sum(
    days_per_season[s] * (
        am.value(model.purchased['Elec', t, s, l]) * ELEC_EMISSION_FACTOR +
        am.value(model.purchased['Gas', t, s, l]) * GAS_EMISSION_FACTOR
    )
    for t in years for s in seasons for l in load_zones
)


objective_breakdown_df = pd.DataFrame([{

    'Subscription Cost': subs_cost_val,
    'Capital Cost': cap_cost_val,
    'O&M Cost': maint_cost_val,
    'Fuel Cost': fuel_cost_val,
    'Carbon Tax Cost': carbon_cost_val,
    'Total Objective': am.value(model.obj),
    'Total Emissions (tons)': total_emissions_val
}])

# =========================================================
# INSTALLED CAPACITY
# =========================================================

capacity_rows = []

for tech in techs:

    for w in years:

        val = am.value(model.y_tech[tech, w])

        if val is not None and val > 1e-6:

            capacity_rows.append({

                'Technology': tech,
                'Install Year': w,
                'Installed Capacity': val,
                'Lifetime': lifespan[tech]
            })

capacity_df = pd.DataFrame(capacity_rows)

# =========================================================
# YEARLY O&M COSTS
# =========================================================

om_rows = []

for t in years:

    for tech in techs:

        active_cap = sum(
            am.value(model.y_tech[tech, w])
            for w in years
            if w <= t and t < w + lifespan[tech]
        )

        used_cap = am.value(
            model.active_cap_used[tech, t]
        )

        om_paid = (
            maint_cost[tech]
            * used_cap
        )

        discounted_om = (
            om_paid
            *
            (1 / (1 + discount_rate) ** t)
        )

        om_rows.append({

            'Year': t,
            'Technology': tech,

            'Installed Active Capacity':
                active_cap,

            'Used Capacity Paying O&M':
                used_cap,

            'O&M Cost':
                om_paid,

            'Discounted O&M Cost':
                discounted_om
        })

om_df = pd.DataFrame(om_rows)

# =========================================================
# DETAILED OPERATIONS
# =========================================================

operation_rows = []

for t in years:

    growth = (1 + demand_growth) ** (t - 1)

    # --- Efficiency Reduction Factor for Result Mapping ---
    red_factor = 1.0
    if enable_efficiency_tech and t >= 5:
        red_factor = 0.8 ** (t - 4)
    # ------------------------------------------------------

    for s in seasons:

        for l in load_zones:

            # -------------------------------------------------
            # DEMANDS
            # -------------------------------------------------

            elec_demand = demand_data[s, l][0]

            heat_demand = (
                demand_data[s, l][1]
                * growth
                * red_factor
            )

            cool_demand = (
                demand_data[s, l][2]
                * growth
                * red_factor
            )

            # -------------------------------------------------
            # PV
            # -------------------------------------------------

            pv_output = am.value(
                model.x_tech['PV', t, s, l]
            )

            # -------------------------------------------------
            # CHP
            # -------------------------------------------------

            chp_elec = am.value(
                model.p_chp[t, s, l]
            )

            chp_heat = am.value(
                model.q_chp[t, s, l]
            )

            chp_gas = am.value(
                model.chp_gas[t, s, l]
            )

            # -------------------------------------------------
            # Boiler
            # -------------------------------------------------

            boiler_heat = am.value(
                model.x_tech['Boiler', t, s, l]
            )

            boiler_gas = (
                boiler_heat / boiler_eff
            )

            # -------------------------------------------------
            # Split AC
            # -------------------------------------------------

            split_cooling = am.value(
                model.x_tech['SplitAC', t, s, l]
            )

            split_elec = (
                split_cooling / splitac_cop
            )

            # -------------------------------------------------
            # EWH
            # -------------------------------------------------

            ewh_heat = am.value(
                model.x_tech['EWH', t, s, l]
            )

            ewh_elec = (
                ewh_heat / ewh_eff
            )

            # -------------------------------------------------
            # GRID PURCHASES
            # -------------------------------------------------

            elec_purchase = am.value(
                model.purchased['Elec', t, s, l]
            )

            gas_purchase = am.value(
                model.purchased['Gas', t, s, l]
            )

            operation_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,

                # DEMANDS
                'Electricity Demand':
                    elec_demand,

                'Heat Demand':
                    heat_demand,

                'Cooling Demand':
                    cool_demand,

                # PV
                'PV Electricity Output':
                    pv_output,

                # CHP
                'CHP Electricity Output':
                    chp_elec,

                'CHP Heat Output':
                    chp_heat,

                'CHP Gas Input':
                    chp_gas,

                # Boiler
                'Boiler Heat Output':
                    boiler_heat,

                'Boiler Gas Input':
                    boiler_gas,

                # Split AC
                'SplitAC Cooling Output':
                    split_cooling,

                'SplitAC Electricity Input':
                    split_elec,

                # EWH
                'EWH Heat Output':
                    ewh_heat,

                'EWH Electricity Input':
                    ewh_elec,

                # Grid
                'Electricity Purchase':
                    elec_purchase,

                'Gas Purchase':
                    gas_purchase,

                # CHP STATUS
                'CHP ON':
                    am.value(
                        model.u_chp_on[t, s, l]
                    )
            })

operations_df = pd.DataFrame(operation_rows)

# =========================================================
# OPERATIONAL COST DETAILS
# =========================================================

op_cost_rows = []

for t in years:

    discount_factor = (
        1 / (1 + discount_rate) ** t
    )

    for s in seasons:

        days = days_per_season[s]

        for l in load_zones:

            # =================================================
            # CHP
            # =================================================

            chp_elec = am.value(
                model.p_chp[t, s, l]
            )

            chp_heat = am.value(
                model.q_chp[t, s, l]
            )

            chp_gas = am.value(
                model.chp_gas[t, s, l]
            )

            chp_cost = (
                chp_gas
                *
                grid_price['Gas']
                *
                days
                *
                discount_factor
            )

            op_cost_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Technology': 'CHP',

                'Electricity Output':
                    chp_elec,

                'Heat Output':
                    chp_heat,

                'Cooling Output':
                    0,

                'Electricity Input':
                    0,

                'Gas Input':
                    chp_gas,

                'Operating Cost':
                    chp_cost
            })

            # =================================================
            # BOILER
            # =================================================

            boiler_heat = am.value(
                model.x_tech['Boiler', t, s, l]
            )

            boiler_gas = (
                boiler_heat / boiler_eff
            )

            boiler_cost = (
                boiler_gas
                *
                grid_price['Gas']
                *
                days
                *
                discount_factor
            )

            op_cost_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Technology': 'Boiler',

                'Electricity Output':
                    0,

                'Heat Output':
                    boiler_heat,

                'Cooling Output':
                    0,

                'Electricity Input':
                    0,

                'Gas Input':
                    boiler_gas,

                'Operating Cost':
                    boiler_cost
            })

            # =================================================
            # SPLIT AC
            # =================================================

            split_cooling = am.value(
                model.x_tech['SplitAC', t, s, l]
            )

            split_elec = (
                split_cooling / splitac_cop
            )

            split_cost = (
                split_elec
                *
                grid_price['Elec']
                *
                days
                *
                discount_factor
            )

            op_cost_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Technology': 'SplitAC',

                'Electricity Output':
                    0,

                'Heat Output':
                    0,

                'Cooling Output':
                    split_cooling,

                'Electricity Input':
                    split_elec,

                'Gas Input':
                    0,

                'Operating Cost':
                    split_cost
            })

            # =================================================
            # EWH
            # =================================================

            ewh_heat = am.value(
                model.x_tech['EWH', t, s, l]
            )

            ewh_elec = (
                ewh_heat / ewh_eff
            )

            ewh_cost = (
                ewh_elec
                *
                grid_price['Elec']
                *
                days
                *
                discount_factor
            )

            op_cost_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Technology': 'EWH',

                'Electricity Output':
                    0,

                'Heat Output':
                    ewh_heat,

                'Cooling Output':
                    0,

                'Electricity Input':
                    0,

                'Gas Input':
                    0,

                'Operating Cost':
                    ewh_cost
            })

            # =================================================
            # PV
            # =================================================

            pv_output = am.value(
                model.x_tech['PV', t, s, l]
            )

            op_cost_rows.append({

                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Technology': 'PV',

                'Electricity Output':
                    pv_output,

                'Heat Output':
                    0,

                'Cooling Output':
                    0,

                'Electricity Input':
                    0,

                'Gas Input':
                    0,

                'Operating Cost':
                    0
            })

op_cost_df = pd.DataFrame(op_cost_rows)

# =========================================================
# GRID SUBSCRIPTION DETAILS
# =========================================================

grid_rows = []

for c in grid_carriers:

    total_purchase = sum(
        am.value(model.purchased[c, t, s, l])
        for t in years
        for s in seasons
        for l in load_zones
    )

    total_cost = sum(

        (
            1 / (1 + discount_rate) ** t
        )

        *

        days_per_season[s]

        *

        am.value(model.purchased[c, t, s, l])

        *

        grid_price[c]

        for t in years
        for s in seasons
        for l in load_zones
    )

    grid_rows.append({

        'Carrier': c,

        'Subscribed':
            am.value(model.u_grid[c]),

        'Subscription Cost':
            (
                am.value(model.u_grid[c])
                * sub_cost[c]
            ),

        'Total Purchased':
            total_purchase,

        'Total Purchase Cost':
            total_cost
    })

grid_df = pd.DataFrame(grid_rows)

# =========================================================
# DETAILED EMISSIONS EXPORT
# =========================================================

emissions_data = []
for t in years:
    for s in seasons:
        for l in load_zones:
            elec_purchased = am.value(model.purchased['Elec', t, s, l])
            gas_purchased = am.value(model.purchased['Gas', t, s, l])
            elec_emissions_per_day = elec_purchased * ELEC_EMISSION_FACTOR
            gas_emissions_per_day = gas_purchased * GAS_EMISSION_FACTOR
            total_period_emissions = (elec_emissions_per_day + gas_emissions_per_day) * days_per_season[s]
            emissions_data.append({
                'Year': t,
                'Season': s,
                'Load Zone': l,
                'Electricity Purchased (kWh)': elec_purchased,
                'Gas Purchased (kWh)': gas_purchased,
                'Electricity Emissions (tons CO2/day)': elec_emissions_per_day,
                'Gas Emissions (tons CO2/day)': gas_emissions_per_day,
                'Total Period Emissions (tons CO2)': total_period_emissions
            })
detailed_emissions_df = pd.DataFrame(emissions_data)


# =========================================================
# EXPORT TO EXCEL
# =========================================================

excel_file = 'energy_system_results_FULL.xlsx'

# Round down any number less than 0.000001 (1e-6) to 0 across all DataFrames
# Using .map() instead of .applymap() to avoid deprecation warnings
objective_breakdown_df = objective_breakdown_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
capacity_df = capacity_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
om_df = om_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
operations_df = operations_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
op_cost_df = op_cost_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
grid_df = grid_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)
detailed_emissions_df = detailed_emissions_df.map(lambda x: 0 if isinstance(x, (int, float)) and x < 1e-6 else x)

with pd.ExcelWriter(
    excel_file,
    engine='openpyxl'
) as writer:

    objective_breakdown_df.to_excel(
        writer,
        sheet_name='Objective Breakdown',
        index=False
    )

    capacity_df.to_excel(
        writer,
        sheet_name='Installed Capacity',
        index=False
    )

    om_df.to_excel(
        writer,
        sheet_name='Yearly O&M Costs',
        index=False
    )

    operations_df.to_excel(
        writer,
        sheet_name='Detailed Operations',
        index=False
    )

    op_cost_df.to_excel(
        writer,
        sheet_name='Operational Costs',
        index=False
    )

    grid_df.to_excel(
        writer,
        sheet_name='Grid Subscription',
        index=False
    )

    detailed_emissions_df.to_excel(
        writer,
        sheet_name='Detailed Emissions',
        index=False
    )

print(f"\nResults exported to: {excel_file}")
