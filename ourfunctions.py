import random
import numpy as np
import pandas as pd
import time 
from gurobipy import Model, GRB, quicksum
import math
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Dirichlet

import os
from functools import lru_cache
from collections import defaultdict
import importlib.util
from pathlib import Path


K = 25  #number of PRB
T = 10  #number of Slot

    
# Apps SLA dictionaries in bits
SLA_critic = {1: 400, 2:900}   
SLA_perf   = {3: 10000, 4: 20}  
SLA_busi   = {5: 20000}        


# Application categories identifier
Ac = [1, 2]  # critical 1: ETCS  2: Voice 
Ap = [3, 4]  # performance  3: CCTV , 4: PID
Ab = [5]     # Business 5: Pwifi

U_c = 6 # number of critical UE
U_p = 5 # number of performance UE
U_b = 4 # number of business UE
hidden_size=128
n_steps=250
agent_gamma = 0.99
lam=0.25
clip_range=0.2
lr=0.005
batch_size=64
n_epochs=8
alpha=0.8
lambda_weight=0.5
weights = {1:120, 2:8, 3:2}
entropy_coef=0.02 
value_coef = 0.5
partial_train_steps= 2000
train_episode= 40
violation_threshold = 2
usage_threshold = 99
frames = []
# Application labels
app_labels = ['ETCS', 'Voice', 'CCTV', 'PIS', 'Pwifi']
sla_values = [
    SLA_critic.get(1, 0),
    SLA_critic.get(2, 0),
    SLA_perf.get(3, 0),
    SLA_perf.get(4, 0),
    SLA_busi.get(5, 0)
]

#Data Rate parameters

J = 1  # Number of component carriers
v_Llayers = 1  # Maximum number of MIMO layers

OH = 0.14  # Overhead (e.g., FR1 DL)
f = 1  # Scaling factor
mu = 0  # Numerology (0 for 15 kHz SCS)

# CQI to Modulation Order (Q_m)
CQI_TO_MCS = {
    1: 2, 2: 2, 3: 2, 4: 4, 5: 4,
    6: 4, 7: 4, 8: 6, 9: 6, 10: 6,
    11: 6, 12: 6, 13: 8, 14: 8, 15: 8
}

# CQI to Coding Rate (R_max)
CQI_TO_Rmax = {
    1: 0.07617188,  # QPSK
    2: 0.18847656,  # QPSK
    3: 0.43847656,  # QPSK
    4: 0.36914063,  # QPSK
    5: 0.47851563,  # QPSK
    6: 0.6015625,   # QPSK
    7: 0.45507813,  # 16-QAM
    8: 0.55371094,  # 16-QAM
    9: 0.65039063,  # 16-QAM
    10: 0.75390625, # 64-QAM
    11: 0.85253906, # 64-QAM
    12: 0.69433594, # 64-QAM
    13: 0.77832031, # 64-QAM
    14: 0.86425781, # 64-QAM
    15: 0.92578125, # 64-QAM
}

# OFDM symbol duration (T_s) based on numerology
T_s = 10**-3 / (14 * 2**mu)  

def calculate_minmum_required_datarate():
    # Sum required bits for critical, performance and business  apps 
    crit_req = U_c * (SLA_critic[1] + SLA_critic[2])
    
    perf_req = U_p * (SLA_perf[3] + SLA_perf[4])
    
    busi_req = U_b * SLA_busi[5]
    
    total_required_bits = crit_req + perf_req + busi_req
    total_required_bits = total_required_bits/1000 
    return total_required_bits
min_required_datarate_KB = calculate_minmum_required_datarate()


# ------------------------------------------------------------
# generate_gamma(): using real sub-band-CQI measurements
# ------------------------------------------------------------

DATA_DIR = "subband_cqi_v2"            # folder that holds TOBA gateway subband CQI : UE0.csv … UE<n>.csv


@lru_cache(maxsize=None)
def _load_ue_dataframe(ue_idx: int) -> pd.DataFrame:
    """Read (and cache) the CSV of one UE."""
    path = os.path.join(DATA_DIR, f"Subband_UE{ue_idx}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing CQI file: {path}")
    return pd.read_csv(path)

def _pick_row(df: pd.DataFrame, time_idx: Optional[int] = None) -> pd.Series:
    """Return one row of CQI values (RGB0…RGB6)."""
    if time_idx is None:                       # if no index specified choose a random instant
        return df.sample(n=1).iloc[0]
    else:                                      # else deterministic index specified
        return df.iloc[time_idx % len(df)]

def _rgb_for_prb(k_: int) -> int:
    """Map PRB index → RGB index (0-6)."""
    return k_ // 4 if k_ < 24 else 6

def generate_gamma(time_idx: Optional[int] = None):
    """
    Build the γ-dictionary using real CQI traces.
    these achievable data rate base on subband CQI
    Parameters
    ----------
    time_idx : int or None
        • None  → use a random row from each UE file (default)  
        • int   → deterministic: row `time_idx` 
    """
    gamma = {}

    def add_entries(local_i: int, global_i: int, slice_id: int, apps: list[int]):
        # --- pick one CQI row for this UE --------------------
        row = _pick_row(_load_ue_dataframe(global_i), time_idx)
        rgb_cqi = [int(row[f"RGB{g}"]) for g in range(7)]

        for a in apps:                        # same CQI for every app of that UE (=TOBA Garteway)
            for k_ in range(K):               # PRB ➜ RGB mapping
                cqi = rgb_cqi[_rgb_for_prb(k_)]
                R_max = CQI_TO_Rmax.get(cqi, 0)
                Q_m   = CQI_TO_MCS .get(cqi, 0)
                thr = (1e-3 * J * v_Llayers * Q_m * f * R_max * (12 / T_s) * (1 - OH))
                gamma[(global_i, a, slice_id, k_)] = thr   # **global_i here**

    # ── enter appplications into gamma based on specifie Number of user in Ac, Ab and Ac (Critic. Perf.Busi)
    for i in range(U_c):
        add_entries(i, i, 1, Ac)
    
    for i in range(U_p):
        add_entries(i, U_c + i, 2, Ap)

    for i in range(U_b):
        add_entries(i, U_c + U_p + i, 3, Ab)

    return gamma

#build list of applications
def build_appkey_list():
    appkey_list = []

    # Critical slice (UE 0-5)
    for i in range(U_c):
        for a in Ac:
            appkey_list.append((i, a, 1))

    # Performance slice (UE 6-10)
    for i in range(U_p):
        g = U_c + i
        for a in Ap:
            appkey_list.append((g, a, 2))

    # Business slice (UE 11-14)
    for i in range(U_b):
        g = U_c + U_p + i
        for a in Ab:
            appkey_list.append((g, a, 3))

    return appkey_list

APPKEY_LIST = build_appkey_list()
N_APPKEYS = len(APPKEY_LIST)

@lru_cache(maxsize=None)
def get_sla(app):
    if app in SLA_critic:
        return SLA_critic[app]
    if app in SLA_perf:
        return SLA_perf[app]
    if app in SLA_busi:
        return SLA_busi[app]
    return 0

@lru_cache(maxsize=None)
def get_slice_for_app(app_id):
    if app_id in Ac:
        return 1
    elif app_id in Ap:
        return 2
    else:
        return 3

@lru_cache(maxsize=None)
def get_num_users_for_slice(slice_id):
    if slice_id == 1:
        return U_c
    elif slice_id == 2:
        return U_p
    elif slice_id == 3:
        return U_b

def run_allocation_solver(gamma, wc=0.01, wp=0.6, wb=1):
    """
    Optimal ILP (Gurobi) allocation 
    """

    # Sort TOBA gatexyas (UE)  ids per slices
    crit_users = sorted({i for (i, a, s, _) in gamma if s == 1})  
    perf_users = sorted({i for (i, a, s, _) in gamma if s == 2})  
    busi_users = sorted({i for (i, a, s, _) in gamma if s == 3})  

    U_c, U_p, U_b = len(crit_users), len(perf_users), len(busi_users)

    # ------------------------------------------------------------
    # 1)  Model & variables 
    # ------------------------------------------------------------
    model = Model("Optimal_ILP")
    model.Params.LogToConsole = 0
    model.setParam("NodefileStart", 0.010)
    model.setParam("NodefileDir", "/tmp")
    model.setParam("Threads", 16)
    model.setParam("MIPFocus", 1)
    model.setParam("ConcurrentMIP", 4)
    model.setParam("TimeLimit", 10) 

    x = model.addVars(U_c, len(Ac), K, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(U_p, len(Ap), K, T, vtype=GRB.BINARY, name="y")
    z = model.addVars(U_b, len(Ab), K, T, vtype=GRB.BINARY, name="z")

    SlackCrit = model.addVars(U_c, len(Ac), vtype=GRB.CONTINUOUS, lb=0)
    SlackPerf = model.addVars(U_p, len(Ap), vtype=GRB.CONTINUOUS, lb=0)
    SlackBus  = model.addVars(U_b, len(Ab), vtype=GRB.CONTINUOUS, lb=0)

    # 2)  PRB-capacity constraints 

    for t_ in range(T):
        for k_ in range(K):
            model.addConstr(quicksum(x[i, a, k_, t_] for i in range(U_c) for a in range(len(Ac))) <= 1)
            model.addConstr(quicksum(y[i, a, k_, t_] for i in range(U_p) for a in range(len(Ap))) <= 1)
            model.addConstr(quicksum(z[i, a, k_, t_] for i in range(U_b) for a in range(len(Ab))) <= 1)

            model.addConstr(
                quicksum(x[i, a, k_, t_] for i in range(U_c) for a in range(len(Ac))) +
                quicksum(y[i, a, k_, t_] for i in range(U_p) for a in range(len(Ap))) +
                quicksum(z[i, a, k_, t_] for i in range(U_b) for a in range(len(Ab))) <= 1
            )

    # 3)  SLA constraints for each Slice

    for i_loc in range(U_c):
        i_glob = crit_users[i_loc]
        for a_idx, a in enumerate(Ac):
            model.addConstr(
                quicksum(
                    x[i_loc, a_idx, k_, t_] * gamma.get((i_glob, a, 1, k_), 0)
                    for k_ in range(K) for t_ in range(T)
                ) >= SLA_critic[a]
            )

    for i_loc in range(U_p):
        i_glob = perf_users[i_loc]
        for a_idx, a in enumerate(Ap):
            model.addConstr(
                quicksum(
                    y[i_loc, a_idx, k_, t_] * gamma.get((i_glob, a, 2, k_), 0)
                    for k_ in range(K) for t_ in range(T)
                )+ SlackPerf[i_loc, a_idx] >= SLA_perf[a]
            )

    for i_loc in range(U_b):
        i_glob = busi_users[i_loc]
        for a_idx, a in enumerate(Ab):
            model.addConstr(
                quicksum(
                    z[i_loc, a_idx, k_, t_] * gamma.get((i_glob, a, 3, k_), 0)
                    for k_ in range(K) for t_ in range(T)
                ) + SlackBus[i_loc, a_idx] >= SLA_busi[a]
            )

    # 4)  Objective 

    alpha_c, alpha_p, alpha_b = 2000, 2000, 1000
    model.setObjective(
        wp * quicksum(y[i, a, k_, t] for i in range(U_p) for a in range(len(Ap)) for k_ in range(K) for t in range(T))
      + wb * quicksum(z[i, a, k_, t] for i in range(U_b) for a in range(len(Ab)) for k_ in range(K) for t in range(T))
      + wc * quicksum(x[i, a, k_, t] for i in range(U_c) for a in range(len(Ac)) for k_ in range(K) for t in range(T))
      + alpha_b * quicksum(SlackBus[i, a] / SLA_busi[Ab[a]] for i in range(U_b) for a in range(len(Ab)))
      + alpha_p * quicksum(SlackPerf[i, a] / SLA_perf[Ap[a]] for i in range(U_p) for a in range(len(Ap))),
      GRB.MINIMIZE
    )

 
    # 5)  Solve

    try:
        model.optimize()
    except Exception as e:
        print(f"Gurobi optimization failed: {e}")
        return (None,)*7

    if model.SolCount == 0:          # no feasible solution found at all
        print("No incumbent solution.")
        return (None,)*7

    # ------------------------------------------------------------
    # 6)  Extract the solution 
    # ------------------------------------------------------------
    prb_assignments, total_throughput = [], 0.0

    for t_ in range(T):
        for k_ in range(K):

            # --- critical slice
            for i_loc in range(U_c):
                i_glob = crit_users[i_loc]
                for a_idx, a in enumerate(Ac):
                    if x[i_loc, a_idx, k_, t_].X > 0.5:
                        thr = gamma.get((i_glob, a, 1, k_), 0)
                        prb_assignments.append((t_, k_, i_glob, a, 1, thr))
                        total_throughput += thr

            # --- performance slice
            for i_loc in range(U_p):
                i_glob = perf_users[i_loc]
                for a_idx, a in enumerate(Ap):
                    if y[i_loc, a_idx, k_, t_].X > 0.5:
                        thr = gamma.get((i_glob, a, 2, k_), 0)
                        prb_assignments.append((t_, k_, i_glob, a, 2, thr))
                        total_throughput += thr

            # --- business slice
            for i_loc in range(U_b):
                i_glob = busi_users[i_loc]
                for a_idx, a in enumerate(Ab):
                    if z[i_loc, a_idx, k_, t_].X > 0.5:
                        thr = gamma.get((i_glob, a, 3, k_), 0)
                        prb_assignments.append((t_, k_, i_glob, a, 3, thr))
                        total_throughput += thr

    # ------------------------------------------------------------
    # 7)  Metrics & DataFrames 
    # ------------------------------------------------------------
    allocation_df = pd.DataFrame(prb_assignments,
                                 columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput'])
    slice_throughput_df = (
        allocation_df
          .groupby('Slice')['Throughput']
          .sum()
          .reset_index()
          .rename(columns={'Throughput': 'Total_Throughput'})
    )
    allocation_df['App_Key'] = allocation_df.apply(
        lambda r: (r['User'], r['App'], r['Slice']), axis=1)

    app_throughput_df = (allocation_df.groupby('App_Key')['Throughput']
                         .sum().reset_index())
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index)
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    app_episode_thr = (app_throughput_df.groupby('App')['Throughput']
                       .mean().reset_index()
                       .rename(columns={'Throughput': 'Average_Throughput'}))


    violation_counts_per_slice = {1: 0, 2: 0, 3: 0}
    violation_gap_per_slice    = {1: 0, 2: 0, 3: 0}
    for _, row in app_throughput_df.iterrows():
        i, a, s, thr = row['User'], row['App'], row['Slice'], row['Throughput']
        sla = SLA_critic.get(a, 0) if s == 1 else SLA_perf.get(a, 0) if s == 2 else SLA_busi.get(a, 0)
        if thr < sla:
            violation_counts_per_slice[s] += 1
            violation_gap_per_slice[s]   += abs(sla - thr)

    total_violations = sum(violation_counts_per_slice.values())
    violation_gap_per_slice[1] /= ((SLA_critic[1]+SLA_critic[2]) * U_c)
    violation_gap_per_slice[2] /= ((SLA_perf[3]+SLA_perf[4])     * U_p)
    violation_gap_per_slice[3] /= (SLA_busi[5]                   * U_b)

    return (allocation_df, total_throughput, app_throughput_df,
            app_episode_thr, violation_counts_per_slice,
            total_violations, violation_gap_per_slice, slice_throughput_df)


def run_bestcqi_allocation(gamma):
    # Assign each PRB in each time slot to (i, a, s) that has the highest achievable datarate in gamma

    # Initialize per-app cumulative allocation counter
    allocated_so_far = { key: 0.0 for key in APPKEY_LIST }
    drift = 0.8
    prb_assignments = []

    for t in range(T):
        for k_ in range(K):
            best_thr = -float('inf')
            best_keys = []  # will hold all (i,a,s) that achieve best_thr

            # Scan through all candidates for this PRB k_
            for (i, a, s, prb) in gamma:
                if prb == k_:
                    thr_val = gamma[(i, a, s, prb)]
                    if thr_val > best_thr:
                        best_thr = thr_val
                        best_keys = [(i, a, s)]
                    elif thr_val == best_thr:
                        best_keys.append((i, a, s))

            # If we found at least one, pick one
            if best_keys:
                i_sel, a_sel, s_sel = random.choice(best_keys)

                # Compute remaining demand vs. SLA
                app_key = (i_sel, a_sel, s_sel)
                sla = get_sla(a_sel)
                remaining = sla - allocated_so_far[app_key]

                # Allocate zero if SLA already met; otherwise cap by remaining
                if remaining <= 0:
                    #delta = 0.0
                    delta = best_thr * 0.5
                    prb_assignments.append((t, k_, i_sel, a_sel, s_sel, delta))
                    allocated_so_far[app_key] += delta
                else:
                    delta = best_thr
                    delta= delta * drift
                    # Record and update
                    prb_assignments.append((t, k_, i_sel, a_sel, s_sel, delta))
                    allocated_so_far[app_key] += delta


    allocation_df = pd.DataFrame(prb_assignments, 
                                 columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput'])
    slice_throughput_df = (
        allocation_df
          .groupby('Slice')['Throughput']
          .sum()
          .reset_index()
          .rename(columns={'Throughput': 'Total_Throughput'})
    )    
    # allocation_df.to_csv("bestcqi_allocation_df.csv", index=False)
    allocation_df['App_Key'] = allocation_df.apply(
        lambda row: (row['User'], row['App'], row['Slice']), axis=1
    )
    
    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput'].sum().reset_index()
    )

    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)
    # app_throughput_df.to_csv("bestcqi_allocation_df.csv", index=False)
    
    # ===  Create a DataFrame of all ID combos from APPKEY_LIST ===
    all_combos_df = pd.DataFrame(APPKEY_LIST, columns=['User','App','Slice'])

    # ===  Merge with app_throughput_df, filling missing ID combos as Throughput=0 ===
    merged_df = all_combos_df.merge(
        app_throughput_df,
        on=['User','App','Slice'],
        how='left'
    )
    merged_df['Throughput'] = merged_df['Throughput'].fillna(0.0)

    # === compute the average throughput per App ===
    app_episode_thr = (
        merged_df
        .groupby('App')['Throughput']
        .mean()
        .reset_index()
        .rename(columns={'Throughput': 'Average_Throughput'})
    )
    merged_df.to_csv("CQI_episode_app.csv", index=True)
    violation_counts_per_slice = {1: 0, 2: 0, 3: 0}
    violation_gap_per_slice = {1: 0, 2: 0, 3: 0}

    for (i_, a_, s_) in APPKEY_LIST:
        # Grab how much throughput was allocated to this (user, app, slice)
        row_match = app_throughput_df[
            (app_throughput_df['User'] == i_) &
            (app_throughput_df['App'] == a_) &
            (app_throughput_df['Slice'] == s_)
        ]
        sla_needed = get_sla(a_)
        if len(row_match) > 0:
            allocated_thr = row_match.iloc[0]['Throughput']
            ratio = allocated_thr / sla_needed
            if ratio < 1 :
                violation_counts_per_slice[s_] += 1
                violation_gap_per_slice[s_] += (1.0 - ratio)
        else:
            violation_counts_per_slice[s_] += 1
            violation_gap_per_slice[s_] += 1.0

    for s_ in [1, 2, 3]:
        if violation_counts_per_slice[s_] > 0:
            violation_gap_per_slice[s_] /= violation_counts_per_slice[s_]
        else:
            violation_gap_per_slice[s_] = 0.0

    total_violations = sum(violation_counts_per_slice.values())
    total_throughput = allocation_df['Throughput'].sum()

    return (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,slice_throughput_df
    )


def run_myheuristic_allocation(gamma):
    """
    Two-phase cascaded scheduler:
      1) Phase Critique : minimal-slack greedy
      2) Phase Performance & Business : weighted Best-CQI
    """
    w_perf= 0.5
    w_busi= 0.5
    drift = 1
    # Initialise slack bits for every app
    slack = {key: get_sla(key[1]) for key in APPKEY_LIST}
    prb_assignments = []

    # Phase 1: Critical slice greedy slack reduction
    for t in range(T):
        for k in range(K):
            best_delta = 0.0
            best_choice = None
            # scan critical apps
            for (i, a, s, prb), g in gamma.items():
                if prb != k or s != 1:
                    continue
                need = slack[(i, a, s)]
                if need <= 0:
                    continue
                delta = min(need, g)
                if delta > best_delta:
                    best_delta = delta
                    best_choice = (i, a, s, g)
            if best_choice:
                i, a, s, g = best_choice
                slack[(i, a, s)] -= best_delta
                prb_assignments.append((t, k, i, a, s, g))

          # Phase 2: Weighted Best-CQI for Performance & Business
            weights = {2: w_perf, 3: w_busi}

            # skip PRBs already allocated in Phase 1
            if any(t == rec[0] and k == rec[1] for rec in prb_assignments):
                continue
            best_score = 0.0
            best_choice = None
            for (i, a, s, prb), g in gamma.items():
                if prb != k or s not in (2, 3):
                    continue
                need = slack[(i, a, s)]
                if need <= 0:
                    continue
                score = g * weights[s]
                if score > best_score:
                    best_score = score
                    best_choice = (i, a, s, g)
            if best_choice:
                i, a, s, g = best_choice
                delta = min(slack[(i, a, s)], g)
                g = g * drift
                slack[(i, a, s)] -= g
                prb_assignments.append((t, k, i, a, s, g))

    # Build allocation DataFrame
    allocation_df = pd.DataFrame(
        prb_assignments,
        columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput']
    )
    slice_throughput_df = (
        allocation_df
          .groupby('Slice')['Throughput']
          .sum()
          .reset_index()
          .rename(columns={'Throughput': 'Total_Throughput'})
    )
    allocation_df['App_Key'] = allocation_df.apply(
        lambda r: (r['User'], r['App'], r['Slice']), axis=1
    )

    # Per-app throughput
    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput']
                     .sum().reset_index()
    )
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    # Include zero rows for apps with no allocation
    all_combos = pd.DataFrame(APPKEY_LIST, columns=['User','App','Slice'])
    merged = all_combos.merge(
        app_throughput_df, on=['User','App','Slice'], how='left'
    ).fillna(0.0)

    # Average throughput per episode
    app_episode_thr = (
        merged.groupby('App')['Throughput']
              .mean().reset_index()
              .rename(columns={'Throughput':'Average_Throughput'})
    )

    # Violation counts and gap ratios
    violation_counts_per_slice = {1:0, 2:0, 3:0}
    violation_gap_per_slice = {1:0.0,2:0.0,3:0.0}
    for (i, a, s) in APPKEY_LIST:
        sla = get_sla(a)
        allocated = merged.loc[
            (merged.User==i)&(merged.App==a)&(merged.Slice==s),'Throughput'
        ].iloc[0]
        ratio = allocated / sla if sla>0 else 1.0
        if ratio < 1:
            violation_counts_per_slice[s] += 1
            violation_gap_per_slice[s]   += (1 - ratio)
    for s in violation_gap_per_slice:
        if violation_counts_per_slice[s] > 0:
            violation_gap_per_slice[s] /= violation_counts_per_slice[s]

    total_throughput = allocation_df['Throughput'].sum()
    total_violations = sum(violation_counts_per_slice.values())

    return (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice, slice_throughput_df
    )



def run_radiosaber_allocation(gamma):
    """
    RadioSaber-inspired RBG scheduler with dynamic per-slot quotas and per-PRB payload capping.

    1) Compute slice weights based on remaining SLA demand per slot.
    2) For each slot, derive per-slice RBG quotas.
    3) For each RBG (with offset), pick first slice under quota & demand,
       then select the UE-App with max sum throughput over that RBG.
    4) Allocate each PRB within the RBG individually, capping by app's remaining payload.

    Returns:
      allocation_df, total_throughput, app_throughput_df,
      app_episode_thr, violation_counts_per_slice,
      total_violations, violation_gap_per_slice, slice_throughput_df
    """
    # RBG parameters
    RBG_size    = 4
    num_RBG     = math.ceil(K / RBG_size)
    slice_order = [1, 2, 3]

    # Initialize remaining payloads
    payload_to_send = {key: get_sla(key[1]) for key in APPKEY_LIST}
    slice_payload   = defaultdict(float)
    allocated       = defaultdict(float)
    for (u, a, s), rem in payload_to_send.items():
        slice_payload[s] += rem

    prb_assignments = []

    # ----- Per-slot scheduling -----
    for t in range(T):
        # 1) Dynamic weights & quotas based on remaining slice demand
        total_rem = sum(slice_payload.values())
        if total_rem > 0:
            weights = {s: slice_payload[s] / total_rem for s in slice_order}
        else:
            weights = {s: 0 for s in slice_order}
        quotas = {s: int(round(weights[s] * num_RBG)) for s in slice_order}
        used   = {s: 0 for s in slice_order}

        # 2) PRB sequence offset for RBG boundaries
        offset  = t % RBG_size
        prb_seq = list(range(offset, K)) + list(range(0, offset))

        # 3) RBG-level allocation
        for r in range(num_RBG):
            prb_group = prb_seq[r * RBG_size : (r + 1) * RBG_size]

            # Stage 1: slice selection under quota and demand
            sel_slice = None
            for s in slice_order:
                if used[s] < quotas[s] and slice_payload[s] > 0:
                    sel_slice = s
                    used[s] += 1
                    break
            if sel_slice is None:
                continue

            # Stage 2: enterprise scheduling (max RBG throughput)
            sum_thr = defaultdict(float)
            for (u, a, s_val, prb) in gamma:
                if s_val != sel_slice or prb not in prb_group:
                    continue
                remaining = payload_to_send.get((u, a, s_val), 0)
                if remaining <= 0:
                    continue
                sum_thr[(u, a)] += gamma[(u, a, s_val, prb)]

            # pick best UE-App pair
            if not sum_thr:
                continue
            best_val   = max(sum_thr.values())
            best_pairs = [ua for ua, val in sum_thr.items() if val == best_val]
            u_sel, a_sel = random.choice(best_pairs)

            # 4) per-PRB allocation with payload cap
            for prb in prb_group:
                thr_prb = gamma.get((u_sel, a_sel, sel_slice, prb), 0.0)
                if thr_prb <= 0:
                    continue
                remaining = payload_to_send.get((u_sel, a_sel, sel_slice), 0)
                if remaining <= 0:
                    break
                actual = min(thr_prb, remaining)
                # update trackers
                payload_to_send[(u_sel, a_sel, sel_slice)] -= actual
                slice_payload[sel_slice] -= actual
                allocated[(u_sel, a_sel, sel_slice)] += actual
                # record assignment
                prb_assignments.append((t, prb, u_sel, a_sel, sel_slice, actual))

    # ----- Build DataFrames & metrics -----
    allocation_df = pd.DataFrame(
        prb_assignments,
        columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput']
    )

    slice_throughput_df = (
        allocation_df.groupby('Slice')['Throughput']
                     .sum().reset_index()
                     .rename(columns={'Throughput': 'Total_Throughput'})
    )

    allocation_df['App_Key'] = allocation_df.apply(
        lambda r: (r['User'], r['App'], r['Slice']), axis=1
    )

    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput']
                     .sum().reset_index()
    )
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    all_combos_df = pd.DataFrame(APPKEY_LIST, columns=['User', 'App', 'Slice'])
    merged_df     = all_combos_df.merge(
        app_throughput_df, on=['User', 'App', 'Slice'], how='left'
    ).fillna({'Throughput': 0.0})

    app_episode_thr = (
        merged_df.groupby('App')['Throughput']
                 .mean().reset_index()
                 .rename(columns={'Throughput': 'Average_Throughput'})
    )

    merged_df.to_csv("CQI_episode_app.csv", index=True)

    violation_counts_per_slice = {s: 0 for s in slice_order}
    violation_gap_per_slice    = {s: 0.0 for s in slice_order}
    for (u, a, s) in APPKEY_LIST:
        sla_needed = get_sla(a)
        alloc_thr  = allocated.get((u, a, s), 0.0)
        if alloc_thr < sla_needed:
            violation_counts_per_slice[s] += 1
            violation_gap_per_slice[s]   += (1.0 - alloc_thr / sla_needed)

    for s in violation_gap_per_slice:
        cnt = violation_counts_per_slice[s]
        if cnt > 0:
            violation_gap_per_slice[s] /= cnt

    total_violations = sum(violation_counts_per_slice.values())
    total_throughput = allocation_df['Throughput'].sum()

    return (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )

def run_nvs_allocation(gamma):
    """
    NVS- static reservation and dynamic reuse of unused slots.
    
    gamma: dict mapping (i, app, slice, prb) -> achievable throughput
    APPKEY_LIST: list of all (user, app, slice) combos
    get_sla: function mapping app -> SLA throughput requirement
    K: total number of PRBs per slot
    """
    prb_assignments = []
    # 1) fixed reservation per slice as SLA (50%,30%,20%)
    slot_reservation = {1: 5, 2: 3, 3: 2}
    current_slot = 0

    # track cumulative allocated throughput per (i, app, slice)
    alloc_thr = defaultdict(float)
    
    # extract per-slice list of (user, app)
    slice_apps = {
        s: sorted({(i, a) for (i, a, s_val, k) in gamma if s_val == s})
        for s in slot_reservation
    }
    
    leftover_slots = []

    # First pass: allocate each slice's reserved slots
    for s, rsv_slots in slot_reservation.items():
        apps = slice_apps[s]
        if not apps:
            # no apps in this slice — skip and mark slots as leftover
            leftover_slots.extend(range(current_slot, current_slot + rsv_slots))
            current_slot += rsv_slots
            continue

        app_idx = 0
        for t in range(current_slot, current_slot + rsv_slots):
            # if all apps have met their SLA, free remaining slots
            if all(alloc_thr[(i, a, s)] >= get_sla(a) for (i, a) in apps):
                leftover_slots.extend(range(t, current_slot + rsv_slots))
                break

            # otherwise assign this slot in round-robin among apps
            for k in range(K):
                attempts = 0
                assigned = False
                while not assigned and attempts < len(apps):
                    i_app, a_app = apps[app_idx]
                    app_idx = (app_idx + 1) % len(apps)
                    key = (i_app, a_app, s, k)
                    if key in gamma:
                        thr = gamma[key]
                        prb_assignments.append((t, k, i_app, a_app, s, thr))
                        alloc_thr[(i_app, a_app, s)] += thr
                        assigned = True
                    attempts += 1
        current_slot += rsv_slots

    # Second pass: reuse leftover slots for slices still under SLA
    needy_slices = [
        s for s, apps in slice_apps.items()
        if any(alloc_thr[(i, a, s)] < get_sla(a) for (i, a) in apps)
    ]
    needy_rr = {
        s: {'apps': slice_apps[s], 'idx': 0}
        for s in needy_slices
    }

    for t in leftover_slots:
        for k in range(K):
            for s in needy_slices:
                apps = needy_rr[s]['apps']
                if not apps:
                    continue
                idx = needy_rr[s]['idx']
                i_app, a_app = apps[idx]
                needy_rr[s]['idx'] = (idx + 1) % len(apps)
                key = (i_app, a_app, s, k)
                if key in gamma:
                    thr = gamma[key]
                    prb_assignments.append((t, k, i_app, a_app, s, thr))
                    alloc_thr[(i_app, a_app, s)] += thr
                    break
            else:
                # slot remains unused
                continue
            break

    # Build allocation DataFrame
    allocation_df = pd.DataFrame(
        prb_assignments,
        columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput']
    )
    slice_throughput_df = (
        allocation_df
          .groupby('Slice')['Throughput']
          .sum()
          .reset_index()
          .rename(columns={'Throughput': 'Total_Throughput'})
    )
    allocation_df['App_Key'] = allocation_df.apply(
        lambda row: (row['User'], row['App'], row['Slice']), axis=1
    )

    # Total throughput per (user, app, slice)
    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput']
        .sum().reset_index()
    )
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(),
        index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    # DataFrame of all combos for zero-fill
    all_combos_df = pd.DataFrame(APPKEY_LIST, columns=['User', 'App', 'Slice'])
    merged_df = all_combos_df.merge(
        app_throughput_df, on=['User', 'App', 'Slice'], how='left'
    ).fillna({'Throughput': 0.0})

    # Average throughput per App
    app_episode_thr = (
        merged_df.groupby('App')['Throughput']
        .mean().reset_index()
        .rename(columns={'Throughput': 'Average_Throughput'})
    )

    # SLA violation metrics
    violation_counts_per_slice = {s: 0 for s in slot_reservation}
    violation_gap_per_slice = {s: 0.0 for s in slot_reservation}

    for (i_, a_, s_) in APPKEY_LIST:
        sla_needed = get_sla(a_)
        allocated = app_throughput_df.loc[
            (app_throughput_df['User']==i_) &
            (app_throughput_df['App']==a_) &
            (app_throughput_df['Slice']==s_),
            'Throughput'
        ]
        allocated_thr = float(allocated.iloc[0]) if not allocated.empty else 0.0
        if allocated_thr < sla_needed:
            violation_counts_per_slice[s_] += 1
            violation_gap_per_slice[s_] += (1.0 - allocated_thr/sla_needed)

    for s in violation_gap_per_slice:
        if violation_counts_per_slice[s]:
            violation_gap_per_slice[s] /= violation_counts_per_slice[s]

    total_violations = sum(violation_counts_per_slice.values())
    total_throughput = allocation_df['Throughput'].sum()

    return (
        allocation_df, total_throughput, app_throughput_df,
        app_episode_thr, violation_counts_per_slice,
        total_violations, violation_gap_per_slice, slice_throughput_df
    )


def prbs_used_per_slice(alloc_df: pd.DataFrame) -> dict[int,int]:
    """
    Count unique (Slot,PRB) pairs used *per* slice.
    """
    uniq = alloc_df[['Slot','PRB','Slice']].drop_duplicates()
    return uniq.groupby('Slice').size().to_dict()


def slice_spectral_efficiency(alloc_df: pd.DataFrame) -> dict[int,float]:
    """
    SE[s] = total throughput in slice s  /  (# PRBs used by slice s)
    Returns a dict mapping slice→SE.
    """
    # total throughput per slice
    thr = alloc_df.groupby('Slice')['Throughput'].sum().to_dict()
    # PRB counts per slice
    prb_counts = prbs_used_per_slice(alloc_df)
    # build SE dict
    return {s: (thr.get(s,0) / prb_counts.get(s,1)) 
            for s in set(list(thr)+list(prb_counts))}


@lru_cache(maxsize=None)
def _load_kbl_modules():
    base = Path(__file__).resolve().parent / "network-slicing"
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    kbrl = _load("kbrl_control", base / "kbrl_control.py")
    kernel = _load("kbrl_kernel", base / "algorithms" / "kernel.py")
    projectron = _load("kbrl_projectron", base / "algorithms" / "projectron.py")
    return kbrl, kernel, projectron


def run_kbl_allocation(gamma, accuracy_range=(0.99, 0.999), alfa=0.05, drift_fallback=0.5):
    """
    KBL (KBRL) allocation using the kernel-based RL controller from the repo.
    We adapt it to the current (gamma, SLA) setting by:
      - building a per-slice state vector
      - using KBRL to choose PRB counts per slice
      - allocating PRBs within each slice by best available gamma
    """
    kbrl, kernel_mod, proj_mod = _load_kbl_modules()
    Learner = kbrl.Learner
    KBRL_Control = kbrl.KBRL_Control
    GaussianKernel = kernel_mod.GaussianKernel
    SVvariable = proj_mod.SVvariable
    Projectron = proj_mod.Projectron

    n_slices = 3
    n_prbs = K
    state_dim = 4  # per slice

    learners = []
    for s in range(n_slices):
        sv = SVvariable()
        kernel = GaussianKernel(sv, 1)
        algorithm = Projectron(kernel)
        initial_action = max(1, n_prbs // n_slices)
        security_factor = 2
        indexes = slice(s * state_dim, (s + 1) * state_dim)
        learners.append(Learner(algorithm, indexes, initial_action, security_factor))

    kbl = KBRL_Control(learners, n_prbs, alfa=alfa, accuracy_range=accuracy_range)

    # Precompute per-slice appkeys and total SLA
    slice_appkeys = {s: [ak for ak in APPKEY_LIST if ak[2] == s] for s in [1, 2, 3]}
    slice_total_sla = {
        s: sum(get_sla(ak[1]) for ak in slice_appkeys[s]) for s in [1, 2, 3]
    }
    max_gamma = max(gamma.values()) if gamma else 1.0

    allocation_records = []
    allocation_so_far = {ak: 0.0 for ak in APPKEY_LIST}
    slack = {ak: get_sla(ak[1]) for ak in APPKEY_LIST}
    prev_prb_used = {1: 0, 2: 0, 3: 0}

    # Precompute best thr per slice/prb for state feature
    best_thr_by_slice_prb = {s: [0.0] * K for s in [1, 2, 3]}
    for (u, a, s, prb), thr in gamma.items():
        if thr > best_thr_by_slice_prb[s][prb]:
            best_thr_by_slice_prb[s][prb] = thr

    for t in range(T):
        # Build KBL state
        state = []
        for s in [1, 2, 3]:
            total_sla = slice_total_sla[s] or 1.0
            allocated = sum(allocation_so_far[ak] for ak in slice_appkeys[s])
            remaining = max(total_sla - allocated, 0.0)
            remaining_ratio = remaining / total_sla
            cumulative_ratio = allocated / total_sla
            avg_best_thr = sum(best_thr_by_slice_prb[s]) / K
            avg_best_thr_norm = avg_best_thr / max_gamma if max_gamma > 0 else 0.0
            prev_prb_ratio = prev_prb_used[s] / K
            state.extend([remaining_ratio, avg_best_thr_norm, prev_prb_ratio, cumulative_ratio])

        state = np.array(state, dtype=np.float32)
        action, _ = kbl.select_action(state)

        # Greedy PRB allocation within slice quotas
        slice_quota = {1: int(action[0]), 2: int(action[1]), 3: int(action[2])}
        slice_thr_slot = {1: 0.0, 2: 0.0, 3: 0.0}
        used_prbs = {1: 0, 2: 0, 3: 0}

        for prb in range(K):
            if sum(slice_quota.values()) <= 0:
                break

            candidates = []
            for s in [1, 2, 3]:
                if slice_quota[s] <= 0:
                    continue
                best = None
                best_thr = -1.0
                for ak in slice_appkeys[s]:
                    thr = gamma.get((ak[0], ak[1], ak[2], prb), 0.0)
                    if thr <= 0:
                        continue
                    if slack[ak] > 0 and thr > best_thr:
                        best_thr = thr
                        best = ak
                if best is not None:
                    candidates.append((best_thr, s, best, 1.0))

            if not candidates:
                # fallback: allow allocation even if SLA already met
                for s in [1, 2, 3]:
                    if slice_quota[s] <= 0:
                        continue
                    best = None
                    best_thr = -1.0
                    for ak in slice_appkeys[s]:
                        thr = gamma.get((ak[0], ak[1], ak[2], prb), 0.0)
                        if thr > best_thr:
                            best_thr = thr
                            best = ak
                    if best is not None and best_thr > 0:
                        candidates.append((best_thr, s, best, drift_fallback))

            if not candidates:
                continue

            best_thr, s_sel, ak_sel, drift = max(candidates, key=lambda x: x[0])
            thr_used = best_thr * drift
            allocation_records.append((t, prb, ak_sel[0], ak_sel[1], ak_sel[2], thr_used))
            allocation_so_far[ak_sel] += thr_used
            slack[ak_sel] = max(slack[ak_sel] - thr_used, 0.0)
            slice_thr_slot[s_sel] += thr_used
            slice_quota[s_sel] -= 1
            used_prbs[s_sel] += 1

        # Update KBL with per-slice SLA labels for this slot
        SLA_labels = np.array([
            1 if slice_thr_slot[s] >= (slice_total_sla[s] / T) else -1
            for s in [1, 2, 3]
        ], dtype=np.int16)
        kbl.update_control(state, action, SLA_labels)
        prev_prb_used = used_prbs

    allocation_df = pd.DataFrame(
        allocation_records,
        columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput']
    )

    slice_throughput_df = (
        allocation_df.groupby('Slice')['Throughput']
        .sum().reset_index()
        .rename(columns={'Throughput': 'Total_Throughput'})
    )

    allocation_df['App_Key'] = allocation_df.apply(
        lambda r: (r['User'], r['App'], r['Slice']), axis=1
    )

    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput']
        .sum().reset_index()
    )
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    all_combos_df = pd.DataFrame(APPKEY_LIST, columns=['User', 'App', 'Slice'])
    merged_df = all_combos_df.merge(
        app_throughput_df, on=['User', 'App', 'Slice'], how='left'
    ).fillna({'Throughput': 0.0})

    app_episode_thr = (
        merged_df.groupby('App')['Throughput']
        .mean().reset_index()
        .rename(columns={'Throughput': 'Average_Throughput'})
    )

    violation_counts_per_slice = {1: 0, 2: 0, 3: 0}
    violation_gap_per_slice = {1: 0.0, 2: 0.0, 3: 0.0}

    for (i_, a_, s_) in APPKEY_LIST:
        sla_needed = get_sla(a_)
        allocated = merged_df.loc[
            (merged_df['User'] == i_) &
            (merged_df['App'] == a_) &
            (merged_df['Slice'] == s_),
            'Throughput'
        ].iloc[0]
        ratio = allocated / sla_needed if sla_needed > 0 else 1.0
        if ratio < 1:
            violation_counts_per_slice[s_] += 1
            violation_gap_per_slice[s_] += (1.0 - ratio)

    for s_ in [1, 2, 3]:
        if violation_counts_per_slice[s_] > 0:
            violation_gap_per_slice[s_] /= violation_counts_per_slice[s_]
        else:
            violation_gap_per_slice[s_] = 0.0

    total_violations = sum(violation_counts_per_slice.values())
    total_throughput = allocation_df['Throughput'].sum()

    return (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )


class XSliceActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=64, n_slices=3):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.alpha_head = nn.Linear(hidden, n_slices)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        alpha = torch.nn.functional.softplus(self.alpha_head(x)) + 1e-3
        value = self.v_head(x).squeeze(-1)
        return alpha, value


def _build_xslice_state(gamma, allocation_so_far, prev_prb_used):
    slice_appkeys = {s: [ak for ak in APPKEY_LIST if ak[2] == s] for s in [1, 2, 3]}
    slice_total_sla = {
        s: sum(get_sla(ak[1]) for ak in slice_appkeys[s]) for s in [1, 2, 3]
    }
    max_gamma = max(gamma.values()) if gamma else 1.0
    best_thr_by_slice_prb = {s: [0.0] * K for s in [1, 2, 3]}
    for (u, a, s, prb), thr in gamma.items():
        if thr > best_thr_by_slice_prb[s][prb]:
            best_thr_by_slice_prb[s][prb] = thr

    state = []
    for s in [1, 2, 3]:
        total_sla = slice_total_sla[s] or 1.0
        allocated = sum(allocation_so_far[ak] for ak in slice_appkeys[s])
        remaining = max(total_sla - allocated, 0.0)
        remaining_ratio = remaining / total_sla
        cumulative_ratio = allocated / total_sla
        avg_best_thr = sum(best_thr_by_slice_prb[s]) / K
        avg_best_thr_norm = avg_best_thr / max_gamma if max_gamma > 0 else 0.0
        prev_prb_ratio = prev_prb_used[s] / K
        state.extend([remaining_ratio, avg_best_thr_norm, prev_prb_ratio, cumulative_ratio])
    return np.array(state, dtype=np.float32)


def _xslice_allocate_slot(gamma, slice_quota, slack, allocation_so_far, t, drift_fallback=0.5):
    slice_appkeys = {s: [ak for ak in APPKEY_LIST if ak[2] == s] for s in [1, 2, 3]}
    allocation_records = []
    slice_thr_slot = {1: 0.0, 2: 0.0, 3: 0.0}
    used_prbs = {1: 0, 2: 0, 3: 0}

    for prb in range(K):
        if sum(slice_quota.values()) <= 0:
            break

        candidates = []
        for s in [1, 2, 3]:
            if slice_quota[s] <= 0:
                continue
            best = None
            best_thr = -1.0
            for ak in slice_appkeys[s]:
                thr = gamma.get((ak[0], ak[1], ak[2], prb), 0.0)
                if thr <= 0:
                    continue
                if slack[ak] > 0 and thr > best_thr:
                    best_thr = thr
                    best = ak
            if best is not None:
                candidates.append((best_thr, s, best, 1.0))

        if not candidates:
            for s in [1, 2, 3]:
                if slice_quota[s] <= 0:
                    continue
                best = None
                best_thr = -1.0
                for ak in slice_appkeys[s]:
                    thr = gamma.get((ak[0], ak[1], ak[2], prb), 0.0)
                    if thr > best_thr:
                        best_thr = thr
                        best = ak
                if best is not None and best_thr > 0:
                    candidates.append((best_thr, s, best, drift_fallback))

        if not candidates:
            continue

        best_thr, s_sel, ak_sel, drift = max(candidates, key=lambda x: x[0])
        thr_used = best_thr * drift
        allocation_records.append((t, prb, ak_sel[0], ak_sel[1], ak_sel[2], thr_used))
        allocation_so_far[ak_sel] += thr_used
        slack[ak_sel] = max(slack[ak_sel] - thr_used, 0.0)
        slice_thr_slot[s_sel] += thr_used
        slice_quota[s_sel] -= 1
        used_prbs[s_sel] += 1

    return allocation_records, slice_thr_slot, used_prbs


def run_xslice_allocation(
    gamma,
    model_path="xslice_policy.pth",
    train_if_missing=True,
    train_episodes=30,
    ppo_epochs=4,
    lr=3e-4,
    gamma_rl=0.95,
    clip_eps=0.2,
    drift_fallback=0.5,
):
    obs_dim = 12
    model = XSliceActorCritic(obs_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if train_if_missing and not os.path.isfile(model_path):
        for _ in range(train_episodes):
            allocation_so_far = {ak: 0.0 for ak in APPKEY_LIST}
            slack = {ak: get_sla(ak[1]) for ak in APPKEY_LIST}
            prev_prb_used = {1: 0, 2: 0, 3: 0}
            log_probs, values, rewards = [], [], []

            for t in range(T):
                state = _build_xslice_state(gamma, allocation_so_far, prev_prb_used)
                st = torch.from_numpy(state).float().unsqueeze(0)
                alpha, value = model(st)
                dist = Dirichlet(alpha.squeeze(0))
                proportions = dist.sample()
                log_prob = dist.log_prob(proportions)
                action = proportions.detach().cpu().numpy()
                quota = {
                    1: int(np.round(action[0] * K)),
                    2: int(np.round(action[1] * K)),
                    3: int(np.round(action[2] * K)),
                }
                # ensure total K
                diff = K - sum(quota.values())
                if diff != 0:
                    quota[1] += diff
                alloc_records, slice_thr_slot, used_prbs = _xslice_allocate_slot(
                    gamma, quota, slack, allocation_so_far, t, drift_fallback=drift_fallback
                )

                slice_total_sla = {
                    s: sum(get_sla(ak[1]) for ak in APPKEY_LIST if ak[2] == s) for s in [1, 2, 3]
                }
                slot_target = {s: slice_total_sla[s] / T for s in [1, 2, 3]}
                violations = sum(1 for s in [1, 2, 3] if slice_thr_slot[s] < slot_target[s])
                reward = (sum(slice_thr_slot.values()) / 1000.0) - 5.0 * violations

                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                rewards.append(reward)
                prev_prb_used = used_prbs

            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + gamma_rl * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            values_t = torch.stack(values)
            logp_t = torch.stack(log_probs)

            advantages = returns - values_t.detach()
            policy_loss = -(logp_t * advantages).mean()
            value_loss = nn.functional.mse_loss(values_t, returns)
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save({'model': model.state_dict()}, model_path)

    if os.path.isfile(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt['model'])
    model.eval()

    allocation_records = []
    allocation_so_far = {ak: 0.0 for ak in APPKEY_LIST}
    slack = {ak: get_sla(ak[1]) for ak in APPKEY_LIST}
    prev_prb_used = {1: 0, 2: 0, 3: 0}

    for t in range(T):
        state = _build_xslice_state(gamma, allocation_so_far, prev_prb_used)
        st = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            alpha, _ = model(st)
            proportions = (alpha / alpha.sum()).squeeze(0).cpu().numpy()
        quota = {
            1: int(np.round(proportions[0] * K)),
            2: int(np.round(proportions[1] * K)),
            3: int(np.round(proportions[2] * K)),
        }
        diff = K - sum(quota.values())
        if diff != 0:
            quota[1] += diff
        alloc_records, slice_thr_slot, used_prbs = _xslice_allocate_slot(
            gamma, quota, slack, allocation_so_far, t, drift_fallback=drift_fallback
        )
        allocation_records.extend(alloc_records)
        prev_prb_used = used_prbs

    allocation_df = pd.DataFrame(
        allocation_records,
        columns=['Slot', 'PRB', 'User', 'App', 'Slice', 'Throughput']
    )
    slice_throughput_df = (
        allocation_df.groupby('Slice')['Throughput']
        .sum().reset_index()
        .rename(columns={'Throughput': 'Total_Throughput'})
    )
    allocation_df['App_Key'] = allocation_df.apply(
        lambda r: (r['User'], r['App'], r['Slice']), axis=1
    )
    app_throughput_df = (
        allocation_df.groupby('App_Key')['Throughput']
        .sum().reset_index()
    )
    app_throughput_df[['User', 'App', 'Slice']] = pd.DataFrame(
        app_throughput_df['App_Key'].tolist(), index=app_throughput_df.index
    )
    app_throughput_df.drop(columns=['App_Key'], inplace=True)

    all_combos_df = pd.DataFrame(APPKEY_LIST, columns=['User', 'App', 'Slice'])
    merged_df = all_combos_df.merge(
        app_throughput_df, on=['User', 'App', 'Slice'], how='left'
    ).fillna({'Throughput': 0.0})

    app_episode_thr = (
        merged_df.groupby('App')['Throughput']
        .mean().reset_index()
        .rename(columns={'Throughput': 'Average_Throughput'})
    )

    violation_counts_per_slice = {1: 0, 2: 0, 3: 0}
    violation_gap_per_slice = {1: 0.0, 2: 0.0, 3: 0.0}
    for (i_, a_, s_) in APPKEY_LIST:
        sla_needed = get_sla(a_)
        allocated = merged_df.loc[
            (merged_df['User'] == i_) &
            (merged_df['App'] == a_) &
            (merged_df['Slice'] == s_),
            'Throughput'
        ].iloc[0]
        ratio = allocated / sla_needed if sla_needed > 0 else 1.0
        if ratio < 1:
            violation_counts_per_slice[s_] += 1
            violation_gap_per_slice[s_] += (1.0 - ratio)

    for s_ in [1, 2, 3]:
        if violation_counts_per_slice[s_] > 0:
            violation_gap_per_slice[s_] /= violation_counts_per_slice[s_]
        else:
            violation_gap_per_slice[s_] = 0.0

    total_violations = sum(violation_counts_per_slice.values())
    total_throughput = allocation_df['Throughput'].sum()

    return (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )

#--------------------PLOTS-------------------------------------------

def plot_training_metrics(training_results_df):
    
    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0}) 

    # Plot Episode Reward
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Episode_Reward', data=training_results_df, marker='o', color='purple')
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Training: Episode Rewards')
    plt.tight_layout()
    plt.show()

    # Plot Total Throughput
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Episode', y='Total_Throughput_KB', data=training_results_df, marker='o', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Total Throughput (KB)')
    plt.title('Training: Total Throughput per Episode')
    plt.tight_layout()
    plt.show()

def plot_all_results(results_df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    sns.set(style="whitegrid")
    palette = {
        'HSRS':       '#b185d4',
        'RadioSaber': '#407552',
        'BestCQI':    '#A5D6A7',
        'NVS':        '#8f9394',
        'KBL':        '#000000',
        'XSlice':     '#1f77b4',
    }

    # --- consistent algorithm order + numbering -------
    methods = [
        'HSRS',
        'RadioSaber',
        'BestCQI',
        'NVS',
        'KBL',
        'XSlice',
    ]
    idx_map = {m: i+1 for i, m in enumerate(methods)}

    # --- 1) Total throughput ----------------------------
    min_datarate = results_df['Min_required_datarate_KB'].iloc[0]
    plt.figure(figsize=(10, 6))
    for lbl, col, marker in [
        ('HSRS',        'HSRS_Total_Throughput',        'x'),
        ('RadioSaber',  'RadioSaber_Total_Throughput',  '<'),
        ('BestCQI',     'BestCQI_Total_Throughput',     's'),
        ('NVS',         'NVS_Total_Throughput',         '^'),
        ('KBL',         'KBL_Total_Throughput',         'o'),
        ('XSlice',      'XSlice_Total_Throughput',      'd'),
    ]:
        plt.plot(results_df['Simulation'], results_df[col],
                 marker=marker, label=lbl, color=palette[lbl])
    plt.axhline(y=min_datarate, color='r', linestyle='--')
    plt.text(results_df['Simulation'].min(), min_datarate,
             f'Min total: {min_datarate} KB',
             color='r', fontsize=16, ha='left')
    plt.xlabel('Allocation Episodes', fontsize=22)
    plt.ylabel('Throughput (Kbit/frame)', fontsize=22)
    plt.title('Throughput per Episodes', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=20)
    plt.tight_layout()
    plt.show()

    # --- 2) PRB-usage ----------------------------------------
    plt.figure(figsize=(10, 6))
    for lbl, col, marker in [
        ('HSRS',        'HSRS_PRB_Usage (%)',        'x'),
        ('RadioSaber',  'RadioSaber_PRB_Usage (%)',  '<'),
        ('BestCQI',     'BestCQI_PRB_Usage (%)',     's'),
        ('NVS',         'NVS_PRB_Usage (%)',         '^'),
        ('KBL',         'KBL_PRB_Usage (%)',         'o'),
        ('XSlice',      'XSlice_PRB_Usage (%)',      'd'),
    ]:
        plt.plot(results_df['Simulation'], results_df[col],
                 marker=marker, label=lbl, color=palette[lbl])
    plt.xlabel('Allocation Episodes', fontsize=18)
    plt.ylabel('PRB Usage (%)', fontsize=18)
    plt.title('PRB Usage Rate', fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- 3) Spectral Efficiency per Episode --------------------
    plt.figure(figsize=(10, 6))
    total_prbs = 250
    for lbl, thr_col, usage_col, marker in [
        ('HSRS',        'HSRS_Total_Throughput',        'HSRS_PRB_Usage (%)',        'x'),
        ('RadioSaber',  'RadioSaber_Total_Throughput',  'RadioSaber_PRB_Usage (%)',  '<'),
        ('BestCQI',     'BestCQI_Total_Throughput',     'BestCQI_PRB_Usage (%)',     's'),
        ('NVS',         'NVS_Total_Throughput',         'NVS_PRB_Usage (%)',         '^'),
        ('KBL',         'KBL_Total_Throughput',         'KBL_PRB_Usage (%)',         'o'),
        ('XSlice',      'XSlice_Total_Throughput',      'XSlice_PRB_Usage (%)',      'd'),
    ]:
        used_prbs = (results_df[usage_col]/100) * total_prbs
        se = results_df[thr_col] / used_prbs
        plt.plot(results_df['Simulation'], se,
                 marker=marker, label=lbl, color=palette[lbl])
    plt.xlabel('Allocation Episodes', fontsize=18)
    plt.ylabel('Spectral Efficiency\n(kbit per PRB)', fontsize=18)
    plt.title('Spectral Efficiency per Episode', fontsize=22)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=14)
    plt.tight_layout()
    plt.show()

    # --- helper maps -----------------------------------------
    slice_mapping = {1: 'Critical Slice', 2: 'Performance Slice', 3: 'Business Slice'}
    slice_nums = [1, 2, 3]
    # --- 4) Violation Counts per Slice ----------------------
    viol_data = []
    for _, row in results_df.iterrows():
        for method, pref in [
            ('HSRS',       'HSRS'),
            ('RadioSaber', 'RadioSaber'),
            ('BestCQI',    'BestCQI'),
            ('NVS',        'NVS'),
            ('KBL',        'KBL'),
            ('XSlice',     'XSlice'),
        ]:
            for s in slice_nums:
                viol_data.append({
                    'Method':          method,
                    'Slice':           slice_mapping[s],
                    'Violation_Count': row.get(f'{pref}_Violations_S{s}', 0)
                })
    viol_df = pd.DataFrame(viol_data)
    # enforce your desired order
    viol_df['Method'] = pd.Categorical(
        viol_df['Method'], categories=methods, ordered=True
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=viol_df, x='Slice', y='Violation_Count',
        hue='Method', hue_order=methods,
        palette=palette
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)

    # rebuild legend with numbers
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=14)

    plt.title('Average Violation Occurrence per Slice', fontsize=22)
    plt.xlabel('Slice', fontsize=18)
    plt.ylabel('Violations per Episode', fontsize=18)
    plt.tight_layout()
    plt.show()

    # --- 5) Violation Rates per Slice -----------------------
    rate_data = []
    for _, row in results_df.iterrows():
        for method, pref in [
            ('HSRS',       'HSRS'),
            ('RadioSaber', 'RadioSaber'),
            ('BestCQI',    'BestCQI'),
            ('NVS',        'NVS'),
            ('KBL',        'KBL'),
            ('XSlice',     'XSlice'),
        ]:
            for s in slice_nums:
                rate_data.append({
                    'Method':             method,
                    'Slice':              slice_mapping[s],
                    'Violation_Rate (%)': row.get(f'{pref}_Violation_Rate_S{s}', 0)
                })
    rate_df = pd.DataFrame(rate_data)
    # enforce order
    rate_df['Method'] = pd.Categorical(
        rate_df['Method'], categories=methods, ordered=True
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=rate_df,
        x='Slice', y='Violation_Rate (%)',
        hue='Method', hue_order=methods,
        palette=palette
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)

    # rebuild legend with numbers
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=14)

    plt.xlabel('')
    plt.ylabel('Violation Rate (%)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()


    # --- 6) Violation Gap boxplot ---------------------------
    gap_data = []
    for _, row in results_df.iterrows():
        for method, pref in [
            ('HSRS',       'HSRS'),
            ('RadioSaber', 'RadioSaber'),
            ('BestCQI',    'BestCQI'),
            ('NVS',        'NVS'),
            ('KBL',        'KBL'),
            ('XSlice',     'XSlice'),
        ]:
            for s in slice_nums:
                gap_data.append({
                    'Method':       method,
                    'Slice':        slice_mapping[s],
                    'Violation_gap': row.get(f'{pref}_Violation_gap_S{s}', 0) * 100
                })
    gap_df = pd.DataFrame(gap_data)
    # enforce order
    gap_df['Method'] = pd.Categorical(
        gap_df['Method'], categories=methods, ordered=True
    )

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=gap_df,
        x='Slice', y='Violation_gap',
        hue='Method', hue_order=methods,
        palette=palette,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)

    # rebuild legend with numbers
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=14)

    plt.ylabel('Violation Gap (%)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # --- 7) Per-Application Average Throughput -------------
    cols_map = {
        m: [c for c in results_df.columns if c.startswith(f"{m.replace(' ', '_')}_App_")] for m in methods
    }
    avg = {m: results_df[cols_map[m]].mean().to_dict() for m in methods}
    achieved = {
        'Application': app_labels,
        'SLA':         sla_values,
    }
    for m in methods:
        achieved[m] = [avg[m].get(f"{m.replace(' ', '_')}_App_{i+1}_Throughput", 0)
                        for i in range(len(app_labels))]
    bar_df = pd.DataFrame(achieved).melt(
        id_vars=['Application', 'SLA'], var_name='Method', value_name='Throughput'
    )
    bar_df['Method'] = pd.Categorical(bar_df['Method'], categories=methods, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=bar_df, x='Application', y='Throughput', hue='Method',
                     hue_order=methods, palette=palette)
    for x in [1.5, 3.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
    y_max = bar_df['Throughput'].max()
    ax.text(0.5, y_max*1.05, 'Critical slice', ha='center', va='bottom', fontsize=20)
    ax.text(2.5, y_max*1.05, 'Performance slice', ha='center', va='bottom', fontsize=20)
    ax.text(4.0, y_max*1.05, 'Business slice', ha='center', va='bottom', fontsize=20)
    # SLA lines
    bar_width = 0.25
    app_positions = np.arange(len(app_labels))
    for i, sla in enumerate(sla_values):
        x_start = app_positions[i] - (1.5 * bar_width)
        x_end   = app_positions[i] + (1.5 * bar_width)
        plt.plot([x_start, x_end], [sla, sla], color='red', linestyle='--', linewidth=2)
        plt.text(i, sla + y_max * 0.01, f"SLA: {sla}", color='red', ha='right', fontsize=11)
    plt.ylabel('Average throughput (Bits/frame)', fontsize=20)
    plt.xlabel('Applications', fontsize=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=18)
    plt.tight_layout()
    plt.show()

    # --- 8) Radar: Avg Total Throughput per Slice per Method ----
    plt.figure(figsize=(10, 6))
    slice_labels = ['Critical Slice', 'Performance Slice', 'Business Slice']
    slice_nums = [1, 2, 3]
    angles = np.linspace(0, 2 * np.pi, len(slice_labels), endpoint=False).tolist()
    angles += angles[:1]
    ax = plt.subplot(polar=True)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(slice_labels, fontsize=14)
    ax.set_rlabel_position(0)
    ax.tick_params(axis='y', labelsize=12)
    methods_radar = [
        ('HSRS',       'HSRS',       dict(color=palette['HSRS'],       linestyle='-.', marker='x')),
        ('RadioSaber', 'RadioSaber', dict(color=palette['RadioSaber'], linestyle=':',  marker='<')),
        ('BestCQI',    'BestCQI',    dict(color=palette['BestCQI'],    linestyle='--', marker='s')),
        ('NVS',        'NVS',        dict(color=palette['NVS'],        linestyle=':',  marker='^')),
        ('KBL',        'KBL',        dict(color=palette['KBL'],        linestyle='-',  marker='o')),
        ('XSlice',     'XSlice',     dict(color=palette['XSlice'],     linestyle='-',  marker='d')),
    ]
    for name, prefix, style in methods_radar:
        vals = [results_df[f'{prefix}_S{s}_Throughput'].mean() for s in slice_nums]
        vals += vals[:1]
        ax.plot(angles, vals, label=f"{idx_map[name]}. {name}", **style)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title('Average Total Throughput per Slice by Method', fontsize=18)
    plt.tight_layout()
    plt.show()

    # --- 9) Average Spectral Efficiency per Slice (Stacked Bar) ----
    methods_se = [
        ('HSRS',       'HSRS',       palette['HSRS']),
        ('RadioSaber', 'RadioSaber', palette['RadioSaber']),
        ('BestCQI',    'BestCQI',    palette['BestCQI']),
        ('NVS',        'NVS',        palette['NVS']),
        ('KBL',        'KBL',        palette['KBL']),
        ('XSlice',     'XSlice',     palette['XSlice']),
    ]
    slice_labels = ['Critical', 'Performance', 'Business']
    slice_ids    = [1, 2, 3]
    data = np.array([[results_df[f'{pref}_SE_S{s}'].mean() for s in slice_ids]
                     for _, pref, _ in methods_se])
    plt.figure(figsize=(10,6))
    x = np.arange(len(methods_se))
    bottom = np.zeros(len(methods_se))
    bar_width = 0.6
    colors = ['orange', 'cyan', '#A5D6A7']
    for i, sl in enumerate(slice_labels):
        vals = data[:, i]
        plt.bar(x, vals, bar_width, bottom=bottom, label=sl, color=colors[i])
        for j, v in enumerate(vals):
            if v > 0:
                plt.text(x[j], bottom[j] + v/2, f'{v:.2f}', ha='center', va='center', fontsize=12)
        bottom += vals
    plt.xticks(x, [m[0] for m in methods_se], rotation=0, fontsize=12)
    plt.ylabel('Avg Spectral Efficiency\n(Bit per RB)', fontsize=14)
    plt.title('Average Per-Slice Spectral Efficiency', fontsize=16)
    plt.legend(title='Slice', fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
def print_final_results(results_df):

    summary_cols = [
        'Simulation',
        'HSRS_Total_Throughput',
        'RadioSaber_Total_Throughput',
        'BestCQI_Total_Throughput',
        'NVS_Total_Throughput',
        'KBL_Total_Throughput',
        'XSlice_Total_Throughput',

        'HSRS_PRB_Usage (%)',
        'RadioSaber_PRB_Usage (%)',
        'BestCQI_PRB_Usage (%)',
        'NVS_PRB_Usage (%)',
        'KBL_PRB_Usage (%)',
        'XSlice_PRB_Usage (%)',

        'HSRS_Time_Taken',
        'RadioSaber_Time_Taken',
        'BestCQI_Time_Taken',
        'NVS_Time_Taken',
        'KBL_Time_Taken',
        'XSlice_Time_Taken',
    ]

    app_cols_my  = [c for c in results_df.columns if c.startswith("HSRS_App_")]
    app_cols_rad = [c for c in results_df.columns if c.startswith("RadioSaber_App_")]
    app_cols_cqi = [c for c in results_df.columns if c.startswith("BestCQI_App_")]
    app_cols_nvs = [c for c in results_df.columns if c.startswith("NVS_App_")]
    app_cols_kbl = [c for c in results_df.columns if c.startswith("KBL_App_")]
    app_cols_xs = [c for c in results_df.columns if c.startswith("XSlice_App_")]

    all_cols = summary_cols + app_cols_my + app_cols_rad + app_cols_cqi + app_cols_nvs + app_cols_kbl + app_cols_xs
    print("\n=== Final Results Overview ===")
    print(results_df[all_cols].to_string(index=False))

def plot_compare_training(df_dqn, df_ppo):
    """
    Compare training metrics between ADSS_DQN and ADSS_PPO.
    
    Parameters
    ----------
    df_dqn : pandas.DataFrame
        Training results for ADSS_DQN, with columns ['Episode', 'Episode_Reward', 'Total_Throughput_KB'].
    df_ppo : pandas.DataFrame
        Training results for ADSS_PPO, same column structure as df_dqn.
    """
    sns.set(style="whitegrid")
    plt.rcParams.update({'figure.max_open_warning': 0})

    # 1) Episode Reward comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x='Episode',
        y='Episode_Reward',
        data=df_dqn,
        marker='o',
        color='purple',
        label='ADSS_DQN'
    )
    sns.lineplot(
        x='Episode',
        y='Episode_Reward',
        data=df_ppo,
        marker='s',
        color='orangered',
        label='ADSS_PPO'
    )
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Episode Reward', fontsize=14)
    plt.title('Training: Episode Reward Comparison', fontsize=16)
    plt.legend(title='Method', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 2) Total Throughput comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x='Episode',
        y='Total_Throughput_KB',
        data=df_dqn,
        marker='o',
        color='purple',
        label='ADSS_DQN'
    )
    sns.lineplot(
        x='Episode',
        y='Total_Throughput_KB',
        data=df_ppo,
        marker='s',
        color='orangered',
        label='ADSS_PPO'
    )
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Throughput (KB)', fontsize=14)
    plt.title('Training: Total Throughput per Episode Comparison', fontsize=16)
    plt.legend(title='Method', fontsize=12)
    plt.tight_layout()
    plt.show()
