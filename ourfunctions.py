import random
import numpy as np
import pandas as pd
import time 
from gurobipy import Model, GRB, quicksum, Env
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
import re
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

APP_PACKET_PROFILE = {
    1: {"payload_bits": 400.0,  "pdb_ms": 5.0},   # ETCS
    2: {"payload_bits": 900.0,  "pdb_ms": 5.0},   # Voice
    3: {"payload_bits": 1000.0, "pdb_ms": 10.0},  # CCTV
    4: {"payload_bits": 20.0,   "pdb_ms": 10.0},  # PIS
    5: {"payload_bits": 1000.0, "pdb_ms": 10.0},  # Pwifi
}


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

DATA_DIR = "subband_cqi"            # folder that holds TOBA gateway subband CQI : UE0.csv … UE<n>.csv


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

_GRB_SILENT_ENV = None

def _get_silent_gurobi_env():
    global _GRB_SILENT_ENV
    if _GRB_SILENT_ENV is None:
        try:
            # Works on newer gurobipy versions and starts env silently.
            _GRB_SILENT_ENV = Env(params={"OutputFlag": 0})
        except TypeError:
            # Backward-compatible path for older gurobipy versions.
            env = Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            _GRB_SILENT_ENV = env
    return _GRB_SILENT_ENV

def run_allocation_solver(gamma, wc=0.01, wp=0.6, wb=1, return_decision_time=False):
    """
    Optimal ILP (Gurobi) allocation 
    """

    decision_t0 = time.perf_counter()

    # Sort TOBA gatexyas (UE)  ids per slices
    crit_users = sorted({i for (i, a, s, _) in gamma if s == 1})  
    perf_users = sorted({i for (i, a, s, _) in gamma if s == 2})  
    busi_users = sorted({i for (i, a, s, _) in gamma if s == 3})  

    U_c, U_p, U_b = len(crit_users), len(perf_users), len(busi_users)

    # ------------------------------------------------------------
    # 1)  Model & variables 
    # ------------------------------------------------------------
    model = Model("Optimal_ILP", env=_get_silent_gurobi_env())
    model.Params.NodefileStart = 0.010
    model.Params.NodefileDir = "/tmp"
    model.Params.Threads = 16
    model.Params.MIPFocus = 1
    model.Params.ConcurrentMIP = 4
    model.Params.TimeLimit = 10

    x = model.addVars(U_c, len(Ac), K, T, vtype=GRB.BINARY, name="x")
    y = model.addVars(U_p, len(Ap), K, T, vtype=GRB.BINARY, name="y")
    z = model.addVars(U_b, len(Ab), K, T, vtype=GRB.BINARY, name="z")

    SlackCrit = model.addVars(U_c, len(Ac), vtype=GRB.CONTINUOUS, lb=0)
    SlackPerf = model.addVars(U_p, len(Ap), vtype=GRB.CONTINUOUS, lb=0)
    SlackBus  = model.addVars(U_b, len(Ab), vtype=GRB.CONTINUOUS, lb=0)
    # Delay slack for critical apps: unmet demand before app deadline.
    SlackDelayCrit = model.addVars(U_c, len(Ac), vtype=GRB.CONTINUOUS, lb=0)

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

    # 3-bis) Delay deadline constraints for critical slice only.
    # deadline_slot = ceil(pdb_ms / slot_ms), with slot_ms=1 ms and clamped in [1, T].
    slot_ms = 1.0
    delay_deadline_slot = {}
    for a in Ac:
        cfg = APP_PACKET_PROFILE.get(a, {})
        pdb_ms = float(cfg.get("pdb_ms", T * slot_ms))
        if not np.isfinite(pdb_ms) or pdb_ms <= 0:
            pdb_ms = T * slot_ms
        deadline_slot = int(math.ceil(pdb_ms / slot_ms))
        delay_deadline_slot[a] = max(1, min(T, deadline_slot))

    for i_loc in range(U_c):
        i_glob = crit_users[i_loc]
        for a_idx, a in enumerate(Ac):
            dslot = delay_deadline_slot[a]
            model.addConstr(
                quicksum(
                    x[i_loc, a_idx, k_, t_] * gamma.get((i_glob, a, 1, k_), 0)
                    for k_ in range(K) for t_ in range(dslot)
                ) + SlackDelayCrit[i_loc, a_idx] >= SLA_critic[a]
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
    alpha_delay_crit = 3000
    model.setObjective(
        wp * quicksum(y[i, a, k_, t] for i in range(U_p) for a in range(len(Ap)) for k_ in range(K) for t in range(T))
      + wb * quicksum(z[i, a, k_, t] for i in range(U_b) for a in range(len(Ab)) for k_ in range(K) for t in range(T))
      + wc * quicksum(x[i, a, k_, t] for i in range(U_c) for a in range(len(Ac)) for k_ in range(K) for t in range(T))
      + alpha_b * quicksum(SlackBus[i, a] / SLA_busi[Ab[a]] for i in range(U_b) for a in range(len(Ab)))
      + alpha_p * quicksum(SlackPerf[i, a] / SLA_perf[Ap[a]] for i in range(U_p) for a in range(len(Ap)))
      + alpha_delay_crit * quicksum(
            SlackDelayCrit[i, a] / max(SLA_critic[Ac[a]], 1e-9)
            for i in range(U_c) for a in range(len(Ac))
        ),
      GRB.MINIMIZE
    )

 
    # 5)  Solve

    try:
        model.optimize()
    except Exception as e:
        print(f"Gurobi optimization failed: {e}")
        return (None,)*9 if return_decision_time else (None,)*8

    if model.SolCount == 0:          # no feasible solution found at all
        print("No incumbent solution.")
        return (None,)*9 if return_decision_time else (None,)*8

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

    decision_time_s = time.perf_counter() - decision_t0

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

    out = (
        allocation_df, total_throughput, app_throughput_df,
        app_episode_thr, violation_counts_per_slice,
        total_violations, violation_gap_per_slice, slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out


def run_bestcqi_allocation(gamma, return_decision_time=False):
    # Assign each PRB in each time slot to (i, a, s) that has the highest achievable datarate in gamma

    # Initialize per-app cumulative allocation counter
    allocated_so_far = { key: 0.0 for key in APPKEY_LIST }
    drift = 0.8
    prb_assignments = []
    decision_t0 = time.perf_counter()

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
                    delta = best_thr * 0.4
                    prb_assignments.append((t, k_, i_sel, a_sel, s_sel, delta))
                    allocated_so_far[app_key] += delta
                else:
                    delta = best_thr
                    delta= delta * drift
                    # Record and update
                    prb_assignments.append((t, k_, i_sel, a_sel, s_sel, delta))
                    allocated_so_far[app_key] += delta

    decision_time_s = time.perf_counter() - decision_t0


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

    out = (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out


def run_myheuristic_allocation(gamma, return_decision_time=False):
    """
    Two-phase cascaded scheduler:
      1) Phase Critique : minimal-slack greedy
      2) Phase Performance & Business : weighted Best-CQI
    """
    w_perf= 0.5
    w_busi= 0.5
    drift = 0.9
    # Initialise slack bits for every app
    slack = {key: get_sla(key[1]) for key in APPKEY_LIST}
    prb_assignments = []
    decision_t0 = time.perf_counter()

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

    decision_time_s = time.perf_counter() - decision_t0

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

    out = (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice, slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out



def run_radiosaber_allocation(gamma, return_decision_time=False):
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
    decision_t0 = time.perf_counter()

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

    decision_time_s = time.perf_counter() - decision_t0

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

    out = (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out

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


def _split_episode_volume_into_jobs(episode_bits_expected: float, payload_bits: float) -> list[float]:
    """
    Split one app's episode demand into packet jobs by payload size.
    """
    if episode_bits_expected <= 0:
        return []
    payload = max(float(payload_bits), 1e-9)
    full_jobs = int(episode_bits_expected // payload)
    rem_bits = float(episode_bits_expected - (full_jobs * payload))
    jobs = [payload] * full_jobs
    if rem_bits > 1e-9:
        jobs.append(rem_bits)
    if not jobs:
        jobs = [float(episode_bits_expected)]
    return jobs


def compute_delay_violation_metrics(
    allocation_df: Optional[pd.DataFrame],
    app_packet_profile: Optional[dict] = None,
    slot_ms: float = 1.0,
    episode_slots: int = T,
    appkeys: Optional[list[tuple[int, int, int]]] = None,
):
    """
    Job-based delay metric computation (single episode, no carry-over between episodes).

    Rules:
      - Demand per (user, app, slice) is the SLA bits for this episode.
      - Demand is split into jobs using APP_PACKET_PROFILE payload_bits.
      - Jobs are served FIFO from slot throughput allocations of that app key.
      - Delay violation if completion time > app pdb_ms, or unfinished by episode end.
      - Unfinished jobs are dropped at end of episode.
    """
    slice_ids = [1, 2, 3]
    total_jobs_per_slice = {s: 0 for s in slice_ids}
    viol_jobs_per_slice = {s: 0 for s in slice_ids}
    dropped_jobs_per_slice = {s: 0 for s in slice_ids}
    delay_sum_per_app = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    delay_cnt_per_app = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    if app_packet_profile is None:
        app_packet_profile = APP_PACKET_PROFILE
    if appkeys is None:
        appkeys = APPKEY_LIST

    # Normalize profile keys to int app IDs.
    profile = {}
    for k, v in app_packet_profile.items():
        try:
            profile[int(k)] = v
        except Exception:
            continue

    # Empty allocation is valid: every job remains unserved/dropped.
    if allocation_df is None or allocation_df.empty:
        allocation_df = pd.DataFrame(columns=['User', 'App', 'Slice', 'Slot', 'Throughput'])

    app_slot_thr = (
        allocation_df.groupby(['User', 'App', 'Slice', 'Slot'])['Throughput']
        .sum()
        .to_dict()
    )

    episode_ms = float(episode_slots) * float(slot_ms)

    for (u, a, s) in appkeys:
        sla_bits = float(get_sla(a))
        if sla_bits <= 0:
            continue

        cfg = profile.get(int(a), {})
        payload_bits = float(cfg.get("payload_bits", sla_bits))
        pdb_ms = float(cfg.get("pdb_ms", episode_ms))
        if not np.isfinite(pdb_ms) or pdb_ms <= 0:
            pdb_ms = episode_ms

        jobs = _split_episode_volume_into_jobs(sla_bits, payload_bits)
        if not jobs:
            continue

        slot_budget = [
            float(app_slot_thr.get((u, a, s, t), 0.0))
            for t in range(int(episode_slots))
        ]

        for job_bits in jobs:
            total_jobs_per_slice[s] += 1
            bits_left = float(job_bits)
            completion_slot = None

            for t in range(int(episode_slots)):
                available = slot_budget[t]
                if available <= 0:
                    continue
                served = min(bits_left, available)
                slot_budget[t] -= served
                bits_left -= served
                if bits_left <= 1e-9:
                    completion_slot = t
                    break

            if completion_slot is None:
                # Not completed by episode end -> dropped and violated.
                viol_jobs_per_slice[s] += 1
                dropped_jobs_per_slice[s] += 1
                delay_sum_per_app[a] += episode_ms
                delay_cnt_per_app[a] += 1
            else:
                completion_ms = (completion_slot + 1) * float(slot_ms)
                delay_sum_per_app[a] += completion_ms
                delay_cnt_per_app[a] += 1
                if completion_ms > (pdb_ms + 1e-12):
                    viol_jobs_per_slice[s] += 1

    delay_rate_per_slice = {}
    for s in slice_ids:
        tot = total_jobs_per_slice[s]
        vio = viol_jobs_per_slice[s]
        delay_rate_per_slice[s] = (100.0 * vio / tot) if tot > 0 else 0.0
    avg_delay_per_app = {
        a: (delay_sum_per_app[a] / delay_cnt_per_app[a]) if delay_cnt_per_app[a] > 0 else 0.0
        for a in delay_sum_per_app
    }

    return {
        'Delay_Total_Jobs_Per_Slice': total_jobs_per_slice,
        'Delay_Violations_Per_Slice': viol_jobs_per_slice,
        'Delay_Violation_Rate_Per_Slice': delay_rate_per_slice,
        'Delay_Dropped_Per_Slice': dropped_jobs_per_slice,
        'Delay_Avg_Delay_Per_App_ms': avg_delay_per_app,
        'Delay_Total_Jobs': int(sum(total_jobs_per_slice.values())),
        'Delay_Total_Violations': int(sum(viol_jobs_per_slice.values())),
        'Delay_Total_Dropped': int(sum(dropped_jobs_per_slice.values())),
    }


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


def _safe_torch_load(path, map_location="cpu"):
    """
    Compatibility wrapper:
      - prefers weights_only=True (avoids FutureWarning and is safer)
      - falls back for older torch versions that do not support it
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def run_kbl_allocation(
    gamma,
    accuracy_range=(0.99, 0.999),
    alfa=0.05,
    drift_fallback=0.1,
    return_decision_time=False
):
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
    decision_t0 = time.perf_counter()

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

    decision_time_s = time.perf_counter() - decision_t0

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

    out = (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out


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


def _xslice_allocate_slot(gamma, slice_quota, slack, allocation_so_far, t, drift_fallback=0.1):
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
    drift_fallback=0.1,
    return_decision_time=False,
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
        ckpt = _safe_torch_load(model_path, map_location="cpu")
        model.load_state_dict(ckpt['model'])
    model.eval()

    allocation_records = []
    allocation_so_far = {ak: 0.0 for ak in APPKEY_LIST}
    slack = {ak: get_sla(ak[1]) for ak in APPKEY_LIST}
    prev_prb_used = {1: 0, 2: 0, 3: 0}
    decision_t0 = time.perf_counter()

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

    decision_time_s = time.perf_counter() - decision_t0

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

    out = (
        allocation_df,
        total_throughput,
        app_throughput_df,
        app_episode_thr,
        violation_counts_per_slice,
        total_violations,
        violation_gap_per_slice,
        slice_throughput_df
    )
    if return_decision_time:
        return out + (decision_time_s,)
    return out

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
    sns.set(style="whitegrid")
    method_specs = [
        ('Optimal',    'Optimal_ILP', '#81D4FA', 'P', '--'),
        ('ADSS_PPO',   'ADSS_PPO',    'black',   'D', '-'),
        ('ADSS_DQN',   'ADSS_DQN',    'orangered', 'X', '-'),
        ('HSRS',       'HSRS',        '#b185d4', 'x', '-.'),
        ('XSlice',     'XSlice',      '#49a3e4', 'd', '-'),
        ('RadioSaber', 'RadioSaber',  '#407552', '<', ':'),
        ('KBL',        'KBL',         'grey',    'o', '-'),
        ('BestCQI',    'BestCQI',     '#A5D6A7', 's', '--'),
    ]
    methods = [m for m, _, _, _, _ in method_specs]
    prefix_map = {m: p for m, p, _, _, _ in method_specs}
    palette = {m: c for m, _, c, _, _ in method_specs}
    marker_map = {m: mk for m, _, _, mk, _ in method_specs}
    linestyle_map = {m: ls for m, _, _, _, ls in method_specs}
    idx_map = {m: i + 1 for i, m in enumerate(methods)}

    def _set_axis_fonts(xlabel=None, ylabel=None):
        if xlabel is not None:
            plt.xlabel(xlabel, fontsize=20)
        if ylabel is not None:
            plt.ylabel(ylabel, fontsize=20)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=18)

    def _barplot_no_error(**kwargs):
        # seaborn>=0.12 uses errorbar=None; older versions use ci=None
        try:
            return sns.barplot(errorbar=None, **kwargs)
        except TypeError:
            return sns.barplot(ci=None, **kwargs)

    active_methods = [m for m in methods if f"{prefix_map[m]}_Total_Throughput" in results_df.columns]
    if not active_methods:
        print("No method throughput columns found in results_df.")
        return

    # --- 1) Total throughput ----------------------------
    min_datarate = results_df['Min_required_datarate_KB'].iloc[0]
    plt.figure(figsize=(10, 6))
    for m in active_methods:
        col = f"{prefix_map[m]}_Total_Throughput"
        plt.plot(
            results_df['Simulation'],
            results_df[col],
            marker=marker_map[m],
            label=m,
            color=palette[m],
            linestyle=linestyle_map[m],
        )
    plt.axhline(y=min_datarate, color='r', linestyle='--')
    plt.text(
        results_df['Simulation'].min(),
        min_datarate,
        f'Min total: {min_datarate} KB',
        color='r',
        fontsize=16,
        ha='left'
    )
    _set_axis_fonts('Allocation Episodes', 'Throughput (Kbit/frame)')
    plt.title('Throughput per Episodes', fontsize=26)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- 2) PRB usage ----------------------------------------
    plt.figure(figsize=(10, 6))
    prb_methods = [m for m in active_methods if f"{prefix_map[m]}_PRB_Usage (%)" in results_df.columns]
    for m in prb_methods:
        col = f"{prefix_map[m]}_PRB_Usage (%)"
        plt.plot(
            results_df['Simulation'],
            results_df[col],
            marker=marker_map[m],
            label=m,
            color=palette[m],
            linestyle=linestyle_map[m],
        )
    _set_axis_fonts('Allocation Episodes', 'PRB Usage (%)')
    plt.title('PRB Usage Rate', fontsize=26)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- 3) Spectral Efficiency per Episode --------------------
    plt.figure(figsize=(10, 6))
    total_prbs = 250
    for m in prb_methods:
        thr_col = f"{prefix_map[m]}_Total_Throughput"
        usage_col = f"{prefix_map[m]}_PRB_Usage (%)"
        used_prbs = (results_df[usage_col] / 100) * total_prbs
        se = results_df[thr_col] / used_prbs.replace(0, np.nan)
        plt.plot(
            results_df['Simulation'],
            se,
            marker=marker_map[m],
            label=m,
            color=palette[m],
            linestyle=linestyle_map[m],
        )
    _set_axis_fonts('Allocation Episodes', 'Spectral Efficiency (kbit per PRB)')
    plt.title('Spectral Efficiency per Episode', fontsize=26)
    handles, labels = plt.gca().get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    plt.legend(handles, numbered, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- 3-bis) Average total spectral efficiency (horizontal bar) ---
    if prb_methods:
        avg_total_se = []
        for m in prb_methods:
            thr_col = f"{prefix_map[m]}_Total_Throughput"
            usage_col = f"{prefix_map[m]}_PRB_Usage (%)"
            used_prbs = (results_df[usage_col] / 100) * total_prbs
            se = results_df[thr_col] / used_prbs.replace(0, np.nan)
            avg_total_se.append({
                'Method': m,
                'Average_Total_SE': se.mean(skipna=True)
            })

        avg_total_se_df = pd.DataFrame(avg_total_se)
        avg_total_se_df['Method'] = pd.Categorical(
            avg_total_se_df['Method'], categories=prb_methods, ordered=True
        )
        avg_total_se_df = avg_total_se_df.sort_values('Method')

        method_display_names = [
            "Optimal" if m in ("Optimal ILP", "Optimal") else m
            for m in avg_total_se_df['Method']
        ]
        se_values = avg_total_se_df['Average_Total_SE'].to_numpy(dtype=float)
        colors = [palette[m] for m in avg_total_se_df['Method']]
        x_max = np.nanmax(se_values) if np.size(se_values) else 1.0
        text_offset = 0.01 * (x_max if x_max > 0 else 1.0)

        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(method_display_names))
        bars = plt.barh(y_pos, se_values, color=colors)
        for bar, val in zip(bars, se_values):
            plt.text(
                bar.get_width() + text_offset,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}',
                va='center', ha='left', fontsize=13
            )
        plt.yticks(y_pos, method_display_names, fontsize=16)
        plt.xlim(0, max(x_max * 1.20, 1.0))
        plt.gca().invert_yaxis()
        _set_axis_fonts('Average Spectral Efficiency (kbit per RB)', '')
        plt.title('Average Spectral Efficiency', fontsize=26)
        plt.tight_layout()
        plt.show()

    # --- 3-ter) Average user Jain fairness (bar) --------------
    jain_methods = [m for m in active_methods if f"{prefix_map[m]}_Jain_Fairness" in results_df.columns]
    if jain_methods:
        avg_jain_df = pd.DataFrame({
            'Method': jain_methods,
            'Average_Jain_Fairness': [
                results_df[f"{prefix_map[m]}_Jain_Fairness"].mean() for m in jain_methods
            ]
        })
        avg_jain_df['Method'] = pd.Categorical(
            avg_jain_df['Method'], categories=jain_methods, ordered=True
        )

        plt.figure(figsize=(10, 6))
        ax = _barplot_no_error(
            data=avg_jain_df,
            x='Method',
            y='Average_Jain_Fairness',
            hue='Method',
            hue_order=jain_methods,
            palette={m: palette[m] for m in jain_methods},
            order=jain_methods,
            dodge=False
        )
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.3f',
                label_type='edge',
                padding=3,
                fontsize=12,
                color='black'
            )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        _set_axis_fonts('', "Jain's Fairness Index")
        plt.ylim(0, 1.05)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Inter-train Jain's Fairness", fontsize=26)
        plt.tight_layout()
        plt.show()

    # --- 3-ter) Per-application fairness across users ---------
    app_jain_methods = [
        m for m in active_methods
        if all(f"{prefix_map[m]}_App_{a}_Jain_Fairness" in results_df.columns for a in [1, 2, 3, 4, 5])
    ]
    if app_jain_methods:
        app_fairness = {'Application': app_labels}
        for m in app_jain_methods:
            pref = prefix_map[m]
            app_fairness[m] = [
                results_df[f"{pref}_App_{a}_Jain_Fairness"].mean() for a in [1, 2, 3, 4, 5]
            ]

        app_jain_df = pd.DataFrame(app_fairness).melt(
            id_vars=['Application'], var_name='Method', value_name='Jain_Fairness'
        )
        app_jain_df['Method'] = pd.Categorical(
            app_jain_df['Method'], categories=app_jain_methods, ordered=True
        )

        plt.figure(figsize=(12, 6))
        ax = _barplot_no_error(
            data=app_jain_df,
            x='Application',
            y='Jain_Fairness',
            hue='Method',
            hue_order=app_jain_methods,
            palette={m: palette[m] for m in app_jain_methods}
        )
        for x in [1.5, 3.5]:
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
        for container in ax.containers:
            labels = []
            for patch in container:
                height = patch.get_height()
                if np.isnan(height):
                    labels.append('')
                elif height < 1:
                    labels.append(f"{height:.2f}".replace("0.", ""))
                else:
                    labels.append(f"{height:.2f}")
            ax.bar_label(
                container,
                labels=labels,
                label_type='edge',
                padding=3,
                fontsize=12,
                color='black'
            )
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.ylim(0, 1.05)
        _set_axis_fonts('Applications', "App Jain's Fairness")
        plt.title("Applications inter-train  Jain's Fairness", fontsize=26)
        plt.tight_layout()
        plt.show()
    # --- helper maps -----------------------------------------
    slice_mapping = {1: 'Critical Slice', 2: 'Performance Slice', 3: 'Business Slice'}
    slice_nums = [1, 2, 3]

    # --- 4) Violation occurrence per Slice ----------------------
    viol_data = []
    for _, row in results_df.iterrows():
        for m in active_methods:
            pref = prefix_map[m]
            for s in slice_nums:
                viol_data.append({
                    'Method': m,
                    'Slice': slice_mapping[s],
                    'Violation_Count': row.get(f'{pref}_Violations_S{s}', 0)
                })
    viol_df = pd.DataFrame(viol_data)
    viol_df['Method'] = pd.Categorical(viol_df['Method'], categories=active_methods, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = _barplot_no_error(
        data=viol_df, x='Slice', y='Violation_Count',
        hue='Method', hue_order=active_methods,
        palette={m: palette[m] for m in active_methods}
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f',
            label_type='edge',
            padding=3,
            fontsize=12,
            color='black'
        )
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=12)
    _set_axis_fonts('', 'Violations per Episode')
    plt.title('Throughput Violation Occurrence per Slice', fontsize=26)
    plt.tight_layout()
    plt.show()

    # --- 5) Violation rates per Slice -----------------------
    rate_data = []
    for _, row in results_df.iterrows():
        for m in active_methods:
            pref = prefix_map[m]
            for s in slice_nums:
                rate_data.append({
                    'Method': m,
                    'Slice': slice_mapping[s],
                    'Violation_Rate (%)': row.get(f'{pref}_Violation_Rate_S{s}', 0)
                })
    rate_df = pd.DataFrame(rate_data)
    rate_df['Method'] = pd.Categorical(rate_df['Method'], categories=active_methods, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = _barplot_no_error(
        data=rate_df,
        x='Slice', y='Violation_Rate (%)',
        hue='Method', hue_order=active_methods,
        palette={m: palette[m] for m in active_methods}
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.1f',
            label_type='edge',
            padding=3,
            fontsize=12,
            color='black'
        )
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=12)
    _set_axis_fonts('', 'Violation Rate (%)')
    plt.title('Throughput Violation Rate per Slice', fontsize=26)
    plt.tight_layout()
    plt.show()

    # --- 6) Violation gap boxplot ---------------------------
    gap_data = []
    for _, row in results_df.iterrows():
        for m in active_methods:
            pref = prefix_map[m]
            for s in slice_nums:
                gap_data.append({
                    'Method': m,
                    'Slice': slice_mapping[s],
                    'Violation_gap': row.get(f'{pref}_Violation_gap_S{s}', 0) * 100
                })
    gap_df = pd.DataFrame(gap_data)
    gap_df['Method'] = pd.Categorical(gap_df['Method'], categories=active_methods, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=gap_df,
        x='Slice', y='Violation_gap',
        hue='Method', hue_order=active_methods,
        palette={m: palette[m] for m in active_methods},
        showmeans=True,showfliers=False,
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"}
    )
    for x in [0.5, 1.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=12)
    _set_axis_fonts('', 'Violation Gap (%)')
    plt.title('Throughput Violation Gap per Slice', fontsize=26)
    plt.tight_layout()
    plt.show()

    # --- 7) Delay violation rates per Slice ----------
    delay_methods = [
        m for m in active_methods
        if any(f"{prefix_map[m]}_Delay_Violation_Rate_S{s}" in results_df.columns for s in [1, 2, 3])
    ]
    if delay_methods:
        delay_rate_data = []
        for _, row in results_df.iterrows():
            for m in delay_methods:
                pref = prefix_map[m]
                for s in slice_nums:
                    delay_rate_data.append({
                        'Method': m,
                        'Slice': slice_mapping[s],
                        'Delay_Violation_Rate (%)': row.get(f'{pref}_Delay_Violation_Rate_S{s}', 0)
                    })
        delay_rate_df = pd.DataFrame(delay_rate_data)
        delay_rate_df['Method'] = pd.Categorical(
            delay_rate_df['Method'], categories=delay_methods, ordered=True
        )

        plt.figure(figsize=(10, 6))
        ax = _barplot_no_error(
            data=delay_rate_df, x='Slice', y='Delay_Violation_Rate (%)',
            hue='Method', hue_order=delay_methods,
            palette={m: palette[m] for m in delay_methods}
        )
        for x in [0.5, 1.5]:
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
        for container in ax.containers:
            ax.bar_label(
                container,
                fmt='%.1f',
                label_type='edge',
                padding=3,
                fontsize=12,
                color='black'
            )
        handles, labels = ax.get_legend_handles_labels()
        numbered = [f"{idx_map[l]}. {l}" for l in labels]
        ax.legend(handles, numbered, fontsize=12)
        _set_axis_fonts('', 'Delay Violation Rate (%)')
        plt.title('Delay Violation Rate per Slice', fontsize=26)
        plt.tight_layout()
        plt.show()


    # --- 10) Per-application average throughput -------------
    cols_map = {
        m: [c for c in results_df.columns if c.startswith(f"{prefix_map[m]}_App_")]
        for m in active_methods
    }
    avg = {
        m: (results_df[cols_map[m]].mean().to_dict() if cols_map[m] else {})
        for m in active_methods
    }
    achieved = {'Application': app_labels, 'SLA': sla_values}
    for m in active_methods:
        pref = prefix_map[m]
        achieved[m] = [avg[m].get(f"{pref}_App_{i+1}_Throughput", 0) for i in range(len(app_labels))]

    bar_df = pd.DataFrame(achieved).melt(
        id_vars=['Application', 'SLA'], var_name='Method', value_name='Throughput'
    )
    bar_df['Method'] = pd.Categorical(bar_df['Method'], categories=active_methods, ordered=True)

    plt.figure(figsize=(12, 6))
    ax = _barplot_no_error(
        data=bar_df, x='Application', y='Throughput', hue='Method',
        hue_order=active_methods, palette={m: palette[m] for m in active_methods}
    )
    for x in [1.5, 3.5]:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
    y_max = bar_df['Throughput'].max() if len(bar_df) else 1
    ax.text(0.5, y_max * 1.05, 'Critical slice', ha='center', va='bottom', fontsize=20)
    ax.text(2.5, y_max * 1.05, 'Performance slice', ha='center', va='bottom', fontsize=20)
    ax.text(4.0, y_max * 1.05, 'Business slice', ha='center', va='bottom', fontsize=20)

    bar_width = 0.25
    app_positions = np.arange(len(app_labels))
    for i, sla in enumerate(sla_values):
        x_start = app_positions[i] - (1.5 * bar_width)
        x_end = app_positions[i] + (1.5 * bar_width)
        plt.plot([x_start, x_end], [sla, sla], color='red', linestyle='--', linewidth=2)
        plt.text(i, sla + y_max * 0.01, f"SLA: {sla}", color='red', ha='right', fontsize=11)

    _set_axis_fonts('Applications', 'Average throughput (Bits/frame)')
    plt.title('Per-Application Average Throughput', fontsize=26)
    handles, labels = ax.get_legend_handles_labels()
    numbered = [f"{idx_map[l]}. {l}" for l in labels]
    ax.legend(handles, numbered, fontsize=12)
    plt.tight_layout()
    plt.show()

    # --- 9) Per-application average delay -------------------
    delay_app_cols_map = {
        m: [
            c for c in results_df.columns
            if c.startswith(f"{prefix_map[m]}_App_") and c.endswith("_Avg_Delay_ms")
        ]
        for m in active_methods
    }
    delay_app_methods = [m for m in active_methods if delay_app_cols_map[m]]
    if delay_app_methods:
        delay_data = {'Application': app_labels}
        for m in delay_app_methods:
            pref = prefix_map[m]
            avg_vals = results_df[delay_app_cols_map[m]].mean().to_dict()
            delay_data[m] = [
                avg_vals.get(f"{pref}_App_{i+1}_Avg_Delay_ms", 0.0)
                for i in range(len(app_labels))
            ]

        delay_df = pd.DataFrame(delay_data).melt(
            id_vars=['Application'], var_name='Method', value_name='Average_Delay_ms'
        )
        delay_df['Method'] = pd.Categorical(
            delay_df['Method'], categories=delay_app_methods, ordered=True
        )

        plt.figure(figsize=(12, 6))
        ax = _barplot_no_error(
            data=delay_df, x='Application', y='Average_Delay_ms', hue='Method',
            hue_order=delay_app_methods, palette={m: palette[m] for m in delay_app_methods}
        )
        for x in [1.5, 3.5]:
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1)
        _set_axis_fonts('Applications', 'Average Delay (ms)')
        plt.title('Average Delay per App', fontsize=24)
        handles, labels = ax.get_legend_handles_labels()
        numbered = [f"{idx_map[l]}. {l}" for l in labels]
        ax.legend(handles, numbered, fontsize=12)
        plt.tight_layout()
        plt.show()

    # --- 10) Spectral efficiency per slice (horizontal grouped) ----
    se_methods = [
        m for m in active_methods
        if all(f"{prefix_map[m]}_SE_S{s}" in results_df.columns for s in [1, 2, 3])
    ]
    if se_methods:
        slice_labels = ['Critical', 'Performance', 'Business']
        slice_ids = [1, 2, 3]
        data = np.array([
            [results_df[f'{prefix_map[m]}_SE_S{s}'].mean() for s in slice_ids]
            for m in se_methods
        ])

        method_names = se_methods
        method_colors = {m: palette[m] for m in method_names}
        method_display_names = [("Optimal" if m in ("Optimal ILP", "Optimal") else m) for m in method_names]

        ##### 1) Stacked bar per-method #####
        plt.figure(figsize=(10, 6))
        x = np.arange(len(method_names))
        bottom = np.zeros_like(x, dtype=float)
        bar_width = 0.6
        slice_colors = {
            'Critical': 'orange',
            'Performance': 'cyan',
            'Business': '#A5D6A7',
        }

        for i, sl in enumerate(slice_labels):
            vals = data[:, i]
            plt.bar(
                x, vals, bar_width,
                bottom=bottom,
                label=sl,
                color=slice_colors[sl]
            )
            for j, v in enumerate(vals):
                if v > 0:
                    plt.text(
                        x[j], bottom[j] + v / 2,
                        f'{v:.2f}',
                        ha='center', va='center', fontsize=10
                    )
            bottom += vals

        plt.xticks(x, method_display_names)
        plt.ylabel('(Bit per RB)', fontsize=14)
        plt.xlabel('', fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=18)
        plt.title('Average Per-Slice Spectral Efficiency', fontsize=26)
        plt.legend(title='Slice', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        plt.show()

        ##### 2) Horizontal grouped bar by method #####
        n_methods, n_slices = data.shape
        y = np.arange(n_methods)
        bar_h = 0.8 / n_slices

        ##### 3) Horizontal grouped bar by slice #####
        data_t = data.T  # slices x methods
        ys = np.arange(len(slice_labels))
        bar_h2 = 0.8 / n_methods
        x_max = np.nanmax(data_t) if np.size(data_t) else 1.0
        text_offset = 0.01 * (x_max if x_max > 0 else 1.0)

        plt.figure(figsize=(10, 6))
        for j, m in enumerate(method_names):
            vals = data_t[:, j]
            display_label = "Optimal" if m in ("Optimal ILP", "Optimal") else m
            legend_label = f"{idx_map[m]}. {display_label}"
            bars = plt.barh(
                ys + (j - (n_methods - 1) / 2) * bar_h2,
                vals,
                height=bar_h2,
                label=legend_label,
                color=method_colors[m]
            )
            for bar in bars:
                w = bar.get_width()
                yc = bar.get_y() + bar.get_height() / 2
                plt.text(w + text_offset, yc, f'{w:.2f}',
                         va='center', ha='left', fontsize=13)

        plt.yticks(ys, slice_labels, fontsize=16)
        plt.xticks(fontsize=14)
        plt.xlabel('Avg Spectral Efficiency (Bit per RB)', fontsize=16)
        plt.xlim(0, max(x_max * 1.20, 1.0))
        plt.legend(loc='lower left', bbox_to_anchor=(0.02, 0.02), fontsize=14, frameon=True)
        plt.tight_layout()
        plt.show()

def save_plot_all_results(results_df, output_dir="images", dpi=300, close_figures=True):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original_show = plt.show
    saved_files = []
    name_counts = {}

    def _title_to_name(title, fallback_index):
        cleaned = title.strip().lower()
        cleaned = re.sub(r'[^a-z0-9]+', '_', cleaned).strip('_')
        return cleaned or f"figure_{fallback_index}"

    def _show_and_save(*args, **kwargs):
        fig = plt.gcf()
        axes = fig.get_axes()
        title = ''
        for ax in axes:
            title = ax.get_title().strip()
            if title:
                break

        base_name = _title_to_name(title, len(saved_files) + 1)
        count = name_counts.get(base_name, 0) + 1
        name_counts[base_name] = count
        file_name = f"{base_name}.png" if count == 1 else f"{base_name}_{count}.png"
        file_path = output_path / file_name

        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        saved_files.append(str(file_path))
        original_show(*args, **kwargs)
        if close_figures:
            plt.close(fig)

    try:
        plt.show = _show_and_save
        plot_all_results(results_df)
    finally:
        plt.show = original_show

    return saved_files

# ------------------------------------------------------------------
def print_final_results(results_df):
    methods = ["ADSS_DQN", "ADSS_PPO"]
    slice_labels = {
        1: "Critic",
        2: "Perf",
        3: "Business",
    }
    metric_specs = [
        ("Delay", "{method}_Delay_Violation_Rate_S{slice_id}"),
        ("Throughput", "{method}_Violation_Rate_S{slice_id}"),
    ]

    rows = {}
    for method in methods:
        row = {}
        for metric_name, col_template in metric_specs:
            for slice_id, slice_label in slice_labels.items():
                col = col_template.format(method=method, slice_id=slice_id)
                if col in results_df.columns:
                    values = pd.to_numeric(results_df[col], errors="coerce")
                    row[(metric_name, slice_label)] = values.mean()
                else:
                    row[(metric_name, slice_label)] = np.nan
        rows[method] = row

    summary_df = pd.DataFrame.from_dict(rows, orient="index")
    summary_df.index.name = "Method"
    summary_df.columns = pd.MultiIndex.from_tuples(summary_df.columns)
    summary_df = summary_df.reindex(columns=pd.MultiIndex.from_tuples([
        ("Delay", "Critic"),
        ("Delay", "Perf"),
        ("Delay", "Business"),
        ("Throughput", "Critic"),
        ("Throughput", "Perf"),
        ("Throughput", "Business"),
    ]))

    print("\n=== ADSS Delay and Throughput Violation Rates (%) ===")
    print(summary_df.round(2).to_string())

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
        color='orangered',
        label='ADSS_DQN'
    )
    sns.lineplot(
        x='Episode',
        y='Episode_Reward',
        data=df_ppo,
        marker='s',
        color='black',
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
