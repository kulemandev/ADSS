
# ADDS: Application-aware Slicing for FRMCS: A Deep Reinforcement Learning Approach

Abstract: The Future Railway Mobile Communication System (FRMCS) will replace GSM-R to support safety-critical and high-throughput applications over a limited 5–10 MHz spectrum.
Railway services range from ultra-reliable train control, such as European Train Control System and Automatic Train Operation, to bandwidth-intensive video surveillance and best-effort passenger Wi-Fi, each with distinct requirements.
Existing network slicing solutions designed for public 5G networks focus on aggregate slice-level guarantees, neglecting heterogeneous application requirements and the strong channel fluctuations induced by high-speed train mobility. To overcome this limitation, we propose in this paper an Application-Driven Slice Scheduling (ADSS) approach tailored for railway communications. ADSS leverages Deep Reinforcement Learning combined with channel-aware resource allocation to dynamically assign Resource Blocks, ensuring application-level Service Level Agreement (SLA) fulfillment. Evaluations on real Signal-to-Noise Ratio traces from trains traveling at speeds up to 350 km/h, demonstrate that ADSS achieves superior application-level SLA satisfaction, reduces violation gaps, and improves spectral efficiency compared to heuristic and state-of-the-art schedulers.
## Paper contribution

- We formulate the FRMCS resource allocation and scheduling problem as an Integer Linear Program (ILP) that captures heterogeneous SLAs and slice priorities.
- We design a channel-aware, application-level SLA scheduling algorithm based on DRL with action masking, enabling efficient approximation of the ILP solution and ensuring spectral efficiency under physical constraints.
- We benchmark ADSS against existing approaches using real SNR traces from trains operating at up to 350 km/h, and demonstrate significant improvements in SLA satisfaction, violation reduction, and spectral efficiency. 
## Code Usage
Overview of Simulation Files

In this section, we present each file used in our simulation. These files are well-commented and divided into code sections ("cells") to allow modular execution and improve readability.


## 1. Subband CQI Dataset :
 
 dataset_maker.ipynb


This is the first part of our simulator. It involves loading the SNR traces [1] of the trains from the "SNR_railways_dataset" folder. These SNR values are converted into CQI values using the mapping table proposed in 
[2]. Once the CQI values are obtained, they are plotted to visualize their variation over time for each UE (TOBA gateway).

The second step consists of generating subband CQI values based on the wideband CQI for each entry in the CSV file. The resulting subband CQI data is saved in folder named subband_cqi, which will be used in the simulation.

During the simulation, each subband CQI will be progressively injected according to its corresponding UE and the episode timeline.


## 2. Main simulation : 
HSRS.ipynb <br>

this code strcuture is as follow :

**Parameter Initialization:**
The first part of the code defines all necessary parameters for the simulation, including the number of simulation episodes, the number of UEs per slice, and various system variables used in throughput calculation (e.g., subband CQI, MCS, MIMO layers, etc.).

**Subband CQI Injection:**
Subband CQI values are retrieved from the subband_cqi folder and injected at each resource allocation episode, which occurs every 10 milliseconds.

**Solution Approach Implemented:**

1. Gurobi Solver:
The Gurobi solver is used to solver the Integer Linear Programming (ILP) for resource allocation. An academic free license is available for installation. https://www.gurobi.com/academia/academic-program-and-licenses/

Once the  gurobi academic license key is obtained for eaxmple
`4d7be1b0-9d37-49e0-bc1e-9b11e1c54bba`
, then download the .lic file in a path. insert this :   `export GRB_LICENSE_FILE="/path/to/your/gurobi.lic"`
into bashrc file: 
```nano ~/.bashrc```  

run it with: ```source ~/.bashrc```
finaly install the gurobi using pip: ``` pip install gurobipy```


2. Baseline Algorithm Implementations:
implementations of baseline algorithms such as NVS, BestCQI and RadioSaber.

3. HSRS Implementation: run_myheuristic_allocation():
Our previous heuristic algorithm, HSRS, is implemented and integrated into the framework.

**Main Simulation Loop:**
This section coordinates all components, calling relevant functions in sequence to execute the simulation.

Performance Metrics and Visualization:
Finally, key performance metrics are computed and plotted to evaluate and compare the performance of the different algorithms.

In case you did not success to install Gurobi license the "HSRS.ipynb" code will not work because of errors. You can use the **"HSRS_without_ILP_solver.ipynb"** file where Gurobi solver is removed from the code.

[1] Y. Pan, R. Li, and C. Xu, “The First 5G-LTE Comparative Study in
Extreme Mobility,” Proceedings of the ACM on Measurement and
Analysis of Computing Systems, vol. 6, no. 1, pp. 1–22, Feb. 2022.

[2] A. K. Thyagarajan, P. Balasubramanian, V. D, and K. M, “SNR-CQI Mapping for 5G Downlink Network,” in 2021 IEEE Asia Pacific Conference on Wireless and Mobile (APWiMob), Apr. 2021, pp. 173–177 
