# CSCE 642 Project: Autonomous Drone Navigation using DDPG

## Overview
This project, **Autonomous Drone Navigation using DDPG**, was developed as part of CSCE 642: Reinforcement Learning at Texas A&M University. This project focuses on training an autonomous drone to navigate a 3D environment using policy gradient reinforcement learning. The drone must optimize its flight path to efficiently reach target locations while avoiding obstacles and adapting to environmental factors such as wind. The simulation environment, built with PyFlyt, features customizable drone configurations, dynamic obstacle generation, and integrated wind field models, providing robust and realistic training conditions. The learning algorithm employs a continuous (non-episodic) n-step Deep Deterministic Policy Gradient (n-step) method with priority-state buffer, enabling the drone to make sequential decisions and refine its policy based on observed hazards (out-of-bounds, crashing). This approach ensures the drone effectively balances efficiency, collision avoidance, and adaptability in dynamic scenarios.

## Technologies Used
The following technologies and tools were utilized in this project:
- **Programming Languages:** Python 3.10
- **Frameworks/Libraries:**  PyTorch, PyFlyt, Numpy, Matplotlib, OpenAI Gymnasium, PyBullet

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/Howiemorg/CSCE642_Project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd CSCE642_Project
   ```
3. Create a virutal environment & activate it:
   ```bash
   python -m venv dronetest
   ```
   ```bash
   dronetest/Scripts/activate.sh or .bat
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the project (For example, with full default values):
   ```bash
   python run.py -s myddpg -d Empty
   ```

## File Structure
```
CSCE642_Project/
├── README.md          # Project documentation
├── Domains/           # Premade scenarios (equivalent to envs)
├── Models/            # Physics models to be loaded in the domains
├── Results/           # Additional documentation about the result output
├── Solvers/           # Several agent algorithms
├── utils/             # Utilites such as statistics and plotting
├── Weights/           # Weights for the networks to be stored for failover
└── requirements.txt   # Dependencies list
└── run.py             # Main Operating Script
```

## Usage

Run.py parameters are listed below:

For this project, the way to replicate the results we show in the final report is to run:

```bash
   python run.py -s myddpg -d Empty
   ```

| Short Flag | Long Flag                  | Type   | Default          | Help                                                                                                                                                            |
|------------|----------------------------|--------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -s         | --agent                   | string | myddpg           | Solver from [avs.solvers]                                                                                                                                       |
| -d         | --domain                  | string | Empty            | Domain from [dvs.domains]                                                                                                                                       |
| -o         | --outfile                 | string | out.csv          | Write results to FILE                                                                                                                                           |
| -e         | --episodes                | int    | 4000             | Number of episodes for training                                                                                                                                |
| -l         | --layers                  | string | [512,256]        | Size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the input layer is connected to a layer of size 10 that connects to size 15 etc. |
| -j         | --actor_alpha             | float  | 3e-3             | The learning rate (alpha) for updating actor network parameters                                                                                                |
| -k         | --critic_alpha            | float  | 1e-5             | The learning rate (alpha) for updating critic network parameters                                                                                               |
| -r         | --seed                    | int    | Random integer   | Seed integer for random stream                                                                                                                                 |
| -g         | --gamma                   | float  | 1.00             | The discount factor (gamma)                                                                                                                                    |
| -p         | --epsilon                 | float  | 0.1              | Initial epsilon for epsilon-greedy policies (might decay over time)                                                                                            |
| -m         | --replay                  | int    | 128              | Size of the replay memory                                                                                                                                      |
| -b         | --batch_size              | int    | 20000            | Size of batches to sample from the replay memory                                                                                                               |
| -v         | --mdp_buff_size           | int    | 5                | Size of Markov Decision Prioirty buffer (State) buffer                                                                                                                                             |
|            | --no-plots                |        | False            | Option to disable plots if the solver results any                                                                                                              |


## Contributors
- Daniel Ortiz-Chaves, UIN: 128009829, Email: dortizchaves@tamu.edu
- Howie Morgenthaler, UIN: 130008345, Email: howiemorgenthaler@tamu.edu

## Acknowledgments
This project was inspired by the *csce642-deepRL* repo from the CSCE 642 course. 

## Known Issues

If running the program is raising a `Model/target.urdf` file not found error from the PyFlyt library, here's a simple fix:

1. Navigate to the PyFlyt Libary directory (assuming a venv named dronetest):
   ```bash
   cd dronetest/Lib/site-packages/PyFlyt
   ```
2. Copy the models directory to gym_envs (as a sub-directory):
   ```bash
   cp -a models/. gym_envs/models
   ```


