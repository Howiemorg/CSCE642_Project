import optparse
import sys
import os
import random

# from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
import Domains.Available_domains as dvs
from utils import plotting
import matplotlib.pyplot as plt


def build_parser():
    parser = optparse.OptionParser(
        description="Run a specified RL algorithm on a specified domain."
    )
    parser.add_option(
        "-s",
        "--agent",
        dest="solver",
        type="string",
        default="myddpg",
        help="Solver from " + str(avs.solvers),
    )
    parser.add_option(
        "-d",
        "--domain",
        dest="domain",
        type="string",
        default="Empty",
        help="Domain from "+ str(dvs.domains),
    )
    parser.add_option(
        "-o",
        "--outfile",
        dest="outfile",
        default="out",
        help="Write results to FILE",
        metavar="FILE",
    )
    parser.add_option(
        "-x",
        "--experiment_dir",
        dest="experiment_dir",
        default="Experiments",
        help="Directory to save Tensorflow summaries in",
        metavar="FILE",
    )
    parser.add_option(
        "-e",
        "--episodes",
        type="int",
        dest="episodes",
        default=4000,
        help="Number of episodes for training",
    )
    parser.add_option(
        "-t",
        "--steps",
        type="int",
        dest="steps",
        default=10000,
        help="Maximal number of steps per episode",
    )
    parser.add_option(
        "-l",
        "--layers",
        dest="layers",
        type="string",
        default="[512,256]",
        help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
        "Input layer is connected to a layer of size 10 that is connected to a layer of size 15"
        " that is connected to the output",
    )
    parser.add_option(
        "-j",
        "--actor_alpha",
        dest="actor_alpha",
        type="float",
        default=1e-5,
        help="The learning rate (alpha) for updating actor network parameters",
    )
    parser.add_option(
        "-k",
        "--critic_alpha",
        dest="critic_alpha",
        type="float",
        default=3e-5,
        help="The learning rate (alpha) for updating critic network parameters",
    )
    parser.add_option(
        "-t",
        "--steps",
        type="int",
        dest="steps",
        default=10000,
        help="Maximal number of steps per episode",
    )
    parser.add_option(
        # "-A",
        "--actor_trace_delay",
        dest="actor_trace_delay",
        type="float",
        default=.95,
        help="The trace delay for actor eligibility traces",
    )
    parser.add_option(
        # "-A",
        "--critic_trace_delay",
        dest="critic_trace_delay",
        type="float",
        default=.95,
        help="The trace delay for critic eligibility traces",
    )
    parser.add_option(
        "-r",
        "--seed",
        type="int",
        dest="seed",
        default=random.randint(0, 9999999999),
        help="Seed integer for random stream",
    )
    parser.add_option(
        "-g",
        "--gamma",
        dest="gamma",
        type="float",
        default=0.99,
        help="The discount factor (gamma)",
    )
    parser.add_option(
        "-p",
        "--epsilon",
        dest="epsilon",
        type="float",
        default=0.1,
        help="Initial epsilon for epsilon greedy policies (might decay over time)",
    )
    parser.add_option(
        "-P",
        "--final_epsilon",
        dest="epsilon_end",
        type="float",
        default=0.1,
        help="The final minimum value of epsilon after decaying is done",
    )
    parser.add_option(
        "-E",
        "--decay",
        dest="epsilon_decay",
        type="float",
        default=0.99,
        help="Epsilon decay factor",
    )
    parser.add_option(
        "-m",
        "--replay",
        type="int",
        dest="replay_memory_size",
        default=64,
        help="Size of the replay memory",
    )
    parser.add_option(
        "-b",
        "--batch_size",
        type="int",
        dest="batch_size",
        default=128,
        help="Size of batches to sample from the replay memory",
    )
    parser.add_option(
        "-v",
        "--mdp_buff_size",
        type="int",
        dest="mdp_buff_size",
        default=5,
        help="Size of mdp buffer",
    )
    parser.add_option(
        "--no-plots",
        help="Option to disable plots if the solver results any",
        dest="disable_plots",
        default=False,
        action="store_true",
    )
    parser.add_option(
        "--save_every",
        type="int",
        help="The number/step of episodes to save the model weights",
        default=5000,
    )
    return parser


def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options

def parse_list(string):
    string.strip()
    string = string[1:-1].split(",")  # Change "[0,1,2,3]" to '0', '1', '2', '3'
    l = []
    for n in string:
        l.append(int(n))
    return l


def main(options):
    resultdir = "Results/"

    resultdir = os.path.abspath(f"./{resultdir}")
    # options.experiment_dir = os.path.abspath(f"./{options.experiment_dir}")

    # Create result file if one doesn't exist
    print(os.path.join(resultdir, options.outfile + ".csv"))
    if not os.path.exists(os.path.join(resultdir, options.outfile + ".csv")):
        with open(
            os.path.join(resultdir, options.outfile + ".csv"), "w+"
        ) as result_file:
            result_file.write(AbstractSolver.get_out_header())

    random.seed(options.seed)
    # env = getEnv(options.domain)

    # env = PyFlytGymEnv(options.domain)
    env = dvs.get_domain_class(options.domain)()# args to the domain go in the 2nd pair of parantheis

    env._max_episode_steps = options.steps + 1  # suppress truncation

    eval_env = dvs.get_domain_class(options.domain)(render_mode="human")
    print(f"\n---------- {options.domain} ----------")
    print(f"Domain state space is", ' '.join([f'{k.capitalize()} ({v})\n' for k,v in env.observation_space.items()]))
    print(f"Domain action space is {env.action_space}")
    print("-" * (len(options.domain) + 22) + "\n")
    try:
        options.layers = parse_list(options.layers)
    except ValueError:
        raise Exception(
            "layers argument doesnt follow int array conventions i.e., [<int>,<int>,<int>,...]"
        )
    except:
        pass
    solver = avs.get_solver_class(options.solver)(env, eval_env, options)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=[], episode_rewards=[])
    # print("goals", env.waypoints.targets)
    plt.ion()
    # if not options.disable_plots:
    #     # Detects key press for rendering
    #     from pynput import keyboard
    #     listener = keyboard.Listener(on_press=on_press)
    #     listener.start()  # start listening on a separate thread


    with open(os.path.join(resultdir, options.outfile + ".csv"), "a+") as result_file:
        result_file.write("\n")
        for i_episode in range(options.episodes):
            solver.init_stats()
            solver.statistics[Statistics.Episode.value] += 1
            env.reset(seed=123)
            solver.train_episode()
            result_file.write(solver.get_stat() + "\n")
            # Decay epsilon
            if options.epsilon > options.epsilon_end:
                options.epsilon *= options.epsilon_decay
            # Update statistics
            stats.episode_rewards.append(solver.statistics[Statistics.Rewards.value])
            stats.episode_lengths.append(solver.statistics[Statistics.Steps.value])
            print(
                f"Episode {i_episode+1:4d}: Reward {solver.statistics[Statistics.Rewards.value]:.6f}, Steps {solver.statistics[Statistics.Steps.value]}"
            )

            if not options.disable_plots:
                solver.plot(stats, int(0.1 * options.episodes), False)

        solver.export_weights()


    if not options.disable_plots:
        try:
            solver.load_weights()
        except FileNotFoundError:
            print("Weights not found, starting with no weights")
        for i_episode in range(options.episodes):
            rewards, steps = solver.run_greedy()
            stats.episode_lengths.append(steps)
            stats.episode_rewards.append(rewards)
        solver.plot(stats, int(0.1 * options.episodes), True)

        solver.close()
    plt.ioff()

    return {"stats": stats, "solver": solver}


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    options = readCommand(sys.argv)
    main(options)

