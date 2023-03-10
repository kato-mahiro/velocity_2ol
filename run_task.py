import argparse
import sys
import os
import shutil
from datetime import datetime

# The NEAT-Python library imports
import modneat
from modneat import parallel
# The helper used to visualize experiment results
import velocity_task

# The current working directory
local_dir = os.path.dirname(__file__)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='', default='FeedForwardNetwork')
    parser.add_argument('--config', type=str, help='', default='./configs/default_genome.ini')
    parser.add_argument('--checkpoint_interval', type=int, help='', default=100)
    parser.add_argument('--checkpoint_load', type=str, help='', default='')
    parser.add_argument('--savedir', type=str, help='', default='./results')
    parser.add_argument('--task', type=str, help='', default='velocity_task.velocity_task_N')
    parser.add_argument('--generation', type=int, help='', default=10000)
    parser.add_argument('--run_id', type=str, help='', default='')
    parser.add_argument('--comment', type=str, help='', default='no comment')
    parser.add_argument('--num_workers', type=int, help='', default=0)

    args = parser.parse_args()
    return args

def run_experiment(config_file, num_workers):
    """
    Arguments:
        config_file: the path to the file with experiment configuration
    """
    # Save comment.
    with open(out_dir + '/comment.txt' ,'a') as f:
        f.write('date: ' + NOW.strftime("%y-%m-%d-%H:%M:%S\n") )
        f.write(COMMAND.replace('--', '\n--'))

    # Load configuration.
    config = modneat.Config(GENOME_TYPE, modneat.DefaultReproduction,
                         modneat.DefaultSpeciesSet, modneat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    if(CHECKPOINT_LOAD_PATH == ''):
        p = modneat.Population(config)
    else:
        p = modneat.Checkpointer.restore_checkpoint(CHECKPOINT_LOAD_PATH)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(modneat.StdOutReporter(True))
    p.add_reporter(modneat.FileOutReporter(True, out_dir + '/results.txt'))
    stats = modneat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(modneat.Checkpointer(savedir=out_dir, stats=stats, generation_interval=CHECKPOINT_INTERVAL,\
                                        time_interval_seconds=None))

    # Run for up to args.generations.
    if(num_workers == 0 or num_workers == 1):
        best_genome = p.run(TASK.eval_genomes, GENERATION)
    else:
        if(hasattr(TASK, 'eval_single_genome')):
            parallel_evaluator = parallel.ParallelEvaluator(num_workers=num_workers, eval_function=TASK.eval_single_genome)
            best_genome = p.run(parallel_evaluator.evaluate, GENERATION)
        else:
            print(f"Error: {TASK} has no method 'eval_single_genome'.")
            print("please implement 'eval_single_genome' func for multithreading.")
            sys.exit()
    TASK.show_results(best_genome, config, stats, out_dir)


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)
    os.makedirs(out_dir + '/checkpoints', exist_ok=False)


if __name__ == '__main__':
    # Get args
    args = create_parser()
    NETWORK_TYPE = eval('modneat.nn.' + args.network)
    GENOME_TYPE = NETWORK_TYPE.genome_type()
    TASK = eval(args.task + '(network_type = NETWORK_TYPE)')
    CONFIG_PATH = os.path.join(local_dir, args.config)
    GENERATION = args.generation
    CHECKPOINT_INTERVAL = args.checkpoint_interval
    CHECKPOINT_LOAD_PATH = args.checkpoint_load
    NUM_WORKERS = args.num_workers
    COMMAND = ' '.join(sys.argv)
    NOW = datetime.now()

    if args.run_id == '':
        args.run_id = NOW.strftime("%y%m%d%H%M%S")

    # The directory to store outputs
    if(CHECKPOINT_LOAD_PATH == ''):
        out_dir = os.path.join(local_dir, args.savedir, args.task + '_' + args.network + '_' + args.run_id)
    else:
        out_dir = os.path.join(local_dir, args.savedir, '[CONTINUED]' + args.task + '_' + args.network + '_' + args.run_id)

    # Clean results of previous run if any or init the ouput directory
    clean_output()

    # Save config file
    shutil.copy(CONFIG_PATH, out_dir)

    # Run the experiment
    run_experiment(CONFIG_PATH, NUM_WORKERS)
