import glob
import json

MCTS_CONFIG_NAMES = [
    '1.41421356237-random-false',
    '0.6-random-true',
]
MCTS_RUNTIMES_SEC = [15, 30]

for run_id in range(1,11):
    for config_name in MCTS_CONFIG_NAMES:
        for runtime_sec in MCTS_RUNTIMES_SEC:
            # input_filepaths = glob.glob(f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-{config_name}_{runtime_sec}s_v2_r{run_id}_p*.json')
            # output_filepath = f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/organizer_UCB1Tuned-{config_name}_{runtime_sec}s_v2_r{run_id}.json'

            input_filepaths = glob.glob(f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/extra*_UCB1Tuned-{config_name}_{runtime_sec}s_v2_r{run_id}.json')
            output_filepath = f'StartingPositionEvaluation/Evaluations/FromKaggle_v2/merged_extra_UCB1Tuned-{config_name}_{runtime_sec}s_v2_r{run_id}.json'

            all_luds_to_evals: dict[str, float] = {}
            for input_filepath in input_filepaths:
                with open(input_filepath, 'r') as input_file:
                    luds_to_evals = json.load(input_file)
                    all_luds_to_evals.update(luds_to_evals)

            with open(output_filepath, 'w') as output_file:
                json.dump(all_luds_to_evals, output_file, indent=4)

            print(f'Saved {len(all_luds_to_evals)} LUDs to {output_filepath}')