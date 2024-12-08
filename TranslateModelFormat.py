import os
import joblib
import pickle

# INPUT_MODEL_DIR_PATH = 'models/lgbm_dart_iso_36696_36455_10_et_v4_1.41421356237-random-false_30s_seed1111'
# for model_id in range(10):
#     input_model_path = f'{INPUT_MODEL_DIR_PATH}/{model_id}.p'
#     model = joblib.load(input_model_path)

#     # output_directory_path = f'{INPUT_MODEL_DIR_PATH}_pkl'
#     # os.makedirs(output_directory_path, exist_ok=True)

#     # output_model_path = f'{output_directory_path}/{model_id}.p'
#     # with open(output_model_path, 'wb') as output_model_file:
#     #     pickle.dump(model, output_model_file)

#     output_directory_path = f'{INPUT_MODEL_DIR_PATH}_raw'
#     os.makedirs(output_directory_path, exist_ok=True)

#     output_gbdt_model_path = f'{output_directory_path}/{model_id}_gbdt.txt'
#     model['gbdt_model'].booster_.save_model(output_gbdt_model_path)

#     output_iso_model_path = f'{output_directory_path}/{model_id}_iso.p'
#     joblib.dump(model['isotonic_model'], output_iso_model_path)



INPUT_MODEL_DIR_PATHS = [
    'models/tabm_iso_35779_35762_10_et_v4_1.41421356237-random-false_15s_cfg5_seed4444_r1-10_gaw0',
    'models/tabm_iso_35410_35405_10_et_v4_1.41421356237-random-false_15s_cfg5_seed5555_r1-10_gaw0',
    'models/tabm_iso_35805_35799_10_et_v4_141421356237-random-false_15s_cfg5_seed4444_r1-10_gaw0330',
    'models/tabm_iso_35424_35435_10_et_v4_141421356237-random-false_15s_cfg5_seed5555_r1-10_gaw0330',
]
for input_model_dir_path in INPUT_MODEL_DIR_PATHS:
    for model_id in range(10):
        input_model_path = f'{input_model_dir_path}/{model_id}.pkl'
        model = joblib.load(input_model_path)

        output_directory_path = f'{input_model_dir_path}_cpu'
        os.makedirs(output_directory_path, exist_ok=True)

        model['tabm_model'].model = model['tabm_model'].model.cpu()

        output_model_path = f'{output_directory_path}/{model_id}.pkl'
        joblib.dump(model, output_model_path)