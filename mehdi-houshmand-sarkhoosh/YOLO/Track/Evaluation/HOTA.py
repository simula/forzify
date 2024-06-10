import pandas as pd
import os
import trackeval  # Make sure you have TrackEval installed

# Function to convert CSV to TrackEval format
def convert_csv_to_trackeval_format(csv_file, output_folder, is_gt):
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(csv_file, header=None)

    # Group by the first column (frame number)
    grouped = df.groupby(df.columns[0])

    for frame, group in grouped:
        filename = f"{output_folder}/{frame}.txt"
        with open(filename, 'w') as f:
            for index, row in group.iterrows():
                # Assuming CSV format: FrameID, ID, x, y, w, h, confidence
                id = row[1]
                bbox = row[2:6].tolist()
                conf = row[6] if not is_gt else 1  # For GT, confidence is always 1
                line = f"{id},{','.join(map(str, bbox))},{conf}\n"
                f.write(line)

# Convert ground truth and prediction CSV files
convert_csv_to_trackeval_format('/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-060/filtered_gt.txt', '/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-060/converted_ground_truth')
convert_csv_to_trackeval_format('/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-060/filtered_soccersum_track.csv', '/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-060/converted_predictions')

# Function to perform HOTA evaluation
def evaluate_hota(ground_truth_folder, prediction_folder):
    # Evaluation configuration
    configs = trackeval.Evaluator.get_default_eval_config()
    configs['EVAL']['METRICS'] = ['HOTA']

    dataset_config = trackeval.datasets.TrackEvalDataset.get_default_dataset_config()
    dataset_config['GT_FOLDER'] = ground_truth_folder
    dataset_config['TRACKERS_FOLDER'] = prediction_folder
    dataset_config['TRACKERS_TO_EVAL'] = ['converted_predictions']

    # Evaluate
    evaluator = trackeval.Evaluator(configs)
    metrics_list = trackeval.metrics.get_metrics_list(configs, trackeval.datasets.TrackEvalDataset(dataset_config))
    dataset_list = [trackeval.datasets.TrackEvalDataset(dataset_config)]

    results, messages = evaluator.evaluate(metrics_list, dataset_list)

    # Print results
    for metric, res in results.items():
        print(f"Results for {metric}:")
        print(res)

# Perform evaluation
evaluate_hota('path/to/converted_ground_truth', 'path/to/converted_predictions')