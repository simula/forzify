import pandas as pd
import motmetrics as mm

def filter_csv(file_path, track_ids):
    # Read the CSV file
    df = pd.read_csv(file_path, header=None)

    # Filter rows where the second column matches the track IDs
    filtered_df = df[df.iloc[:, 1].isin(track_ids)]

    # Save the filtered data to a new CSV file
    new_file_path = '/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-071/filtered_' + file_path.split('/')[-1]
    filtered_df.to_csv(new_file_path, index=False, header=False)
    print(f"Filtered data saved to {new_file_path}")
    return new_file_path

def load_data(file_path):
    # Load the CSV file, assuming no header and columns are properly formatted
    return pd.read_csv(file_path, header=None)

def evaluate_mot(soccernet_file, soccer_sum_file):
    # Load ground truth and predictions
    gt = load_data(soccernet_file)
    predictions = load_data(soccer_sum_file)

    # Create an accumulator and accumulate the frames
    acc = mm.MOTAccumulator(auto_id=True)

    for frame_number in range(len(gt)):
        gt_frame = gt[gt[0] == frame_number]  # Assuming 0th column is FrameID
        pred_frame = predictions[predictions[0] == frame_number]  # Same assumption

        gt_ids = gt_frame[1].values  # Assuming 1st column is ObjectID
        pred_ids = pred_frame[1].values  # Same assumption

        # Assuming columns 2, 3, 4, 5 are bounding box coordinates
        gt_boxes = gt_frame.iloc[:, 2:6].values
        pred_boxes = pred_frame.iloc[:, 2:6].values

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=1)  # Adjust max_iou as needed

        acc.update(
            gt_ids,
            pred_ids,
            distances
        )

    # Calculate additional metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                       'recall', 'precision', 'num_objects', \
                                       'mostly_tracked', 'partially_tracked', \
                                       'mostly_lost', 'num_false_positives', \
                                       'num_misses', 'num_switches', \
                                       'num_fragmentations', 'mota', 'motp' \
                                      ], \
                        name='acc')

    # Render and print summary
    strsummary = mm.io.render_summary(
        summary,
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                 'precision': 'Prcn', 'num_objects': 'GT', \
                 'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                 'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                 'num_misses' : 'FN', 'num_switches' : 'IDsw', \
                 'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)

# Filter and generate ground truth and prediction files
soccer_sum_file_path = filter_csv('/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-071/soccersum_track.csv', [3])
soccernet_file_path = filter_csv('/home/mehdihou/D1/soccernet_tracking/tracking/train/SNMOT-071/gt/gt.txt', [19])

# Evaluate MOT
evaluate_mot(soccernet_file_path, soccer_sum_file_path)
