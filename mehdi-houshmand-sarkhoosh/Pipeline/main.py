from ultralytics import YOLO
import json
import torch
import torchvision
import time
import subprocess
import os
import cv2
from collections import defaultdict
import cv2
import numpy as np
import urllib.parse
import requests
import openai
import whisper
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import librosa



has_gpu = torch.cuda.is_available()
has_mps = getattr(torch, 'has_mps', False)
# Once https://github.com/pytorch/pytorch/issues/77818 is resolved we can try again to run on MPS
has_mps = False
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"


class Frame_Type:
    def __init__(self, category, subcategory, start_frame, end_frame):
        
        self.category = category
        self.subcategory = subcategory
        self.start_frame = start_frame
        self.end_frame = end_frame

    def dict(self):
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame
        }



def download_and_convert_m3u8(m3u8_url, output_mp4_path):
    # Determine the ffmpeg command based on file type
    if m3u8_url.endswith('.m3u8'):
        ffmpeg_command = ['ffmpeg', '-i', m3u8_url, '-c', 'copy', '-bsf:a', 'aac_adtstoasc', output_mp4_path]
    elif m3u8_url.endswith('.mp4'):
        ffmpeg_command = ['ffmpeg', '-i', m3u8_url, '-c', 'copy', output_mp4_path]
    else:
        raise ValueError("Unsupported file format. Please provide a '.m3u8' or '.mp4' file.")

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during download/conversion: {e}")
        return None

    return output_mp4_path

def process_video(video_path):
    # Simplify or sanitize the filename
    parsed_url = urllib.parse.urlparse(video_path)
    base_filename = os.path.basename(parsed_url.path)
    simple_filename = os.path.splitext(base_filename)[0] + '.mp4'

    # Specify a directory for saving the output file
    output_directory = '/home/mehdihou/D1/02_Thesis/video'  # Change this to your desired directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_mp4_path = os.path.join(output_directory, simple_filename)

    if video_path.endswith('.m3u8') or video_path.startswith('http://') or video_path.startswith('https://'):
        video_path = download_and_convert_m3u8(video_path, output_mp4_path)

    return video_path




def reduce_fps(input_video_path, method='set_fps', target_fps=1):
    """
    Reduce the frame rate of a video either uniformly or by setting a target FPS, 
    and save it in a 'reduced_fps' subdirectory.

    :param input_video_path: Path to the input video.
    :param method: The method of reducing FPS ('uniform' or 'set_fps').
    :param target_fps: Desired frames per second (used for 'set_fps' method).
    :return: Path where the reduced FPS video was saved, or None if an error occurred.
    """
    base_dir = os.path.dirname(input_video_path)
    base_filename = os.path.basename(input_video_path)
    output_dir = os.path.join(base_dir, 'reduced_fps')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reduced_fps_video_path = os.path.join(output_dir, f'reduced_fps_{base_filename}')

    if method == 'uniform':
        # Calculate frame selection rate (e.g., select one frame every second)
        ffmpeg_command = ['ffmpeg', '-i', input_video_path, '-vf', 'select=not(mod(n\,25))', '-vsync', 'vfr', reduced_fps_video_path]
    elif method == 'set_fps':
        # Set target FPS
        ffmpeg_command = ['ffmpeg', '-i', input_video_path, '-filter:v', f'fps=fps={target_fps}', reduced_fps_video_path]
    else:
        raise ValueError("Invalid method. Choose 'uniform' or 'set_fps'.")

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while reducing FPS: {e}")
        return None

    return reduced_fps_video_path



    
def load_shot_prediction_model():
    # load shot type classification model, training available in https://github.com/sssabet/Shot_Type_Classification
    if device == 'cpu':
        shot_model = torch.load('/home/mehdihou/D1/02_Thesis/shot_type_classification.pt', map_location =torch.device('cpu'))
    else:
        shot_model = torch.load('/home/mehdihou/D1/02_Thesis/shot_type_classification.pt')
    return shot_model



shot_model = load_shot_prediction_model()


# Function that i-recives a frame(128,128) and model 2-does the nessaccery transformations, 3-predicts the types of shot and 4-returns the predicted label of frame type shot
# Works fast on cuda but takes some time for cpu, lower image sizes speeds up the proccess with a lower accuracy
def predict_shot(model, frame):
    IMAGE_SIZE = (128,128)        
    data_transformation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Resize(IMAGE_SIZE),torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    images = data_transformation(frame)
    with torch.no_grad():
        pred = model(images.view([1, 3, 128, 128]).to(device))
        _ , predictes = torch.max(pred,1)

    if predictes.item() == 0:
        predicted_type = 'CS'
    elif predictes.item() == 1:
        predicted_type = 'ECS'
    elif predictes.item() == 2:
        predicted_type = 'FS'
    elif predictes.item() == 3:
        predicted_type = 'LS'
    else:
        predicted_type = 'MS'

    return predicted_type


def predict_frame_type(video_path, model=shot_model) -> [Frame_Type]:
    frame_type = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_n = 0
    start = 1
    predicte = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if frame_n % fps == 0:  # TO reduce the computation run the model once for each half a second
            old_predicte = predicte
            predicte = predict_shot(model,frame)
            
            if old_predicte != predicte or frame_n > cap.get(cv2.CAP_PROP_FRAME_COUNT)-fps:  # if new frame type or the last frame 
                
                frame_type.append(Frame_Type('frame_type', predicte, start_frame= start, end_frame=frame_n-1))
      
                start = frame_n
    

    # Saving transitions into a JSON file
    filename = 'frame_type.json'
    with open(filename, 'w') as f:
        json.dump([ft.dict() for ft in frame_type], f)

    if os.path.isfile(filename):
        print(f"{filename} has been created and frame_types have been written successfully.")
    else:
        print(f"Failed to create {filename}.")



    # Filter for 'LS' subcategory frames
    ls_frames = [ft for ft in frame_type if ft.subcategory == 'LS' or ft.subcategory == 'MS' or ft.subcategory == 'FS']

    # Open the original video
    cap = cv2.VideoCapture(video_path)

    
    
    # Define the output file path
    output_file_name = 'output_ls.mp4'
    output_path = os.path.join('/home/mehdihou/D1/02_Thesis/downsample_video', output_file_name)
    
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'

    # Create VideoWriter with the specified path
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))


    print(f"Video will be saved to: {output_path}")

    # Read through the video and write frames that belong to 'LS' subcategory
    frame_n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        if any(ft.start_frame <= frame_n <= ft.end_frame for ft in ls_frames):
            out.write(frame)

    # Release everything when done
    cap.release()
    out.release()
    cv2.destroyAllWindows()


    return frame_type, output_path




#Starting with Detected Objects

class DetectedObjects:
    def __init__(self, category, confidence, x1, y1, x2, y2, frame_number):
        self.category = category
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return{
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }


class SegmentedObject:
    def __init__(self, category, confidence, x1, y1, x2, y2, frame_number):
        self.category = category
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return{
            "category": self.category,
            "confidence": round(self.confidence, 2),
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }
    

def detect_yolo(output_path):
    
    #Loading the pretrained YOLOv8m Model
    model = YOLO('/home/mehdihou/D1/YOLO-Train/Train/Thesis_detection_03/train/weights/best.pt')

    #Running the Inference on the video
    results = model.predict(output_path, stream=True)

    #Creating a List to store the detected Objects
    detected_objects = []

    class_names = ['Player', 'Goalkeeper', 'Referee', 'Ball', 'Logo', 'Penalty Mark', 'Corner Falgpost', 'Goal Net']


    for frame_n, result in enumerate(results):
        #moving the object to the CPU memory
        result = result.cpu()
        
        #the Boxes object containg the detection bouding boxes
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf)
            clazz = int(box.cls)

            detected_objects.append(DetectedObjects(class_names[clazz], conf, x1, y1, x2, y2, frame_n))

    
    return detected_objects


def segment_yolo(output_path):

    model = YOLO("/home/mehdihou/D1/YOLO-Train/Train/Segmentation_02/train/weights/best.pt")

    results = model.predict(output_path, stream=True)

    segmented_objects = []

    class_names = ['PenaltyBox', 'GoalBox']


    for frame_n, result in enumerate(results):

        result = result.cpu()

        #the Boxes object containg the detection bouding boxes
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf)

            clazz = int(box.cls)

            segmented_objects.append(SegmentedObject(class_names[clazz], conf, x1, y1, x2, y2, frame_n))

    return segmented_objects


class TrackedObject:
    def __init__(self, category, track_id, x1, y1, x2, y2, frame_number):
        self.category = category
        self.track_id = track_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.frame_number = frame_number

    def dict(self):
        return {
            "category": self.category,
            "track_id": self.track_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "frame_number": self.frame_number
        }

def track_yolo(output_path):
    model = YOLO('/home/mehdihou/D1/YOLO-Train/Train/Thesis_detection_03/train/weights/best.pt')
    cap = cv2.VideoCapture(output_path)

    tracked_objects = []
    class_names = ['Player', 'Goalkeeper', 'Referee', 'Ball', 'Logo', 'Penalty Mark', 'Corner Flagpost', 'Goal Net']  # Define your class names here
    goalkeeper_class_id = class_names.index('Goalkeeper')  # Get the class ID for 'Goalkeeper'

    frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)

            if results and results[0] and results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id
                class_ids = results[0].boxes.cls

                if track_ids is not None and class_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                    class_ids = class_ids.int().cpu().tolist()

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        if class_id == goalkeeper_class_id:  # Filter for Goalkeepers
                            x, y, w, h = box
                            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                            tracked_objects.append(TrackedObject('Goalkeeper', track_id, x1, y1, x2, y2, frame_number))
            
            frame_number += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return tracked_objects


#SmartCobo - Trying to Group the elements in the text file

def reduce_floats_in_item(item):
    """ Recursively reduce float numbers in a JSON item (dict or list) """
    if isinstance(item, dict):
        return {key: reduce_floats_in_item(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [reduce_floats_in_item(elem) for elem in item]
    elif isinstance(item, float):
        return round(item, 1)
    return item

def format_and_write_txt(data, txt_path):
    """ Format and group objects by frame number, then write to a text file """
    with open(txt_path, 'w') as file:
        frame_data = {}
        # Grouping objects by frame_number
        for list_name, list_items in data.items():
            if isinstance(list_items, list):
                for obj in list_items:
                    frame = obj.get("frame_number", "unknown")
                    frame_data.setdefault(frame, {}).setdefault(list_name, []).append(obj)

        for frame, objects in frame_data.items():
            file.write(f"frame_number: {frame} {{\n")
            for list_name, items in objects.items():
                file.write(f"  {list_name}:\n")
                for obj in items:
                    line = " ".join([f"{obj.get(key, '')}" for key in obj if key != "frame_number"])
                    file.write(f"    {line}\n")
            file.write("}\n\n")

def reduce_floats_and_convert_to_txt(input_path, output_txt_path):
    """ Process JSON file, reduce floats, and convert to text format """
    try:
        with open(input_path, 'r') as file:
            data = json.load(file)

        modified_data = reduce_floats_in_item(data)
        format_and_write_txt(modified_data, output_txt_path)

        return "TXT file generated successfully."
    except Exception as e:
        return f"Error processing file: {e}"



def filter_frames(file_path, min_detections):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    filtered_data = []
    frame_info = []
    detection_count = 0
    in_detection_block = False
    in_metadata_section = True

    for line in lines:
        if in_metadata_section and 'frame_number' not in line:
            filtered_data.append(line)
        else:
            in_metadata_section = False
            if 'frame_number' in line:
                # Check if previous frame should be kept
                if detection_count >= min_detections:
                    filtered_data.extend(frame_info)
                frame_info = [line]
                detection_count = 0
                in_detection_block = False
            elif 'detected_objects:' in line:
                frame_info.append(line)
                in_detection_block = True
            elif in_detection_block and line.strip() and not line.strip().startswith('}'):
                frame_info.append(line)
                detection_count += 1
            else:
                if in_detection_block and line.strip().startswith('}'):
                    in_detection_block = False
                frame_info.append(line)

    # Check the last frame
    if detection_count >= min_detections:
        filtered_data.extend(frame_info)

    # Write filtered data back to the file
    with open(file_path, 'w') as file:
        file.writelines(filtered_data)


def chat_with_openai(openai_api_key, text_file_path):
    # Initialize conversation history
    conversation_history = []

    # Read the contents of the text file
    with open(text_file_path, 'r') as file:
        file_contents = file.read()

    # Add an initial prompt alongside the file contents
    initial_prompt = "Compose a series of tweets, each with a maximum of 280 characters, narrating the progression of a soccer goal. Use the provided object detection data from three key frames to detail the attack's initiation, build-up, and final goal moment. Include the position of all the detected objects in compare to each other. In which part of the pitch ball and other player are moving and etc. Each tweet should capture a different stage of the play, highlighting player movements, ball trajectory, and field positions. The narrative should start with the attack's setup, move through the key passes and strategic plays, climax with the goal, and conclude with the immediate aftermath of the event. Ensure the language is vivid and engaging, encapsulating the excitement of the crowd and the skill on display. Do not reference frame numbers or technical metadata. Instead, translate the data into a dynamic and flowing story that brings the goal to life for your audience."
    conversation_history.append({"role": "system", "content": initial_prompt + "\n" + file_contents})

    # Set up the OpenAI API key
    openai.api_key = openai_api_key

    # Process the initial prompt and file contents
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1000
        )

        # Get the model's initial response
        ai_response = response['choices'][0]['message']['content']
        print("AI: ", ai_response)

        # Add AI response to the conversation history
        conversation_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # Enter an interactive loop for further conversation
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for termination command
        if user_input.lower() == "exit":
            print("Exiting the chat.")
            break

        # Add user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        # Send the request to OpenAI's API using ChatCompletion
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                messages=conversation_history,
                temperature=0.7,
                max_tokens=1000
            )

            # Get the model's response
            ai_response = response['choices'][0]['message']['content']
            print("AI: ", ai_response)

            # Add AI response to the conversation history
            conversation_history.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            print(f"An error occurred: {e}")

###############################################################################################################
###############################################################################################################
#Process Video Transcript with Whisper OpenAI
            
def extract_audio_from_video(video_url, output_audio_path):
    # Download the video
    download_command = ["ffmpeg", "-i", video_url, "-c", "copy", "downloaded_video.mp4"]
    subprocess.run(download_command, check=True)

    # Extract audio from the downloaded video
    extract_command = ["ffmpeg", "-i", "downloaded_video.mp4", "-vn", "-acodec", "libmp3lame", output_audio_path]
    subprocess.run(extract_command, check=True)

def calculate_similarity(segment1, segment2):
    vectorizer = TfidfVectorizer().fit([segment1, segment2])
    tfidf_matrix = vectorizer.transform([segment1, segment2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim[0][0]

def remove_overlap(transcripts, overlap_threshold=0.5, sample_length=100):
    cleaned_transcripts = [transcripts[0]]
    for i in range(1, len(transcripts)):
        prev_end = cleaned_transcripts[-1][-sample_length:]
        curr_start = transcripts[i][:sample_length]

        if calculate_similarity(prev_end, curr_start) > overlap_threshold:
            overlap_point = curr_start in prev_end
            cleaned_transcripts.append(transcripts[i][overlap_point:])
        else:
            cleaned_transcripts.append(transcripts[i])

    return cleaned_transcripts

def process_video_transcript(video_url):
    # Path for the extracted audio file
    extracted_audio_path = "/home/mehdihou/D1/02_Thesis/whisper_audio/extracted_audio.mp3"

    # Extract audio from the video
    extract_audio_from_video(video_url, extracted_audio_path)

    model = whisper.load_model("base")
    sample_rate = 16000

    # Load the entire audio file
    audio = whisper.load_audio(extracted_audio_path)
    audio_length_seconds = len(audio) / sample_rate
    segment_size = 30  # seconds
    overlap = 5  # seconds for overlap

    segment_starts = [i * (segment_size - overlap) * sample_rate for i in range(math.ceil(audio_length_seconds / (segment_size - overlap)))]
    transcripts = []

    for i, start in enumerate(segment_starts):
        end = min(start + segment_size * sample_rate, len(audio))
        segment_audio = audio[start:end]

        if len(segment_audio) < segment_size * sample_rate:
            padding = np.zeros((segment_size * sample_rate - len(segment_audio),))
            segment_audio = np.concatenate((segment_audio, padding))

        segment_audio = torch.FloatTensor(segment_audio)
        mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)

        if i == 0:
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions()
        try:
            result = whisper.decode(model, mel, options)
            transcripts.append(result.text)
        except Exception as e:
            print(f"Error processing segment {i+1}: {e}")

    cleaned_transcripts = remove_overlap(transcripts)

    final_transcript = " ".join(cleaned_transcripts)
    return final_transcript, extracted_audio_path
    
###############################################################################################
###############################################################################################
# Process the Root Mean Error of the Audio of the video clip to determine the Loudiness in each frame.

def RME_audio(audio_path):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_path)

    # Calculate RMS
    rms = librosa.feature.rms(y=audio_data)

    # Flatten the RMS values to a 1D array
    rms_values = rms.flatten()
    
    # Adjust NumPy print options to display all elements
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    
    # Convert the array to a string
    rms_values_str = np.array2string(rms_values, separator=', ')

    return rms_values_str



if __name__ == "__main__":

    video_path = "https://api.forzify.com/eliteserien/playlist.m3u8/6942:1390000:1425000/Manifest.m3u8"

    processed_video_path = process_video(video_path)

    #Fetchoing the Audio Transcript of the Video
    final_transcript, audio_path = process_video_transcript(video_path)
    print(f"The Transcript = {final_transcript} \n From the {video_path}")

    #Analyzing the RME of the Audio with Librosa library

    rms_values_str = RME_audio(audio_path)

    print(f'The RME value of the the Audio of of the video clip is = \n {rms_values_str}')

    if processed_video_path:

        # Reduce the FPS of the processed video
        reduced_fps_path = reduce_fps(processed_video_path)

        if reduced_fps_path:
            # Unpack the returned values from predict_frame_type
            frame_type, output_path = predict_frame_type(reduced_fps_path)

            detected_objects = detect_yolo(output_path)

            segmented_objects = segment_yolo(output_path)

            tracked_objects = track_yolo(output_path)

            
            ret = {}

            ret['detected_objects'] = [y.dict() for y in detected_objects]
            ret['segmented_objects'] = [y.dict() for y in segmented_objects]
            ret['tracking'] = [y.dict() for y in tracked_objects]

            json_output = json.dumps(ret, indent=4)  # indent for pretty printing

            # Write to a json file
            object_detected_file_path = '/home/mehdihou/D1/02_Thesis/objects.json'
            with open(object_detected_file_path, 'w') as file:
                file.write(json_output)

            # Write the SmartCombo to a text file
            output_txt_path = '/home/mehdihou/D1/02_Thesis/output_file.txt'
            result = reduce_floats_and_convert_to_txt(object_detected_file_path, output_txt_path)

            # Define your metadata
            metadata = '\'2023-10-29 51:00\',\'{"team": {"type": "team", "value": "Everton"}, "action": "goal", "scorer": {"type": "player", "value": "Dominic Calvert-Lewin"},"shot type": {"type": "goal shot type", "value": "standard"}}\''

            # Temporarily store the current contents of the file
            with open(output_txt_path, 'r') as file:
                current_contents = file.read()

            # Write metadata and then the original contents back to the file
            with open(output_txt_path, 'w') as file:
                file.write("Metadata of the game:\n")
                file.write(metadata + '\n\n')
                file.write("Automatic Speech Recognition of the clip: (Needed to be translate to ENGLISH)\n")
                file.write(final_transcript + '\n\n')
                file.write("Root Mean Error of the audio of the clip. RMS is a measure of the average power of the audio signal: \n")
                file.write(rms_values_str + '\n\n')
                file.write(current_contents)

            # Run the filter function to remove frames with low detection from text file
            filter_frames(output_txt_path, 6)

            # Your OpenAI API key
            openai_api_key = openai_api_key
            # Path to your text file
            text_file_path = output_txt_path
            chat_with_openai(openai_api_key, text_file_path)

    
