import os
import csv
import whisper

def transcribe_audio(file_path, model, model_size):
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # Use different mel spectrogram settings based on the model size
    if model_size == 'large-v3':
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    else:
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcribe audio
    options = whisper.DecodingOptions()
    result = model.decode(mel, options)
    transcription = result.text

    return transcription

def process_audio_files(folder_path, model_sizes):
    models = {size: whisper.load_model(size) for size in model_sizes}
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            result = {'file': filename}

            for size, model in models.items():
                transcription = transcribe_audio(file_path, model, size)
                # Enclose the transcription in quotes
                result[f'{size}_transcription'] = f'"{transcription}"'

            results.append(result)

    return results

def save_to_csv(results, output_file, model_sizes):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        header = ['File']
        for size in model_sizes:
            header.append(f'{size} - Transcription')
        writer.writerow(header)

        for result in results:
            row = [result['file']]
            for size in model_sizes:
                row.append(result.get(f'{size}_transcription', 'N/A'))
            writer.writerow(row)

def main():
    folder_path = '/home/mehdihou/D1/whisper_eval/audios'  # Replace with your folder path
    output_file = '/home/mehdihou/D1/whisper_eval/results/transcriptions.csv' # Output CSV file name
    model_sizes = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']  # Add or remove model sizes as needed

    results = process_audio_files(folder_path, model_sizes)
    save_to_csv(results, output_file, model_sizes)

if __name__ == '__main__':
    main()
