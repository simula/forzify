import os
import csv
import whisper

def process_audio_file(file_path, model, model_size):
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)

    # Use different mel spectrogram settings based on the model size
    if model_size == 'large-v3':
        mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(model.device)
    else:
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language
    _, probs = model.detect_language(mel)
    top_languages = sorted(probs, key=probs.get, reverse=True)[:3]
    lang_probs = [(lang, probs[lang]) for lang in top_languages]

    return lang_probs


def process_audio_files(folder_path, model_sizes):
    models = {size: whisper.load_model(size) for size in model_sizes}
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            result = {'file': filename}

            for size, model in models.items():
                lang_probs = process_audio_file(file_path, model, size)
                for i, (lang, prob) in enumerate(lang_probs):
                    result[f'{size}_lang_{i+1}'] = lang
                    result[f'{size}_prob_{i+1}'] = prob

            results.append(result)

    return results


def save_to_csv(results, output_file, model_sizes):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ['File']
        for size in model_sizes:
            for i in range(1, 4):
                header += [f'{size} - detected Lang {i}', f'{size} - Probability {i}']
        writer.writerow(header)

        for result in results:
            row = [result['file']]
            for size in model_sizes:
                for i in range(1, 4):
                    row.append(result.get(f'{size}_lang_{i}', 'N/A'))
                    row.append(result.get(f'{size}_prob_{i}', 0))
            writer.writerow(row)

def main():
    folder_path = '/home/mehdihou/D1/whisper_eval/audios'  # Replace with your folder path
    output_file = '/home/mehdihou/D1/whisper_eval/results/output.csv'           # Output CSV file name
    model_sizes = ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3']       # Add or remove model sizes as needed

    results = process_audio_files(folder_path, model_sizes)
    save_to_csv(results, output_file, model_sizes)

if __name__ == '__main__':
    main()
