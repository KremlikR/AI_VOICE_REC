from script.SpeechRecognizer import SpeechRecognizer


class SpeechRecognitionHandler:

    def __init__(self):
        self.complete_transcription = ""
        self.current_transcription = ""

    def __call__(self, result):
        transcription, context_length = result
        self.current_transcription = transcription
        combined_transcription = " ".join(self.complete_transcription.split() + transcription.split())
        print(combined_transcription)
        if context_length > 10:
            self.complete_transcription = combined_transcription

if __name__ == "__main__":
    import argparse
    import threading
 

    parser = argparse.ArgumentParser(description="Demonstrate the speech recognition engine in the terminal.")
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the optimized model file. Use optimize_graph.py to create it.')
    parser.add_argument('--ken_lm_file', type=str, required=False,
                        help='Path to the KenLM language model file for decoding.')

    args = parser.parse_args()
    recognition_engine = SpeechRecognizer(args.model_file, args.ken_lm_file)
    action_handler = SpeechRecognitionHandler()

    recognition_engine.start(action_handler)
    threading.Event().wait()
