class SpeechRecognizer:

    def __init__(self, model_path, lm_path, context_secs=10):
        self.audio_listener = AudioListener(sample_rate=8000)
        self.recognition_model = torch.jit.load(model_path)
        self.recognition_model.eval().to('cpu')
        self.feature_extractor = get_featurizer(8000)
        self.audio_buffer = list()
        self.hidden_state = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.beam_output = ""
        self.model_output = None
        self.beam_decoder = BeamSearchDecoder(beam_width=100, lm_path=lm_path)
        self.context_length_frames = context_secs * 50
        self.is_active = False

    def save_audio(self, audio_data, filename="audio_temp"):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio_listener.audio_interface.get_sample_size(pyaudio.paInt16))
            wf.setframerate(8000)
            wf.writeframes(b"".join(audio_data))
        return filename

    def transcribe(self, audio):
        with torch.no_grad():
            filename = self.save_audio(audio)
            waveform, _ = torchaudio.load(filename)
            log_mel_spectrogram = self.feature_extractor(waveform).unsqueeze(1)
            model_output, self.hidden_state = self.recognition_model(log_mel_spectrogram, self.hidden_state)
            model_output = torch.nn.functional.softmax(model_output, dim=2)
            model_output = model_output.transpose(0, 1)
            self.model_output = model_output if self.model_output is None else torch.cat((self.model_output, model_output), dim=1)
            transcription = self.beam_decoder(self.model_output)
            context_duration = self.model_output.shape[1] / 50
            if self.model_output.shape[1] > self.context_length_frames:
                self.model_output = None
            return transcription, context_duration

    def process_audio(self, callback):
        while True:
            if len(self.audio_buffer) < 5:
                continue
            else:
                audio_data = self.audio_buffer.copy()
                self.audio_buffer.clear()
                callback(self.transcribe(audio_data))
            time.sleep(0.05)

    def start(self, callback):
        self.audio_listener.start(self.audio_buffer)
        inference_thread = threading.Thread(target=self.process_audio, args=(callback,), daemon=True)
        inference_thread.start()
