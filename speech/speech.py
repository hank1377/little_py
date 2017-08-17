import speech_recognition

r = speech_recognition.Recognizer()

with speech_recognition.Microphone() as source:
	r.adjust_for_ambient_noise(source)
	audio = r.listen(source)
	r.recognize_google(audio, language='zh-TW')