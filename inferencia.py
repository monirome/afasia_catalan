from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

audio_path = "/Users/monicaromero/PycharmProjects/whisper_infer/cv-corpus-16.1-delta-2023-12-06/pruebas/common_voice_es_38506157.wav"

audio_input = processor(audio_path, return_tensors="pt")
input_values = audio_input.input_values
attention_mask = audio_input.attention_mask
language_id = processor.tokenizer.get_language_id("ca")


with torch.no_grad():
    predicted_ids = model.generate(
        input_values=input_values,
        attention_mask=attention_mask,
        forced_bos_token_id=language_id
    )
    # logits = model(input_values=input_values, attention_mask=attention_mask, forced_decoder_ids=forced_decoder_ids).logits

# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)


print(transcription[0])
