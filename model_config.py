from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

def initialize_model(model_name, language="ca"):
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")
    model.generation_config.language = language
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    model.config.use_cache = False
    return model, feature_extractor, tokenizer, processor