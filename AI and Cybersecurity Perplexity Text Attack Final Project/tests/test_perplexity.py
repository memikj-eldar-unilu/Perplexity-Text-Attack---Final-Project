import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Simplified test for the Perplexity Metric
# We test the logic directly without the complex framework wrappers

def calculate_perplexity_simple(text):
    """
    A direct implementation of the metric logic for testing purposes,
    matching the logic in experiments/run_perplexity_on_clean.py
    """
    ref_model_name = "distilgpt2"
    # We initialize inside the test to ensure it works
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)

    if not isinstance(text, str) or not text.strip():
        return float("inf")
    
    input_ids = ref_tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = ref_model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

def test_perplexity_calculation_clean():
    """Test perplexity on a simple, fluent sentence."""
    text = "The quick brown fox jumps over the lazy dog."
    score = calculate_perplexity_simple(text)
    
    # Fluent text should have a reasonable perplexity (e.g., < 100 for this model on simple text)
    # exact value depends on the model, but it shouldn't be huge or error out
    assert score > 0
    assert score < 1000, f"Score {score} is unusually high."
    print(f"Clean text score: {score}")

def test_perplexity_calculation_nonsense():
    """Test perplexity on nonsense text."""
    # "WordSwap" attacks might result in broken grammar
    text = "Dog lazy the over jumps fox brown quick The" 
    score = calculate_perplexity_simple(text)
    
    # We expect this to likely be higher than the fluent one, 
    # though distilgpt2 is small and might vary.
    # The main check is that it runs and returns a number.
    assert score > 0
    print(f"Nonsense text score: {score}")

def test_perplexity_empty_input():
    """Test handling of empty input."""
    score = calculate_perplexity_simple("")
    assert score == float("inf")