use llama_cpp_rs::{LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams, LToken, LTokenSequence};
use regex::Regex;
use std::env;
use std::io::Write;

#[test]
pub fn main() {
    // Setup params
    env::set_var("LLAMA_METAL_KERNEL", "models/ggml-metal.metal");
    let mut config = LContextConfig::new("models/phind-codellama-34b-v2.Q5_K_M.gguf");
    config.n_ctx = 1024;
    config.seed = 2133;
    config.n_gpu_layers = 32;

    // Load model
    let mut context = LContext::new(config).unwrap();

    // Run the generator
    let prompt = "### System Prompt
You are an intelligent programming assistant.

### User Message
Implement a linked list in rust.

### Assistant
...";
    println!("{}", prompt);

    let expected_pattern = Regex::new("(?s).*```.*```.*").unwrap();

    let mut generator = LGenerator::new(context);
    let output = generator
        .generate_incremental(
            prompt,
            LGeneratorParams {
                worker_thread_count: 8,
                generate_tokens: 1024,
                sample_params: LSampleParams {
                    top_p: 0.95f32,
                    temp: 0.7f32,
                    repeat_penalty: 1f32,
                    ..Default::default()
                },
            },
            |generated| {
                print!("{}", generated[generated.len() - 1]);
                std::io::stdout().flush().unwrap();

                // Continue generating until we match an expected pattern or hit the limit
                let full_partial = generated.join("");
                !expected_pattern.is_match(&full_partial)
            },
        )
        .unwrap();
    assert!(!output.is_empty());
    println!("{}", output);
}
