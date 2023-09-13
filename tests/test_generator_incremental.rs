use llama_cpp_rs::{LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams};
use regex::Regex;
use std::io::Write;

#[test]
pub fn main() {
    // Setup params
    let mut config = LContextConfig::new("models/model.gguf");
    config.n_ctx = 1024;
    config.seed = 2133;
    config.n_gpu_layers = 32;

    // Load model
    let context = LContext::new(config).unwrap();

    // Run the generator
    let prompt = "<s>[INST]Replace IMPLEMENT_ME with a real implementation in javascript ```/** Prints o number of times equal to n */\nfunction print_o(int n) { IMPLEMENT_ME };```[/INST]";
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
