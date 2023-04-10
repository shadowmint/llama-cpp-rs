use llama_cpp_rs::{LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams};
use std::env;

#[test]
pub fn main() {
    // Setup params
    env::set_var("LLAMA_METAL_KERNEL", "models/ggml-metal.metal");
    let mut config = LContextConfig::new("models/codellama-13b-instruct.Q5_K_M.gguf");
    config.n_ctx = 512;
    config.seed = 1;
    config.n_gpu_layers = 32;

    // Load model
    let context = LContext::new(config).unwrap();

    // Run the generator
    let prompt = "[INST]bob is a space pilot. Alice is a potato. This is a conversation between bob and alice:[/INST]";
    let mut generator = LGenerator::new(context);
    let output = generator
        .generate(
            prompt,
            LGeneratorParams {
                worker_thread_count: 8,
                generate_tokens: 256,
                sample_params: LSampleParams {
                    top_k: 40,
                    top_p: 0.95f32,
                    temp: 0.8f32,
                    repeat_penalty: 1.1f32,
                    ..Default::default()
                },
            },
        )
        .unwrap();
    assert!(!output.is_empty());
    println!("{}", output);
}
