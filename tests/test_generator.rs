use llama_cpp_rs::{LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams};

#[test]
pub fn main() {
    // Setup params
    let mut config = LContextConfig::new("models/13B/model.bin");
    config.n_ctx = 512;
    config.seed = 0;

    // Load model
    let context = LContext::new(config).unwrap();

    // Run the generator
    let prompt =
        "bob is a space pilot. Alice is a potato. This is a conversation between bob and alice:";
    let mut generator = LGenerator::new(context);
    let output = generator
        .generate(
            prompt,
            LGeneratorParams {
                worker_thread_count: 8,
                prediction_window_length: 64,
                generate_tokens: 256,
                sample_params: LSampleParams {
                    top_k: 40,
                    top_p: 0.95f32,
                    temp: 0.8f32,
                    repeat_penalty: 1.1f32,
                },
            },
        )
        .unwrap();
    assert!(!output.is_empty());
    println!("{}", output);
}
