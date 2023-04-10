use llama_cpp_rs::{LContext, LContextConfig, LSampleParams, LToken, LTokenSequence};
use std::env;
use std::io::Write;
#[test]
pub fn main() {
    let num_predict = 512;
    let sample_worker_threads = 8;
    let context_length: usize = 1024;

    // Setup params
    env::set_var("LLAMA_METAL_KERNEL", "models/ggml-metal.metal");
    let mut config = LContextConfig::new("models/codellama-13b-instruct.Q5_K_M.gguf");
    config.n_ctx = context_length as i32;
    config.n_gpu_layers = 32;
    config.seed = 0;

    // Load model
    let mut context = LContext::new(config).unwrap();

    // Load prompt
    let prompt_tokens = context
        .tokenize("<s>[INST]How would you implement a function that multiplies two matrices together in typescript?[/INST]")
        .unwrap();
    let mut token_stream = prompt_tokens;
    assert!(num_predict + token_stream.len() < context_length);

    // Print prompt
    println!("prompt:");
    for token in token_stream.iter() {
        if token.has_str_value() {
            let token_str = token.as_string(&mut context).unwrap();
            print!("{}", token_str);
        }
    }

    // The query buffer is a window into the token stream to use for inference
    println!("\n\ngenerating...\n");
    let mut gen_buffer = LTokenSequence::new(&context);
    gen_buffer.resize(1); // Add one token at a time during the generate phase

    // Initialize with prompt
    context.load_prompt(&token_stream, sample_worker_threads).unwrap();

    for _ in 0..num_predict {
        // Build a query buffer
        gen_buffer.clear();
        gen_buffer.copy_trailing(&token_stream);

        // Invoke model
        context.step(&gen_buffer, sample_worker_threads).unwrap();

        // Sample result
        let token = context
            .sample(Some(LSampleParams {
                repeat_penalty: 1.1f32,
                temp: 1f32,
                repeat_history_length: 64,
                top_p: 1.1f32,
                ..Default::default()
            }))
            .unwrap();

        if let LToken::EndOfStream = token {
            println!("Received end of stream");
            break;
        }

        // Save token
        token_stream.push(token.clone(), &context);

        // Print incremental output to stdout
        if let Ok(token_str) = token.as_string(&mut context) {
            print!("{}", token_str);
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\n");
}
