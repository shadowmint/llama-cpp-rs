use llama_cpp_rs::{LContext, LContextConfig, LToken, LTokenSequence};
use std::io::Write;

#[test]
pub fn main() {
    let num_predict = 128;
    let sample_worker_threads = 8;
    let sample_window_size = 64;
    let context_length: usize = 512;

    // Setup params
    let mut config = LContextConfig::new("models/13B/model.bin");
    config.n_ctx = context_length as i32;
    config.seed = 0;

    // Load model
    let mut context = LContext::new(config).unwrap();

    // Load prompt
    let prompt_tokens = context.tokenize("bob is a space pilot. Alice is a potato. This is a conversation between bob and alice:").unwrap();
    let mut token_stream = prompt_tokens;
    assert!(num_predict + token_stream.len() < context_length);

    // Print prompt
    println!("prompt:");
    for token in token_stream.iter() {
        if token.has_str_value() {
            let token_str = token.as_string(&context).unwrap();
            print!("{}", token_str);
        }
    }

    // The query buffer is a window into the token stream to use for inference
    println!("\n\ngenerating...\n");
    let mut gen_buffer = LTokenSequence::new();
    gen_buffer.resize(sample_window_size);

    // Initialize with prompt
    context
        .load_prompt(&token_stream, sample_worker_threads)
        .unwrap();

    for _ in 0..num_predict {
        // Build a query buffer
        gen_buffer.clear();
        gen_buffer.copy_trailing(&token_stream);

        // Invoke model
        context
            .step(
                &gen_buffer,
                gen_buffer.len() - 1,
                1,
                token_stream.len(),
                sample_worker_threads,
            )
            .unwrap();

        // Sample result
        let token = context.sample(&gen_buffer, None).unwrap();

        if let LToken::EndOfStream = token {
            println!("Received end of stream");
            break;
        }

        // Save token
        token_stream.push(token.clone());

        // Print incremental output to stdout
        if let Ok(token_str) = token.as_string(&context) {
            print!("{}", token_str);
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\n");
}
