# llama-cpp-rs

Higher level API for the llama-cpp-sys library here: https://github.com/shadowmint/llama-cpp-sys/

A full end-to-end example can be found here: https://github.com/shadowmint/llama-coo-rs/tree/main/example

## Usage

See the `tests` folder, or the `example` folder for an end-to-end example.

Basic usage is:

```
use llama_cpp_rs::{
    LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams, LToken, LTokenSequence,
};

pub fn main() {
    // Setup params
    let mut config = LContextConfig::new("models/13B/model.bin");
    config.n_ctx = 512;
    config.seed = 0;

    // Load model
    let mut context = LContext::new(config).unwrap();

    // Create a generator
    let mut generator = LGenerator::new(context);
    
    // Run the generator
    let prompt = "... whatever... ";
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
        
    println!("{}", output);
}
```

Input:

```
bob is a space pilot. Alice is a potato.
This is a conversation between bob and alice:
```

Output: 

```
ALICE: I am Alice the potato!
BOB: I'm Bob the space pirate!
ALICE: Why are you a space pirate?
BOB: Because I want to fly through space!
ALICE: I want to be made into chips!
BOB: What do you want to talk about?
ALICE: Well, I am Alice the potato.
BOB: I'm Bob the space pirate!
```

For lower level interactions with the library, see the tests.

## Run example

Put your models in the `models` folder; the test expects a file in the path:

    models/13B/model.bin

You can then generate using
    
    cargo test --release --test "test_api" -- --nocapture

Or:

    cargo test --release --test "test_generator" -- --nocapture

Running outside of release mode will be significantly slower.
