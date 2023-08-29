# llama-cpp-rs

Higher level API for the llama-cpp-sys library here: https://github.com/shadowmint/llama-cpp-sys/

## Usage

See the `tests` folder for examples.

Since the llama-cpp project changes constantly, this is going to be unstable forever.

## Run examples

Put your models in the `models` folder; the test expects a file in the path:

    models/13B/model.bin

You can then generate using
    
    cargo test --release --test "test_api" -- --nocapture
    cargo test --release --test "test_generator" -- --nocapture
    cargo test --release --test "test_generator_incremental" -- --nocapture
    cargo test --release --test "test_generator_incremental_instruct" -- --nocapture

Running outside of release mode will be significantly slower.

You need a lot of memory to run the examples successfully. A 64GB M1/M2 mac will work.

A 16GB MBP won't.
