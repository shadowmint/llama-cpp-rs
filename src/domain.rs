use llama_sys;
use std::path::PathBuf;

mod llama_context;
mod llama_context_config;
mod llama_error;
mod llama_sample_params;
mod llama_token;
mod llama_token_sequence;

pub use self::llama_error::LError;

/// You construct a context using these parameters
pub struct LContextConfig {
    model_path: PathBuf,
    params: llama_sys::llama_context_params,
    pub seed: i32,
    pub n_ctx: i32,
    pub n_parts: i32,
    pub f16_kv: bool,
    pub use_mlock: bool,
    pub vocab_only: bool,
    pub logits_all: bool,
    pub embedding: bool,
}

/// Parameters for sampling the context
#[derive(Copy, Clone, Debug)]
pub struct LSampleParams {
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub repeat_penalty: f32,
}

/// A context contains the loaded model
pub struct LContext {
    steps: usize,
    ctx: *mut llama_sys::llama_context,
}

/// A text sequence is represented as a sequence of tokens for inference.
/// A `Context` can convert a token into the associated text sequence.
#[derive(Clone)]
pub enum LToken {
    BeginningOfStream,
    EndOfStream,
    Token(llama_sys::llama_token),
}

/// A set of tokens representing a block of text.
#[derive(Clone)]
pub struct LTokenSequence {
    tokens: Vec<llama_sys::llama_token>,
}
