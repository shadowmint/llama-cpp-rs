use llama_cpp_sys;
use llama_cpp_sys::llama_token_data;
use std::ffi::c_char;
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
    params: llama_cpp_sys::llama_context_params,
    pub seed: u32,
    pub n_ctx: i32,
    pub n_parts: i32,
    pub f16_kv: bool,
    pub use_mlock: bool,
    pub vocab_only: bool,
    pub logits_all: bool,
    pub embedding: bool,
    pub n_gpu_layers: i32,
    pub low_vram: bool,
}

/// Parameters for sampling the context
#[derive(Copy, Clone, Debug)]
pub struct LSampleParams {
    pub top_k: i32,
    pub top_p: f32,
    pub temp: f32,
    pub repeat_penalty: f32,
    pub repeat_history_length: usize,
    pub tfs_z: f32,
    pub typical_p: f32,
}

/// A context contains the loaded model
pub struct LContext {
    steps: usize,
    model: *mut llama_cpp_sys::llama_model,
    ctx: *mut llama_cpp_sys::llama_context,

    // TODO: Split this into a new file
    candidates: Vec<llama_token_data>,
    token_history: Vec<llama_cpp_sys::llama_token>,
    token_buffer: Vec<c_char>,
}

/// A text sequence is represented as a sequence of tokens for inference.
/// A `Context` can convert a token into the associated text sequence.
#[derive(Clone)]
pub struct LToken(llama_cpp_sys::llama_token);

/// A set of tokens representing a block of text.
#[derive(Clone)]
pub struct LTokenSequence {
    tokens: Vec<llama_cpp_sys::llama_token>,
}
