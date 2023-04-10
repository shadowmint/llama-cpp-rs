use crate::LContextConfig;
use llama_cpp_sys::{llama_context_default_params, llama_context_params};
use std::path::{Path, PathBuf};

impl LContextConfig {
    pub fn new<T: AsRef<Path>>(path: T) -> LContextConfig {
        unsafe {
            LContextConfig {
                model_path: PathBuf::from(path.as_ref()),
                params: llama_context_default_params(),
                seed: 0,
                n_ctx: 512,
                n_parts: -1,
                f16_kv: true,
                use_mlock: false,
                vocab_only: false,
                logits_all: false,
                embedding: false,
            }
        }
    }

    pub(crate) unsafe fn native_ptr(&mut self) -> llama_context_params {
        self.params.seed = self.seed;
        self.params.n_ctx = self.n_ctx;
        self.params.n_parts = self.n_parts;
        self.params.f16_kv = self.f16_kv;
        self.params.use_mlock = self.use_mlock;
        self.params.vocab_only = self.vocab_only;
        self.params.logits_all = self.logits_all;
        self.params.embedding = self.embedding;
        self.params.progress_callback = None;
        self.params
    }
}
