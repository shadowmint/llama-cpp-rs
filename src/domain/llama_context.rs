use crate::domain::LTokenSequence;
use crate::{LContext, LContextConfig, LError, LSampleParams, LToken};
use llama_cpp_sys::{
    llama_backend_free, llama_context, llama_free, llama_free_model, llama_get_kv_cache_token_count, llama_get_logits, llama_load_model_from_file,
    llama_n_ctx, llama_n_vocab, llama_new_context_with_model, llama_sample_repetition_penalty, llama_sample_tail_free, llama_sample_temperature,
    llama_sample_token, llama_sample_top_k, llama_sample_top_p, llama_sample_typical, llama_token_data, llama_token_data_array, llama_tokenize,
};
use std::ffi::CString;

impl LContext {
    pub fn new(mut config: LContextConfig) -> Result<LContext, LError> {
        let model_path = config.model_path.to_string_lossy();
        let model_path_c = CString::new(model_path.as_ref())?;
        let context = unsafe {
            let params = config.native_ptr();
            let model = llama_load_model_from_file(model_path_c.as_ptr(), params);
            let ctx = llama_new_context_with_model(model, params);
            LContext {
                model,
                ctx,
                steps: 0,
                candidates: Vec::new(),
                token_history: Vec::new(),
                token_buffer: vec![0; 2048],
            }
        };
        Ok(context)
    }

    /// Convert a string into a token sequence object.
    pub fn tokenize(&self, value: &str) -> Result<LTokenSequence, LError> {
        let mut tokens = LTokenSequence::new();

        // We need to allocate enough space for the entire value to fit into the token space.
        // Since a token can be 1..n in length, we allocate the maximum possible length and
        // shrink afterwards.
        tokens.resize(value.bytes().len() + 1);

        // Use the context to generate tokens for the input sequence.
        unsafe {
            let ctx = self.native_ptr();
            let value_c = CString::new(value)?;
            let tokens_buffer_len = tokens.len() as i32;
            let tokens_buffer_ptr = tokens.native_mut_ptr();
            let token_count = llama_tokenize(ctx, value_c.as_ptr(), tokens_buffer_ptr, tokens_buffer_len, true);
            if token_count < 0 {
                return Err(LError::TokenizationError(format!(
                    "failed to tokenize string; context returned {} tokens for a string of length {}",
                    token_count,
                    value.len()
                )));
            }

            // Shrink buffer to actual token count
            tokens.resize(token_count as usize);
        };

        Ok(tokens)
    }

    /// Load a sequence of tokens into the context
    pub fn load_prompt(&mut self, prompt: &LTokenSequence, num_threads: usize) -> Result<(), LError> {
        self.steps = 0;
        self.step(prompt, num_threads)
    }

    /// Step the model, generating a single new token given the new input tokens from input.
    pub fn step(&mut self, input: &LTokenSequence, num_threads: usize) -> Result<(), LError> {
        let eval_result = unsafe {
            let existing_token_count = llama_get_kv_cache_token_count(self.native_ptr());
            let input_tokens = input.native_ptr();
            let input_token_count = input.len();
            let max_length = llama_n_ctx(self.native_ptr());
            if max_length <= existing_token_count + (input_token_count as i32) {
                return Err(LError::OutOfBufferSpace(format!(
                    "You've requested {} additional tokens to a context that is already {} in size with a max size of {}",
                    input_token_count, existing_token_count, max_length
                )));
            }
            llama_cpp_sys::llama_eval(
                self.native_ptr(),
                input_tokens,
                input_token_count as i32,
                existing_token_count,
                num_threads as i32,
            )
        };
        if eval_result != 0i32 {
            return Err(LError::ApiError(format!("eval returned error code {}", eval_result)));
        }
        self.steps += 1;
        Ok(())
    }

    pub fn sample(&mut self, params: Option<LSampleParams>) -> Result<LToken, LError> {
        if self.steps == 0 {
            return Err(LError::CannotSampleBeforeInference);
        }
        let active_params = params.unwrap_or(Default::default());
        let id = unsafe {
            let logits = llama_get_logits(self.ctx);
            let n_vocab = llama_n_vocab(self.ctx);

            self.candidates.clear();
            for token_id in 0..n_vocab {
                self.candidates.push(llama_token_data {
                    id: token_id,
                    logit: *logits.offset(token_id as isize),
                    p: 0f32,
                });
            }

            let mut candidates_p = llama_token_data_array {
                data: self.candidates.as_mut_ptr(),
                size: self.candidates.len(),
                sorted: false,
            };

            let token_history_max_length = active_params.repeat_history_length;
            let repeat_penalty_scan_length = token_history_max_length.min(self.token_history.len());
            llama_sample_repetition_penalty(
                self.ctx,
                &mut candidates_p,
                self.token_history.as_ptr(),
                repeat_penalty_scan_length,
                active_params.repeat_penalty,
            );

            let ctx = self.native_ptr();
            llama_sample_top_k(ctx, &mut candidates_p, active_params.top_k, 0);
            llama_sample_tail_free(ctx, &mut candidates_p, active_params.tfs_z, 0);
            llama_sample_typical(ctx, &mut candidates_p, active_params.typical_p, 0);
            llama_sample_top_p(ctx, &mut candidates_p, active_params.top_p, 0);
            llama_sample_temperature(ctx, &mut candidates_p, active_params.temp);
            llama_sample_token(ctx, &mut candidates_p)
        };

        self.update_token_history(id, active_params);
        Ok(LToken::from(id))
    }

    fn update_token_history(&mut self, id: llama_cpp_sys::llama_token, params: LSampleParams) {
        self.token_history.push(id);
        if self.token_history.len() > params.repeat_history_length {
            self.token_history.remove(0);
        }
    }

    pub(crate) unsafe fn native_ptr(&self) -> *mut llama_context {
        self.ctx
    }
}

impl Drop for LContext {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.ctx);
            llama_free_model(self.model);
            llama_backend_free();
        }
    }
}
