use crate::domain::LTokenSequence;
use crate::{LContext, LContextConfig, LError, LSampleParams, LToken};
use llama_cpp_sys::{llama_context, llama_init_from_file, llama_tokenize};
use std::ffi::CString;

impl LContext {
    pub fn new(mut config: LContextConfig) -> Result<LContext, LError> {
        let model_path = config.model_path.to_string_lossy();
        let model_path_c = CString::new(model_path.as_ref())?;
        let context = unsafe {
            let params = config.native_ptr();
            let ctx = llama_init_from_file(model_path_c.as_ptr(), params);
            LContext { ctx, steps: 0 }
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
            let token_count = llama_tokenize(
                ctx,
                value_c.as_ptr(),
                tokens_buffer_ptr,
                tokens_buffer_len,
                true,
            );
            if token_count < 0 {
                return Err(LError::TokenizationError(format!("failed to tokenize string; context returned {} tokens for a string of length {}", token_count, value.len())));
            }

            // Shrink buffer to actual token count
            tokens.resize(token_count as usize);
        };

        Ok(tokens)
    }

    /// Load a sequence of tokens into the context
    pub fn load_prompt(
        &mut self,
        prompt: &LTokenSequence,
        num_threads: usize,
    ) -> Result<(), LError> {
        self.steps = 0;
        self.step(prompt, 0, prompt.len(), 0, num_threads)
    }

    /// Step the model by generating 'num_generate' from the context given from the token sequence
    /// offset with a look-back context window of window_size, running using num_threads.
    /// Note that due to the internal implementation, regardless of the num_threads value,
    /// if BLAS is used or num_generate size is too large, the number of threads is locked to 1.
    /// If performance is terrible, try dropping the num_generate batch size down to a lower value.
    pub fn step(
        &mut self,
        seq: &LTokenSequence,
        offset: usize,
        num_generate: usize,
        window_size: usize,
        num_threads: usize,
    ) -> Result<(), LError> {
        let eval_result = unsafe {
            let ctx = self.native_ptr();
            let token_ptr = seq.native_ptr_offset(offset);
            //println!("root token: {}, count: {}", (*token_ptr), num_generate);
            llama_cpp_sys::llama_eval(
                ctx,
                token_ptr,
                num_generate as i32,
                window_size as i32,
                num_threads as i32,
            )
        };
        if eval_result != 0i32 {
            return Err(LError::ApiError(format!(
                "eval returned error code {}",
                eval_result
            )));
        }
        self.steps += 1;
        Ok(())
    }

    pub fn sample(
        &self,
        seq: &LTokenSequence,
        params: Option<LSampleParams>,
    ) -> Result<LToken, LError> {
        if self.steps == 0 {
            return Err(LError::CannotSampleBeforeInference);
        }
        let active_params = params.unwrap_or(Default::default());
        //println!("{:?}, {:?}", active_params, seq);
        let id = unsafe {
            let ctx = self.native_ptr();
            llama_cpp_sys::llama_sample_top_p_top_k(
                ctx,
                seq.native_ptr(),
                seq.len() as i32,
                active_params.top_k,
                active_params.top_p,
                active_params.temp,
                active_params.repeat_penalty,
            )
        };
        Ok(LToken::Token(id))
    }

    pub(crate) unsafe fn native_ptr(&self) -> *mut llama_context {
        self.ctx
    }
}
