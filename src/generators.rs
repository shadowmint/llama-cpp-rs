use crate::{LContext, LError, LSampleParams, LToken, LTokenSequence};

pub struct LGeneratorParams {
    /// Generate this number of tokens before halting
    pub generate_tokens: usize,

    /// This is the window size used for inference in the model evaluation.
    pub prediction_window_length: usize,

    /// The number of threads to process with, more is better, but only if your hardware supports it.
    pub worker_thread_count: usize,

    /// Settings to use for sampling the model
    pub sample_params: LSampleParams,
}

pub struct LGenerator {
    context: LContext,
}

impl LGenerator {
    pub fn new(context: LContext) -> LGenerator {
        LGenerator { context }
    }

    pub fn generate(&mut self, prompt: &str, params: LGeneratorParams) -> Result<String, LError> {
        // Load prompt
        let prompt_tokens = self.context.tokenize(prompt)?;
        let mut token_stream = prompt_tokens;

        // The query buffer is a window into the token stream to use for inference
        let mut gen_buffer = LTokenSequence::new();
        gen_buffer.resize(params.prediction_window_length);

        // Initialize with prompt
        self.context
            .load_prompt(&token_stream, params.worker_thread_count)?;

        for _ in 0..params.generate_tokens {
            gen_buffer.clear();
            gen_buffer.copy_trailing(&token_stream);

            // Invoke model
            self.context.step(
                &gen_buffer,
                gen_buffer.len() - 1,
                1,
                token_stream.len(),
                params.worker_thread_count,
            )?;

            // Sample result
            let token = self
                .context
                .sample(&gen_buffer, Some(params.sample_params))?;
            if let LToken::EndOfStream = token {
                break;
            }

            // Save token
            token_stream.push(token);
        }

        // Convert token stream back into a string
        let mut token_strings = Vec::new();
        for token in token_stream.iter() {
            if token.has_str_value() {
                token_strings.push(token.as_string(&self.context)?);
            }
        }
        Ok(token_strings.join(""))
    }
}
