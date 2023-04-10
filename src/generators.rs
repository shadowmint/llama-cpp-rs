use crate::{LContext, LError, LSampleParams, LToken, LTokenSequence};

pub struct LGeneratorParams {
    /// Generate this number of tokens before halting
    pub generate_tokens: usize,

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

    fn generate_no_op(_value: &[String]) -> bool {
        true
    }

    pub fn generate(&mut self, prompt: &str, params: LGeneratorParams) -> Result<String, LError> {
        self.generate_internal(prompt, params, LGenerator::generate_no_op)
    }

    pub fn generate_incremental(&mut self, prompt: &str, params: LGeneratorParams, callback: impl Fn(&[String]) -> bool) -> Result<String, LError> {
        self.generate_internal(prompt, params, callback)
    }

    pub fn generate_internal(&mut self, prompt: &str, params: LGeneratorParams, callback: impl Fn(&[String]) -> bool) -> Result<String, LError> {
        // Load prompt
        let prompt_tokens = self.context.tokenize(prompt)?;
        let mut token_stream = prompt_tokens;

        // The query buffer is a window into the token stream to use for inference
        let mut gen_buffer = LTokenSequence::new(&self.context);
        gen_buffer.resize(1); // Always generate a single new token per round

        // Initialize with prompt
        self.context.load_prompt(&token_stream, params.worker_thread_count)?;

        let mut token_strings = Vec::new();
        for _ in 0..params.generate_tokens {
            gen_buffer.clear();
            gen_buffer.copy_trailing(&token_stream);

            // Invoke model
            self.context.step(&gen_buffer, params.worker_thread_count)?;

            // Sample result
            let token = self.context.sample(Some(params.sample_params))?;
            if let LToken::EndOfStream = token {
                break;
            }

            // Save token
            token_stream.push(token.clone(), &self.context);

            // Incremental completion callback
            if token.has_str_value() {
                let token_string = token.as_string(&mut self.context)?;
                token_strings.push(token_string);

                // Halt early if the incremental thinks we're done
                if !callback(&token_strings) {
                    break;
                }
            }
        }

        // Convert token stream back into a string
        Ok(token_strings.join(""))
    }
}
