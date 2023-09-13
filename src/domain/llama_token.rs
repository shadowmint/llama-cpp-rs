use crate::{LContext, LError, LToken};
use llama_cpp_sys::{llama_token, llama_token_to_piece};
use std::ffi::CStr;

impl From<llama_token> for LToken {
    fn from(value: llama_token) -> Self {
        LToken { 0: value }
    }
}

impl LToken {
    pub fn as_string(&self, context: &mut LContext) -> Result<String, LError> {
        if !self.has_str_value(context) {
            return Err(LError::TokenizationError("No string repr available for token".to_string()));
        }
        let str_value = unsafe {
            let ctx = context.native_ptr();
            let token = self.native_value();

            context.token_buffer.fill(0);
            let piece_length = llama_token_to_piece(ctx, token, context.token_buffer.as_mut_ptr(), context.token_buffer.len() as i32);
            if piece_length < 0 {
                return Err(LError::InvalidCString(format!(
                    "Unable to render token {}: llama_token_to_piece() returned -1",
                    token,
                )));
            }
            let c_str = CStr::from_ptr(context.token_buffer.as_ptr());
            match c_str.to_str() {
                Ok(str_slice) => str_slice.to_string(),
                Err(failure) => {
                    return Err(LError::InvalidCString(format!(
                        "Unable to render token {} bytes {:?} as string: {}",
                        token,
                        &context.token_buffer[0..piece_length as usize]
                            .iter()
                            .map(|i| *i as u8)
                            .collect::<Vec<u8>>(),
                        failure
                    )));
                }
            }
        };
        Ok(str_value)
    }

    pub fn default_token() -> llama_cpp_sys::llama_token {
        0
    }

    pub fn has_str_value(&self, context: &LContext) -> bool {
        !self.is_end_of_stream(context) && !self.is_beginning_of_stream(context)
    }

    pub fn is_end_of_stream(&self, context: &LContext) -> bool {
        self.0 == unsafe { llama_cpp_sys::llama_token_eos(context.native_ptr()) }
    }

    pub fn is_beginning_of_stream(&self, context: &LContext) -> bool {
        self.0 == unsafe { llama_cpp_sys::llama_token_bos(context.native_ptr()) }
    }

    pub(crate) unsafe fn native_value(&self) -> llama_cpp_sys::llama_token {
        self.0
    }
}
