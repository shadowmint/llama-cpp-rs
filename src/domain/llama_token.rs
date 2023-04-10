use crate::{LContext, LError, LToken};
use std::ffi::CStr;

impl LToken {
    pub fn as_string(&self, context: &LContext) -> Result<String, LError> {
        if !self.has_str_value() {
            return Err(LError::TokenizationError(
                "No string repr available for token".to_string(),
            ));
        }
        let str_value = unsafe {
            let ctx = context.native_ptr();
            let token = self.native_value();
            let c_buf = llama_cpp_sys::llama_token_to_str(ctx, token);
            let c_str: &CStr = CStr::from_ptr(c_buf);
            c_str.to_str()?.to_string()
        };
        Ok(str_value)
    }

    pub fn default_token() -> llama_cpp_sys::llama_token {
        0
    }

    pub fn has_str_value(&self) -> bool {
        match self {
            LToken::BeginningOfStream => false,
            LToken::EndOfStream => false,
            LToken::Token(_) => true,
        }
    }

    pub(crate) unsafe fn native_value(&self) -> llama_cpp_sys::llama_token {
        match self {
            LToken::BeginningOfStream => unsafe { llama_cpp_sys::llama_token_bos() },
            LToken::EndOfStream => unsafe { llama_cpp_sys::llama_token_eos() },
            LToken::Token(t) => *t,
        }
    }
}
