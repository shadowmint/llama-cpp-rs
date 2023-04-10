use crate::domain::LTokenSequence;
use crate::{LContext, LToken};
use std::fmt::{Debug, Formatter};
use std::ptr;

impl LTokenSequence {
    pub fn new(ctx: &LContext) -> LTokenSequence {
        LTokenSequence {
            tokens: Vec::new(),
            end_of_stream: unsafe { llama_cpp_sys::llama_token_eos(ctx.native_ptr()) },
        }
    }

    /// Increase the manifest allocation of tokens in this sequence to length.
    pub fn resize(&mut self, length: usize) {
        if self.tokens.len() > length {
            self.tokens.truncate(length);
        } else {
            let missing_capacity = length - self.tokens.len();
            for _ in 0..missing_capacity {
                self.tokens.push(LToken::default_token());
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = LToken> + '_> {
        Box::new(self.tokens.iter().map(move |t| {
            if *t == self.end_of_stream {
                LToken::EndOfStream
            } else {
                LToken::Token(*t)
            }
        }))
    }

    pub fn push(&mut self, token: LToken, context: &LContext) {
        let value = unsafe { token.native_value(context) };
        self.tokens.push(value);
    }

    pub fn clear(&mut self) {
        if self.is_empty() {
            return;
        }
        unsafe {
            ptr::write_bytes(self.tokens.as_mut_ptr(), 0, self.len());
        }
    }

    /// Copy the N values from 'from' into the last N values of this buffer.
    /// ie. If this buffer is length 8, and from is length 2, you expect
    /// this[6] = from[0], this[7] = from[1]. The remaining values are unchanged,
    /// so you probably want to call clear() first.
    pub fn copy_trailing(&mut self, from: &LTokenSequence) {
        let from_len = from.len();
        let self_len = self.len();
        if from_len < self_len {
            let end_partial = &mut self.tokens[self_len - from_len..];
            unsafe {
                let from_ptr = from.native_ptr_slice();
                end_partial.copy_from_slice(from_ptr);
            }
        } else {
            let from_ptr = unsafe { from.native_ptr_slice() };
            self.tokens.copy_from_slice(&from_ptr[(from_len - self_len)..])
        }
    }

    pub(crate) unsafe fn native_ptr(&self) -> *const llama_cpp_sys::llama_token {
        self.tokens.as_ptr()
    }

    pub(crate) unsafe fn native_mut_ptr(&mut self) -> *mut llama_cpp_sys::llama_token {
        self.tokens.as_mut_ptr()
    }

    pub(crate) unsafe fn native_ptr_slice(&self) -> &[llama_cpp_sys::llama_token] {
        self.tokens.as_ref()
    }
}

impl Debug for LTokenSequence {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let id_stream: Vec<isize> = self.tokens.iter().map(|f| *f as isize).collect();
        write!(f, "{:?}", id_stream)
    }
}
