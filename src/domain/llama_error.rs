use std::error::Error;
use std::ffi::NulError;
use std::fmt;
use std::fmt::Formatter;
use std::str::Utf8Error;

#[derive(Debug, Clone)]
pub enum LError {
    InvalidCString(String),
    TokenizationError(String),
    ApiError(String),

    /// Literally, if you call sample before running .step(), you're sampling garbage.
    /// Refuse to sample and raise this error if someone tries to sample before the model
    /// is ready.
    CannotSampleBeforeInference,

    /// If you try to do something that will not fix in the buffer you've allocated.
    OutOfBufferSpace(String),
}

impl Error for LError {}

impl fmt::Display for LError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<Utf8Error> for LError {
    fn from(value: Utf8Error) -> Self {
        LError::InvalidCString(format!("{:?}", value))
    }
}

impl From<NulError> for LError {
    fn from(value: NulError) -> Self {
        LError::InvalidCString(format!("{:?}", value))
    }
}
